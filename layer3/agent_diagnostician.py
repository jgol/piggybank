"""L3 Diagnostician agent: three-layer strategy assessment.

Produces a StrategyAssessment with:
  Layer 1: Technical gap (proxy vs QC divergence)
  Layer 2: Behavioral profile (loss streaks, regime decomposition)
  Layer 3: Tradeability (can a human execute this?)

Context isolation: receives DiagnosticianInput (no raw tree s-expressions).
Numerical claims are cross-checked by the supervisor.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from layer3.agents import task_sync
from layer3.config import DIAGNOSTICIAN_MODEL
from layer3.prompts_v2 import DIAGNOSTICIAN_SYSTEM
from layer3.tools import DIAGNOSTICIAN_TOOLS
from layer3.schemas import (
    DiagnosticianInput, StrategyAssessment,
    ProxyMetrics, QCMetrics, TradeRecord,
)


def assess_strategy(
    diag_input: DiagnosticianInput,
) -> tuple[Optional[StrategyAssessment], dict]:
    """Run the Diagnostician on a strategy and return structured assessment.

    Args:
        diag_input: Typed input (context isolation enforced by schema).

    Returns:
        (assessment, metadata) where assessment is a StrategyAssessment
        or None if the agent failed to produce structured output.
    """
    # Pre-compute data for tool handlers
    _proxy = diag_input.proxy_metrics.model_dump()
    _qc = diag_input.qc_metrics.model_dump()
    _trades = [t.model_dump() for t in diag_input.trade_log]
    _interp = diag_input.interpretation

    # Tool handlers — provide data when the agent asks for it
    def _read_proxy(inp: dict) -> str:
        return json.dumps(_proxy, indent=2)

    def _read_qc(inp: dict) -> str:
        return json.dumps(_qc, indent=2)

    def _read_trade_log(inp: dict) -> str:
        if not _trades:
            return json.dumps({"trades": [], "note": "No trade log available"})
        # Summarize: exit reason distribution + sample trades
        reasons = {}
        for t in _trades:
            r = t.get("reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        sample = _trades[:20]  # first 20 trades
        return json.dumps({
            "total_trades": len(_trades),
            "exit_reason_distribution": reasons,
            "sample_trades": sample,
        }, indent=2)

    def _read_interpretation(inp: dict) -> str:
        return _interp or "No interpretation available"

    def _request_additional(inp: dict) -> str:
        # For now, return a note that additional data requires supervisor dispatch
        return json.dumps({
            "status": "request_noted",
            "note": "Additional data request forwarded to supervisor. "
                    "Proceed with current data for now.",
        })

    handlers = {
        "read_proxy_metrics": _read_proxy,
        "read_qc_metrics": _read_qc,
        "read_trade_log": _read_trade_log,
        "read_strategy_interpretation": _read_interpretation,
        "request_additional_data": _request_additional,
        "submit_assessment": lambda inp: inp,  # final-answer tool
    }

    # Build user message from DiagnosticianInput
    user_msg = (
        f"Assess strategy {diag_input.strategy_id} ({diag_input.template_name}).\n\n"
        f"Use your tools to read the proxy metrics, QC metrics, trade log, and "
        f"strategy interpretation. Then submit your three-layer assessment.\n"
    )
    if diag_input.memory_context:
        user_msg += (
            f"\nRelevant context from previous batches:\n"
            f"{diag_input.memory_context}\n"
        )

    output, metadata = task_sync(
        role="diagnostician",
        user_message=user_msg,
        system_prompt=DIAGNOSTICIAN_SYSTEM,
        tools=DIAGNOSTICIAN_TOOLS,
        tool_handlers=handlers,
        model=DIAGNOSTICIAN_MODEL,
        max_turns=12,  # 4-5 tool reads + thinking + submit_assessment
    )

    if output:
        try:
            assessment = StrategyAssessment.model_validate(output)
            return assessment, metadata
        except Exception as e:
            metadata["parse_error"] = str(e)
            return None, metadata
    return None, metadata


def cross_check_assessment(
    assessment: StrategyAssessment,
    trade_log: List[TradeRecord],
    proxy_sharpe: float,
    qc_sharpe: float,
) -> List[str]:
    """Deterministic cross-check of Diagnostician's numerical claims.

    Returns list of discrepancies. Empty list = all checks pass.
    """
    issues = []

    # Check gap_severity against actual Sharpe gap
    actual_gap = abs(proxy_sharpe - qc_sharpe) / max(abs(proxy_sharpe), 0.01)
    if abs(assessment.gap_severity - actual_gap) > 0.3:
        issues.append(
            f"gap_severity: claimed {assessment.gap_severity:.2f}, "
            f"actual {actual_gap:.2f} (diff {abs(assessment.gap_severity - actual_gap):.2f})"
        )

    # Check max_consecutive_losses against trade log
    if trade_log:
        max_consec = 0
        current_streak = 0
        for t in trade_log:
            if t.entry_credit < 0:  # loss (entry_credit is negative for losses in some formats)
                current_streak += 1
                max_consec = max(max_consec, current_streak)
            else:
                current_streak = 0
        # Allow ±2 tolerance (trade log may not perfectly match daily aggregation)
        if abs(assessment.max_consecutive_losses - max_consec) > 2:
            issues.append(
                f"max_consecutive_losses: claimed {assessment.max_consecutive_losses}, "
                f"computed {max_consec}"
            )

    return issues
