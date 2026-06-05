"""L3 QC Operator agent: interprets deployment errors and structures results.

The QC Operator is invoked by the deterministic scheduler (qc_scheduler.py)
when it encounters situations that regex-based error classification cannot
handle. The Operator also structures trade-level data from QC's ObjectStore.

The Operator does NOT manage scheduling, polling, or concurrency — those
are handled by qc_scheduler.py. The Operator is called for:
- Novel compile/runtime errors
- Zero-trade diagnosis
- Trade log structuring
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from layer3.agents import task_sync
from layer3.config import QC_OPERATOR_MODEL
from layer3.prompts_v2 import QC_OPERATOR_SYSTEM
from layer3.tools import QC_OPERATOR_TOOLS
from layer3.schemas import BacktestResult as BacktestResultSchema, DeploymentError


def interpret_error(error_message: str, code_snippet: str = "") -> Tuple[str, bool, str]:
    """Ask the QC Operator to interpret a novel error.

    Args:
        error_message: The raw error from QC.
        code_snippet: Relevant code context (optional).

    Returns:
        (error_type, retryable, diagnosis) where error_type is one of
        COMPILE_ERROR, RUNTIME_ERROR, TIMEOUT.
    """
    user_msg = (
        f"Interpret this QC error and classify it:\n\n"
        f"Error message:\n{error_message}\n"
    )
    if code_snippet:
        user_msg += f"\nRelevant code:\n{code_snippet}\n"
    user_msg += (
        "\nCall report_result with:\n"
        "- success: false\n"
        "- error_type: COMPILE_ERROR, RUNTIME_ERROR, or TIMEOUT\n"
        "- error_message: your diagnosis\n"
        "- retryable: true or false\n"
    )

    # Tool handlers — for error interpretation, we only need report_result
    handlers = {
        "report_result": lambda inp: inp,  # final-answer tool — input IS the output
    }

    output, metadata = task_sync(
        role="qc_operator",
        user_message=user_msg,
        system_prompt=QC_OPERATOR_SYSTEM,
        tools=[t for t in QC_OPERATOR_TOOLS if t["name"] == "report_result"],
        tool_handlers=handlers,
        model=QC_OPERATOR_MODEL,
        max_turns=3,
    )

    if output and not output.get("success", True):
        return (
            output.get("error_type", "RUNTIME_ERROR"),
            output.get("retryable", False),
            output.get("error_message", error_message),
        )
    # Fallback if LLM doesn't produce structured output
    return "RUNTIME_ERROR", True, error_message


def diagnose_zero_trades(
    strategy_id: str,
    template_name: str,
    proxy_trades: int,
    qc_log: str = "",
) -> str:
    """Ask the QC Operator why a backtest produced zero trades.

    Returns a text diagnosis.
    """
    user_msg = (
        f"Strategy {strategy_id} ({template_name}) produced ZERO trades in QC.\n"
        f"The proxy evaluator expected ~{proxy_trades} trades.\n\n"
        f"Possible causes:\n"
        f"- Entry condition never fires in QC's data (different terminal values)\n"
        f"- Date range has no trading data\n"
        f"- Option chain too thin (no contracts at target deltas)\n"
    )
    if qc_log:
        user_msg += f"\nQC debug log (last 20 lines):\n{qc_log}\n"
    user_msg += "\nCall report_result with your diagnosis."

    handlers = {
        "report_result": lambda inp: inp,
    }

    output, metadata = task_sync(
        role="qc_operator",
        user_message=user_msg,
        system_prompt=QC_OPERATOR_SYSTEM,
        tools=[t for t in QC_OPERATOR_TOOLS if t["name"] == "report_result"],
        tool_handlers=handlers,
        model=QC_OPERATOR_MODEL,
        max_turns=3,
    )

    if output:
        return output.get("error_message", "Unknown cause")
    return metadata.get("final_text", "QC Operator could not determine cause")
