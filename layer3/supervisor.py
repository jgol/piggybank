"""L3 Supervisor: orchestrates the strategy validation and refinement loop.

The supervisor is an LLM agent that plans batch evaluation, delegates to
subagents, tracks state, and learns across batches via persistent memory.
It follows the Deep Agents pattern: plan → delegate → observe → adapt.

The supervisor does NOT:
- Analyze raw QC numbers (Diagnostician's job)
- Propose tree mutations (Mutator's job)
- Interpret compile errors (QC Operator's job)

It DOES:
- Prioritize candidates (proxy Sharpe × persistence × diversity)
- Route based on Diagnostician's assessment (deploy/diagnose/modify loop)
- Cross-check Diagnostician's numerical claims against actual data
- Learn from failures to reprioritize remaining candidates
- Manage the mutation iteration cap (max 2)
- Trigger human-in-the-loop for approved strategies
- Track costs and enforce budget
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from layer3.schemas import (
    PipelineState, StrategyLedgerEntry, StrategyAssessment,
    MutationProposal, DiagnosticianInput, MutatorInput,
    ProxyMetrics, QCMetrics, TradeRecord,
)
from layer3.state import (
    save_state, update_strategy_status,
)
from layer3.memory import append as memory_append, load_context
from layer3.agent_diagnostician import assess_strategy, cross_check_assessment
from layer3.agent_mutator import propose_mutation, validate_mutation
from layer3.agent_qc_operator import interpret_error, diagnose_zero_trades


# ---------------------------------------------------------------------------
# Constants (from centralized config)
# ---------------------------------------------------------------------------

from layer3.config import (
    MAX_MUTATION_ITERATIONS, COST_BUDGET_USD,
    APPROVE_MIN_QC_SHARPE, APPROVE_MIN_QC_TRADES, APPROVE_MAX_QC_DRAWDOWN,
    COST_PER_1K_INPUT, COST_PER_1K_OUTPUT,
)


def _estimate_cost(metadata: dict) -> float:
    """Estimate LLM API cost from agent metadata (USD)."""
    model = metadata.get("model", "")
    input_t = metadata.get("input_tokens", 0)
    output_t = metadata.get("output_tokens", 0)
    cost_in = COST_PER_1K_INPUT.get(model, 0.003) * input_t / 1000
    cost_out = COST_PER_1K_OUTPUT.get(model, 0.015) * output_t / 1000
    return cost_in + cost_out


# ---------------------------------------------------------------------------
# Per-strategy processing
# ---------------------------------------------------------------------------

def process_strategy(
    state: PipelineState,
    strategy_id: str,
    strategy_data: dict,
    qc_result: dict,
    interpretation: str,
    on_approve: Optional[Callable] = None,
    on_discard: Optional[Callable] = None,
) -> str:
    """Process one strategy through the diagnose-modify loop.

    Args:
        state: Pipeline state (updated in-place).
        strategy_id: Unique strategy identifier.
        strategy_data: Dict with entry_sexpr, exit_sexpr, size_sexpr,
            delta_sexpr, template_name, condition, proxy metrics.
        qc_result: QC backtest results (sharpe, trades, win_rate, etc.).
        interpretation: Plain-language strategy description.
        on_approve: Callback when strategy is approved (triggers HITL).
        on_discard: Callback when strategy is discarded.

    Returns:
        Final status: "approved", "discarded", or "review".
    """
    entry = state.ledger.get(strategy_id)
    if entry is None:
        return "discarded"

    update_strategy_status(state, strategy_id, "analyzing")
    entry.qc_results.append(qc_result)

    # Zero-trade diagnosis: if QC produced 0 trades, ask QC Operator why
    if qc_result.get("total_trades", 0) == 0 and not qc_result.get("error"):
        try:
            zero_diagnosis = diagnose_zero_trades(
                strategy_id,
                strategy_data.get("template_name", ""),
                strategy_data.get("val_trades", 0),
            )
            qc_result["zero_trade_diagnosis"] = zero_diagnosis
        except Exception:
            qc_result["zero_trade_diagnosis"] = "QC Operator unavailable"

    # Build Diagnostician input (context isolation: no raw trees)
    proxy_metrics = ProxyMetrics(
        fold_sharpes=strategy_data.get("fold_sharpes", {}),
        mean_val_sharpe=strategy_data.get("mean_val_sharpe", 0.0),
        val_trades=strategy_data.get("val_trades", 0),
        val_win_rate=strategy_data.get("val_win_rate", 0.0),
        val_drawdown=strategy_data.get("val_drawdown", 0.0),
        pbo_score=strategy_data.get("pbo_score", 0.0),
    )
    qc_metrics = QCMetrics(
        sharpe=qc_result.get("sharpe", 0.0),
        total_trades=qc_result.get("total_trades", 0),
        win_rate=qc_result.get("win_rate", 0.0),
        max_drawdown=qc_result.get("max_drawdown", 0.0),
        net_profit=qc_result.get("net_profit", 0.0),
        total_fees=qc_result.get("total_fees", 0.0),
    )
    trade_log = [
        TradeRecord(**t) for t in qc_result.get("trade_log", [])
    ]

    memory_ctx = load_context(max_per_type=30)

    diag_input = DiagnosticianInput(
        strategy_id=strategy_id,
        template_name=strategy_data.get("template_name", ""),
        interpretation=interpretation,
        proxy_metrics=proxy_metrics,
        qc_metrics=qc_metrics,
        trade_log=trade_log,
        memory_context=memory_ctx,
    )

    # Run Diagnostician (retry once if it fails to produce structured output —
    # the LLM may spend all turns on tool calls without calling submit_assessment)
    assessment, diag_meta = assess_strategy(diag_input)
    state.total_cost_usd += _estimate_cost(diag_meta)
    if assessment is None:
        assessment, diag_meta2 = assess_strategy(diag_input)
        state.total_cost_usd += _estimate_cost(diag_meta2)
    if assessment is None:
        update_strategy_status(state, strategy_id, "pending",
                               "Diagnostician failed to produce assessment (agent error, not strategy failure)")
        return "discarded"

    entry.assessments.append(assessment.model_dump())

    # Cross-check numerical claims
    issues = cross_check_assessment(
        assessment, trade_log,
        proxy_sharpe=strategy_data.get("mean_val_sharpe", 0.0),
        qc_sharpe=qc_result.get("sharpe", 0.0),
    )
    if issues:
        memory_append("terminal_patterns", {
            "terminal": "cross_check",
            "observation": f"Diagnostician claims mismatched for {strategy_id}: {issues}",
            "confidence": 0.3,
        })
        # Reject the assessment — Diagnostician's numerical claims are unreliable
        update_strategy_status(state, strategy_id, "discarded",
                               f"Cross-check failed: {'; '.join(issues)}")
        if on_discard:
            on_discard(strategy_id, f"cross_check_failed: {'; '.join(issues)}")
        return "discarded"

    # Route based on assessment action
    action = assessment.action

    if action == "APPROVE":
        # Deterministic threshold override
        if (qc_metrics.sharpe < APPROVE_MIN_QC_SHARPE
                or qc_metrics.total_trades < APPROVE_MIN_QC_TRADES
                or qc_metrics.max_drawdown > APPROVE_MAX_QC_DRAWDOWN):
            update_strategy_status(state, strategy_id, "discarded",
                                   f"Diagnostician approved but failed hard thresholds: "
                                   f"sharpe={qc_metrics.sharpe:.2f}, "
                                   f"trades={qc_metrics.total_trades}, "
                                   f"dd={qc_metrics.max_drawdown:.2f}")
            if on_discard:
                on_discard(strategy_id, "threshold_override")
            return "discarded"

        update_strategy_status(state, strategy_id, "approved",
                               assessment.tradeability)
        if on_approve:
            on_approve(strategy_id, assessment)

        # Log to memory
        memory_append("template_viability", {
            "template": strategy_data.get("template_name", ""),
            "strategy_id": strategy_id,
            "qc_sharpe": qc_metrics.sharpe,
            "action": "approved",
            "confidence": assessment.confidence,
        })
        return "approved"

    elif action == "DISCARD":
        update_strategy_status(state, strategy_id, "discarded",
                               assessment.technical_gap[:200])
        if on_discard:
            on_discard(strategy_id, assessment.technical_gap[:200])

        memory_append("template_viability", {
            "template": strategy_data.get("template_name", ""),
            "strategy_id": strategy_id,
            "qc_sharpe": qc_metrics.sharpe,
            "action": "discarded",
            "reason": assessment.technical_gap[:100],
            "confidence": assessment.confidence,
        })
        return "discarded"

    elif action == "MODIFY":
        if entry.iteration >= MAX_MUTATION_ITERATIONS:
            # At max iterations, Diagnostician can still approve but not modify
            update_strategy_status(state, strategy_id, "discarded",
                                   f"Max mutation iterations ({MAX_MUTATION_ITERATIONS}) reached")
            if on_discard:
                on_discard(strategy_id, "max_iterations")
            return "discarded"

        # Run Mutator (context isolation: no QC numbers)
        update_strategy_status(state, strategy_id, "modifying")
        mut_input = MutatorInput(
            strategy_id=strategy_id,
            template_name=strategy_data.get("template_name", ""),
            condition=strategy_data.get("condition", "scalar-only"),
            entry_sexpr=strategy_data.get("entry_sexpr", ""),
            exit_sexpr=strategy_data.get("exit_sexpr", ""),
            size_sexpr=strategy_data.get("size_sexpr", "EphReal(0.5)"),
            delta_sexpr=strategy_data.get("delta_sexpr"),
            technical_gap=assessment.technical_gap,
            modification_guidance=assessment.modification_guidance or "",
            available_terminals=strategy_data.get("available_terminals", []),
            grammar_max_nodes=strategy_data.get("grammar_max_nodes", 15),
            grammar_max_depth=strategy_data.get("grammar_max_depth", 5),
        )

        proposal, mut_meta = propose_mutation(mut_input)
        state.total_cost_usd += _estimate_cost(mut_meta)
        if proposal is None:
            update_strategy_status(state, strategy_id, "discarded",
                                   "Mutator failed to produce proposal")
            if on_discard:
                on_discard(strategy_id, "mutator_failure")
            return "discarded"

        # Validate mutation (with one retry on failure)
        original_trees = {
            "entry_sexpr": strategy_data.get("entry_sexpr", ""),
            "exit_sexpr": strategy_data.get("exit_sexpr", ""),
            "size_sexpr": strategy_data.get("size_sexpr", ""),
            "delta_sexpr": strategy_data.get("delta_sexpr", ""),
        }
        valid, error = validate_mutation(
            proposal, original_trees,
            condition=strategy_data.get("condition", "scalar-only"),
        )
        if not valid:
            # Retry once: re-call Mutator with validation error as feedback
            mut_input.modification_guidance = (
                f"Previous mutation was REJECTED: {error}. "
                f"Original guidance: {mut_input.modification_guidance}. "
                f"Please propose a different mutation that satisfies grammar constraints."
            )
            proposal2, mut_meta2 = propose_mutation(mut_input)
            state.total_cost_usd += _estimate_cost(mut_meta2)
            if proposal2:
                valid2, error2 = validate_mutation(
                    proposal2, original_trees,
                    condition=strategy_data.get("condition", "scalar-only"),
                )
                if valid2:
                    proposal = proposal2  # use the retry
                    valid = True
                    error = ""
        if not valid:
            update_strategy_status(state, strategy_id, "discarded",
                                   f"Mutation validation failed after retry: {error}")
            if on_discard:
                on_discard(strategy_id, f"invalid_mutation: {error}")
            return "discarded"

        entry.mutations.append(proposal.model_dump())
        entry.iteration += 1

        # Log mutation outcome (success will be determined after re-deployment)
        memory_append("mutation_outcomes", {
            "strategy_id": strategy_id,
            "mutation_type": proposal.target_tree,
            "from": proposal.original_subtree[:50],
            "to": proposal.replacement_subtree[:50],
            "iteration": entry.iteration,
        })

        # Return "modifying" — the caller must re-deploy and re-process
        return "modifying"

    # Unknown action
    update_strategy_status(state, strategy_id, "discarded",
                           f"Unknown action: {action}")
    return "discarded"


# ---------------------------------------------------------------------------
# Batch-level orchestration
# ---------------------------------------------------------------------------

def run_batch(
    state: PipelineState,
    strategies: Dict[str, dict],
    deploy_fn: Callable,
    interpret_fn: Callable,
    hitl_fn: Optional[Callable] = None,
) -> Dict[str, str]:
    """Run the full supervisor loop for a batch of strategies.

    This is the top-level entry point. It processes the queue,
    deploys strategies, runs the diagnose-modify loop, and
    triggers HITL for approved strategies.

    Args:
        state: Pipeline state with queue populated.
        strategies: Dict of strategy_id → strategy data dict.
        deploy_fn: (strategy_id) → qc_result dict. Handles codegen + QC deployment.
        interpret_fn: (strategy_data) → interpretation string.
        hitl_fn: (strategy_id, assessment) → bool. Returns True if human approves.

    Returns:
        Dict of strategy_id → final status.
    """
    results = {}

    while state.queue:
        strategy_id = state.queue.pop(0)
        strategy_data = strategies.get(strategy_id)
        if strategy_data is None:
            update_strategy_status(state, strategy_id, "discarded", "No strategy data")
            results[strategy_id] = "discarded"
            save_state(state)
            continue

        # Budget check
        if state.total_cost_usd > COST_BUDGET_USD:
            update_strategy_status(state, strategy_id, "discarded", "Budget exceeded")
            results[strategy_id] = "discarded"
            save_state(state)
            continue

        # Generate interpretation
        interpretation = interpret_fn(strategy_data)

        # Deploy → diagnose → (maybe modify → re-deploy) loop
        current_data = strategy_data.copy()
        status = "deploying"

        while status == "deploying" or status == "modifying":
            # Deploy to QC
            update_strategy_status(state, strategy_id, "deploying")
            qc_result = deploy_fn(strategy_id)

            if qc_result is None or qc_result.get("error"):
                update_strategy_status(state, strategy_id, "discarded",
                                       qc_result.get("error", "Deploy failed") if qc_result else "Deploy returned None")
                status = "discarded"
                break

            # Process through diagnose-modify loop
            def _on_approve(sid, assessment):
                if hitl_fn:
                    approved = hitl_fn(sid, assessment)
                    if not approved:
                        update_strategy_status(state, sid, "review", "Human rejected")

            status = process_strategy(
                state, strategy_id, current_data, qc_result,
                interpretation,
                on_approve=_on_approve,
            )

            if status == "modifying":
                # Apply mutation to strategy data for re-deployment
                entry = state.ledger[strategy_id]
                if entry.mutations:
                    last_mut = entry.mutations[-1]
                    target = last_mut.get("target_tree", "entry")
                    key = f"{target}_sexpr"
                    # Simple replacement: swap the original subtree with replacement
                    orig = last_mut.get("original_subtree", "")
                    repl = last_mut.get("replacement_subtree", "")
                    if orig and repl and key in current_data:
                        current_data[key] = current_data[key].replace(orig, repl, 1)

        results[strategy_id] = status
        save_state(state)

    return results
