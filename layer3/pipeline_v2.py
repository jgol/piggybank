"""L3 Pipeline v2: Multi-agent strategy validation and refinement.

Replaces the old 3-stage linear pipeline (pipeline.py) with the
Deep Agents supervisor + subagent architecture.

Usage:
    # Dry mode (gates + statistical tests only, no LLM or QC):
    python -m layer3.pipeline_v2 --results-dir results/wf_scalar-only_levelB_* \\
        --mode dry

    # Diagnose mode (gates + mock-QC + diagnostician, no real QC):
    python -m layer3.pipeline_v2 --results-dir results/wf_scalar-only_levelB_* \\
        --mode diagnose

    # Full mode (gates + real QC + full agent pipeline):
    python -m layer3.pipeline_v2 --results-dir results/wf_scalar-only_levelB_* \\
        --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from layer3.gates import all_prefilter_pass, run_prefilter_gates, gate_d4_dedup
from layer3.schemas import (
    PipelineState, StrategyPacket, StrategyAssessment,
    PortfolioAssessment,
)
from layer3.state import create_state, save_state, load_state, resume_state
from layer3.memory import load_context
from layer3.supervisor import process_strategy, run_batch, COST_BUDGET_USD
from layer3.config import HITL_TIMEOUT_HOURS
from layer3.portfolio import build_portfolio_assessment
from layer3.hitl import (
    request_approval, wait_for_decision, check_existing_decision,
    check_pending_request,
)
from layer2.strategy_interpreter import interpret_strategy
from layer3.config import DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END


def oos_window_for_fold(fold_id):
    """Per-fold out-of-sample QC backtest window (P1: D5-D8 OOS-overlap fix).

    A strategy from walk-forward fold F was evolved on data <= F.train_end and
    must be VALIDATED on data AFTER F.train_end (its val+test span) to be genuinely
    out-of-sample. The fixed DEFAULT_BACKTEST window (2024-10-01..2026-05-22)
    overlaps fold-3/4 TRAINING (train_end 2025-03-31 / 2025-09-30), so those
    strategies were "validated" partly on data they trained on (Lopez de Prado,
    2018, Ch.7 - leakage). Start the QC backtest at the fold's train_end; the
    holdout fold (test_end=None) runs to DEFAULT_BACKTEST_END.

    Returns (start_date, end_date). Unknown/missing fold_id -> the full default
    window (backward-compatible).
    """
    try:
        fid = int(str(fold_id))
    except (ValueError, TypeError):
        return DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END
    from layer2.experiment import WALK_FORWARD_FOLDS
    for f in WALK_FORWARD_FOLDS:
        if f["fold_id"] == fid:
            return f["train_end"], (f["test_end"] or DEFAULT_BACKTEST_END)
    return DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END


# ---------------------------------------------------------------------------
# Proxy re-evaluation for portfolio daily returns
# ---------------------------------------------------------------------------

# Cached L1 data for proxy re-evaluation (loaded once per batch)
_l1_data_cache: Optional[dict] = None


def _load_l1_data():
    """Load L1 parquet + templates once for proxy re-evaluation."""
    global _l1_data_cache
    if _l1_data_cache is not None:
        return _l1_data_cache

    import pandas as pd
    from layer2.templates import TEMPLATE_FACTORIES, BASE_TEMPLATE_FACTORIES

    l1_path = Path("raw_data/local_store/l1_cnn.parquet")
    if not l1_path.exists():
        _l1_data_cache = {"df": None, "templates": {}}
        return _l1_data_cache

    df = pd.read_parquet(l1_path)
    templates = {}
    templates.update(TEMPLATE_FACTORIES)
    templates.update(BASE_TEMPLATE_FACTORIES)
    _l1_data_cache = {"df": df, "templates": templates}
    return _l1_data_cache


def _get_proxy_returns(strategy_data: dict) -> "np.ndarray":
    """Re-run proxy evaluator for a strategy to get daily returns.

    Falls back to empty array if data or evaluator is unavailable.
    """
    import numpy as np
    data = _load_l1_data()
    if data["df"] is None:
        return np.array([])

    try:
        from layer2.grammar import from_sexpr
        from layer2.evaluator_vectorized import vectorized_backtest

        entry_tree = from_sexpr(strategy_data.get("entry_sexpr", "EphReal(0)"))
        exit_tree = from_sexpr(strategy_data.get("exit_sexpr", "EphReal(0)"))
        size_tree = from_sexpr(strategy_data.get("size_sexpr", "EphReal(0.5)"))
        delta_tree = None
        if strategy_data.get("delta_sexpr"):
            delta_tree = from_sexpr(strategy_data["delta_sexpr"])

        template_name = strategy_data.get("template_name", "bull_put_credit")
        template_fn = data["templates"].get(template_name)
        if template_fn is None:
            return np.array([])
        template = template_fn()

        # Re-run with the strategy's evolved stop so the proxy returns used for
        # the proxy↔QC comparison match the fitness it was selected on.
        _stop_kw = ({"stop_loss_credit_multiple": float(strategy_data["stop_mult"])}
                    if strategy_data.get("stop_mult") is not None else {})
        result = vectorized_backtest(
            entry_tree=entry_tree,
            exit_tree=exit_tree,
            size_tree=size_tree,
            data=data["df"],
            template=template,
            delta_tree=delta_tree,
            **_stop_kw,
        )
        return result.returns
    except Exception:
        return np.array([])


# ---------------------------------------------------------------------------
# QC order parsing
# ---------------------------------------------------------------------------

def _parse_orders_to_trade_log(orders: List[dict]) -> List[dict]:
    """Parse QC order list into trade records for the Diagnostician.

    QC orders are individual option leg fills. We group them into trades
    by looking at the 'tag' field which our codegen populates with exit
    reasons (e.g., "signal_exit", "stop_loss", "eod_exit", "max_hold").

    Returns list of dicts compatible with TradeRecord schema.
    """
    if not orders:
        return []

    trades = []
    # Group orders by time (same-second orders = one trade event)
    from collections import defaultdict
    by_time = defaultdict(list)
    for o in orders:
        t = o.get("time", "")
        by_time[t].append(o)

    # Track open/close: opening orders have positive quantity (buying to open
    # credit spread = selling, but QC reports net quantity). We use tags to
    # identify exit vs entry.
    for t, group in sorted(by_time.items()):
        tags = [o.get("tag", "") for o in group]
        combined_tag = " ".join(tags).lower()

        # Determine if this is an exit event
        reason = ""
        if "stop_loss" in combined_tag or "stop" in combined_tag:
            reason = "stop_loss"
        elif "signal" in combined_tag:
            reason = "signal"
        elif "eod" in combined_tag or "end_of_day" in combined_tag:
            reason = "eod"
        elif "max_hold" in combined_tag:
            reason = "max_hold"
        elif "onendofalgorithm" in combined_tag:
            reason = "eod"

        if reason:
            # Calculate net credit from fill prices
            total_value = sum(
                o.get("price", 0) * o.get("quantity", 0)
                for o in group
            )
            trades.append({
                "time": t,
                "reason": reason,
                "bars_held": 0,  # not available from order data alone
                "entry_credit": total_value,
            })

    return trades


# ---------------------------------------------------------------------------
# Strategy loading from GP walk-forward results
# ---------------------------------------------------------------------------

def _infer_condition(results_dir: str) -> str:
    """Infer GP condition from the results directory name."""
    dirname = Path(results_dir).name
    for cond in ("scalar-only", "emb-only", "probes-only", "real-l1", "shuffled-l1"):
        if cond in dirname:
            return cond
    return "scalar-only"


def vix_source_failed(*stat_dicts) -> bool:
    """P0-5b: True if a generated backtest flagged a dead CBOE VIX subscription.

    The generated QC algorithm (layer3/codegen.py) sets the runtime statistic
    ``vix_source_failed=1`` (and logs a loud Error) when ``AddData(CBOE,"VIX")``
    never delivers data and the VIX terminals degrade (VIXSpot -> ATM_IV*100
    proxy; VIXTermSlope/VIXChange/VIXMean5d -> frozen constants) — re-injecting
    the training-serving skew P0-5 removed (Sculley et al., 2015). Such a run
    reports a proxy-scale Sharpe that does NOT reflect the intended strategy, so
    it is INVALID for proxy->QC calibration and must be rejected.

    Defensive across the dicts QC/MCP may surface custom statistics in
    (``statistics`` vs ``runtimeStatistics``; spaces vs underscores in the key).
    The predicate is "key present AND value not falsy" rather than an exact-value
    allowlist, so it keeps working if the producer ever emits a count instead of
    the literal "1".
    """
    _falsy = ("", "0", "0.0", "false", "no", "none")
    for d in stat_dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if str(k).strip().lower().replace(" ", "_") == "vix_source_failed":
                return str(v).strip().lower() not in _falsy
    return False


def load_gp_strategies(
    results_dir: str,
    template_filter: Optional[str] = None,
    max_per_template: int = 10,
) -> Dict[str, dict]:
    """Load strategies from GP walk-forward JSONL files.

    Returns dict of strategy_id → strategy data dict.
    Applies top-N by val_sharpe per template.
    """
    results_path = Path(results_dir)
    condition = _infer_condition(results_dir)
    strategies = {}

    # Find all strategy JSONL files
    for jsonl_path in sorted(results_path.rglob("strategies_*.jsonl")):
        template_name = jsonl_path.stem.replace("strategies_", "")
        if template_filter and template_name != template_filter:
            continue

        # Determine fold from path
        fold_dir = jsonl_path.parent.name  # e.g., "fold_1"
        fold_id = fold_dir.replace("fold_", "") if fold_dir.startswith("fold_") else "0"

        with open(jsonl_path) as f:
            strats = [json.loads(line) for line in f if line.strip()]

        # Sort by val_sharpe descending, take top-N
        strats.sort(key=lambda s: s.get("val_sharpe", 0), reverse=True)
        for rank, s in enumerate(strats[:max_per_template]):
            sid = f"{template_name}_f{fold_id}_r{rank}"
            strategies[sid] = {
                "strategy_id": sid,
                "template_name": template_name,
                "condition": condition,
                "entry_sexpr": s.get("entry_tree", ""),
                "exit_sexpr": s.get("exit_tree", ""),
                "size_sexpr": s.get("size_tree", "EphReal(0.5)"),
                "delta_sexpr": s.get("delta_tree"),
                # GP stop gene — carried through so codegen emits the evolved stop
                # (proxy↔QC parity). None ⇒ legacy strategy (codegen falls back to
                # the proxy default). 0.0 ⇒ hold-to-expiry.
                "stop_mult": s.get("stop_mult"),
                "fold_id": fold_id,
                "mean_val_sharpe": s.get("val_sharpe", 0.0),
                "val_trades": s.get("n_trades_val", 0),
                "val_win_rate": 0.0,
                "val_drawdown": 0.0,
                "pbo_score": s.get("pbo", 0.0),
                "fold_sharpes": {},
                "train_sharpe": s.get("train_sharpe", 0.0),
                "test_sharpe": s.get("test_sharpe", 0.0),
                "fitness": s.get("fitness", []),
                "total_nodes": s.get("total_nodes", 0),
                # For gate checking
                "val_sharpe": s.get("val_sharpe", 0.0),
                "n_trades_val": s.get("n_trades_val", 0),
                "val_psr": s.get("val_psr", 0.0),
                "val_return_skew": s.get("val_return_skew", 0.0),
                "val_return_kurtosis": s.get("val_return_kurtosis", 0.0),
                # BLOCKER B1 (2026-06-01 holistic review): carry the per-fold MINUTE
                # normalization the strategy was evolved under. Without this the L3
                # codegen call at pipeline_v2:579 always read None and fell back to the
                # FROZEN daily TERMINAL_NORM_STATS — a train-serve normalization skew
                # (Sculley et al. 2015) that silently defeated P0-6 and corrupted every
                # proxy↔QC parity measurement. The L2 serializer (experiment.py:2747)
                # writes fold_norm_stats onto each JSONL record; propagate it here.
                "fold_norm_stats": s.get("fold_norm_stats"),
            }
    return strategies


# ---------------------------------------------------------------------------
# Gate filtering
# ---------------------------------------------------------------------------

def apply_prefilter_gates(
    strategies: Dict[str, dict],
) -> Dict[str, dict]:
    """Apply deterministic pre-filter gates (D1-D3 + structural).

    Converts strategy dicts to StrategyPackets for gate compatibility,
    then filters. D4 (dedup) operates at population level separately.
    Returns only strategies that pass all individual gates.
    """
    passed = {}
    for sid, s in strategies.items():
        # Build a minimal StrategyPacket for gate checking
        from layer3.schemas import FoldMetrics
        packet = StrategyPacket(
            strategy_id=sid,
            template_name=s.get("template_name", ""),
            condition=s.get("condition", "scalar-only"),
            entry_sexpr=s.get("entry_sexpr", ""),
            exit_sexpr=s.get("exit_sexpr", ""),
            size_sexpr=s.get("size_sexpr", "EphReal(0.5)"),
            delta_sexpr=s.get("delta_sexpr"),
            stop_mult=s.get("stop_mult"),
            mean_val_sharpe=s.get("val_sharpe", 0.0),
            total_val_trades=s.get("n_trades_val", 0),
        )
        try:
            gate_results = run_prefilter_gates(packet)
            s["gate_results"] = [
                {"gate_id": g.gate_id, "passed": g.passed,
                 "value": g.value, "threshold": g.threshold,
                 "reason": g.reason}
                for g in gate_results
            ]
            # D1 and D2 require cross-fold metrics (fold_metrics populated).
            # When loading from per-fold JSONL (not walk-forward summary),
            # fold_metrics are unavailable. Fall back to val_sharpe > 0.30
            # for D1 and skip D2 (persistence checked at walk-forward level).
            d1_ok = any(g.passed for g in gate_results if g.gate_id == "D1")
            if not d1_ok and packet.mean_val_sharpe > 0.30:
                d1_ok = True  # fallback: raw val_sharpe check
            d2_ok = any(g.passed for g in gate_results if g.gate_id == "D2")
            if not d2_ok and not packet.fold_metrics:
                d2_ok = True  # skip D2 when cross-fold data unavailable
            d3_ok = any(g.passed for g in gate_results if g.gate_id == "D3")
            struct_ok = any(g.passed for g in gate_results if g.gate_id == "STRUCT")

            if d1_ok and d2_ok and d3_ok and struct_ok:
                passed[sid] = s
        except Exception as e:
            print(f"  WARNING: Gate check failed for {sid}: {type(e).__name__}: {e}")
    return passed


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def _interpret(strategy_data: dict) -> str:
    """Generate plain-language interpretation of a strategy."""
    return interpret_strategy(
        entry_sexpr=strategy_data.get("entry_sexpr", ""),
        exit_sexpr=strategy_data.get("exit_sexpr", ""),
        size_sexpr=strategy_data.get("size_sexpr", "EphReal(0.5)"),
        delta_sexpr=strategy_data.get("delta_sexpr"),
        template_name=strategy_data.get("template_name", ""),
        stop_mult=strategy_data.get("stop_mult"),
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    results_dir: str,
    mode: str = "dry",
    template_filter: Optional[str] = None,
    max_per_template: int = 10,
    batch_id: Optional[str] = None,
    cost_budget: float = COST_BUDGET_USD,
) -> dict:
    """Run the L3 multi-agent pipeline.

    Args:
        results_dir: Path to GP walk-forward results.
        mode: "dry" (gates only), "diagnose" (gates + mock-QC + diagnostician),
              "full" (gates + real QC + all agents).
        template_filter: Optional template name to process.
        max_per_template: Top-N strategies per template to process.
        batch_id: Batch ID for state tracking (default: timestamp).
        cost_budget: Maximum LLM API cost in USD.

    Returns:
        Dict with pipeline results summary.
    """
    if batch_id is None:
        batch_id = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  L3 Pipeline v2 — mode={mode}, batch={batch_id}")
    print(f"{'='*60}\n")

    # Step 1: Load GP strategies
    print("[1/6] Loading GP strategies...")
    strategies = load_gp_strategies(results_dir, template_filter, max_per_template)
    print(f"  Loaded {len(strategies)} strategies from {results_dir}")
    if not strategies:
        print("  No strategies found. Exiting.")
        return {"batch_id": batch_id, "n_loaded": 0, "n_passed_gates": 0}

    # Step 2: Apply pre-filter gates
    print("\n[2/6] Applying pre-filter gates (D1-D4)...")
    passed = apply_prefilter_gates(strategies)
    print(f"  {len(passed)}/{len(strategies)} passed gates")
    if not passed:
        print("  No strategies passed gates. Exiting.")
        return {"batch_id": batch_id, "n_loaded": len(strategies), "n_passed_gates": 0}

    # Step 3: Dedup (D4) — population-level deduplication
    pre_dedup = len(passed)
    packets_for_dedup = []
    for sid, s in passed.items():
        packets_for_dedup.append(StrategyPacket(
            strategy_id=sid,
            template_name=s.get("template_name", ""),
            condition=s.get("condition", "scalar-only"),
            entry_sexpr=s.get("entry_sexpr", ""),
            exit_sexpr=s.get("exit_sexpr", ""),
            size_sexpr=s.get("size_sexpr", "EphReal(0.5)"),
            delta_sexpr=s.get("delta_sexpr"),
            stop_mult=s.get("stop_mult"),
            mean_val_sharpe=s.get("val_sharpe", 0.0),
            total_val_trades=s.get("n_trades_val", 0),
        ))
    dedup_results = gate_d4_dedup(packets_for_dedup)
    for sid, gate_res in dedup_results.items():
        if not gate_res.passed and sid in passed:
            del passed[sid]
    if pre_dedup != len(passed):
        print(f"  D4 dedup: {pre_dedup} → {len(passed)} (removed {pre_dedup - len(passed)} duplicates)")

    # Step 4: Generate interpretations
    print("\n[3/6] Generating strategy interpretations...")
    interpretations = {}
    for sid, s in passed.items():
        interpretations[sid] = _interpret(s)
    print(f"  {len(interpretations)} interpretations generated")

    if mode == "dry":
        print("\n[DRY MODE] Gates complete. No LLM or QC calls.")
        # Print summary
        for sid, s in passed.items():
            print(f"  {sid}: val_sharpe={s.get('val_sharpe', 0):.3f}, "
                  f"trades={s.get('n_trades_val', 0)}")
        return {
            "batch_id": batch_id,
            "mode": mode,
            "n_loaded": len(strategies),
            "n_passed_gates": len(passed),
            "passed_strategies": list(passed.keys()),
        }

    # Step 5: Create/resume pipeline state
    print("\n[4/6] Initializing pipeline state...")
    existing_state = load_state(batch_id)
    if existing_state:
        print(f"  Resuming from existing state: {len(existing_state.completed)} completed")
        state = resume_state(existing_state)
    else:
        template_names = {sid: s["template_name"] for sid, s in passed.items()}
        state = create_state(batch_id, results_dir, list(passed.keys()), template_names)
    # Don't reset cost on resume — accumulate across retries
    save_state(state)

    if mode == "diagnose":
        print("\n[DIAGNOSE MODE] Running with mock QC results...")
        # Mock deploy: use proxy metrics as QC results (no real deployment)
        def mock_deploy(sid):
            s = passed.get(sid, {})
            return {
                "sharpe": s.get("val_sharpe", 0.0) * 0.7,  # simulate QC discount
                "total_trades": s.get("n_trades_val", 0),
                "win_rate": 0.5,
                "max_drawdown": 0.08,
                "net_profit": 0.0,
                "total_fees": 0.0,
                "trade_log": [],
            }

        results = run_batch(
            state, passed,
            deploy_fn=mock_deploy,
            interpret_fn=_interpret,
        )
        save_state(state)

        print(f"\n[DIAGNOSE RESULTS]")
        for sid, status in results.items():
            print(f"  {sid}: {status}")
        return {
            "batch_id": batch_id,
            "mode": mode,
            "n_loaded": len(strategies),
            "n_passed_gates": len(passed),
            "results": results,
            "n_approved": len(state.approved),
            "n_discarded": len(state.discarded),
        }

    if mode == "full":
        print("\n[FULL MODE] Running with live QC deployment...")
        print(f"  Budget: ${cost_budget}")
        print(f"  Queue: {len(state.queue)} strategies")

        # Cleanup orphaned L3_ projects from prior failed runs
        _cleanup_script = '''
import asyncio, json, sys
sys.path.insert(0, 'qc_mcp')
from utils.mcp_connection import QCMCPConnection

async def cleanup():
    async with QCMCPConnection(tool_timeout=60) as conn:
        raw = await conn.call_tool('list_projects', {'model': {}})
        try:
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, Exception):
            data = {}  # MCP validation errors return non-JSON
        for p in data.get('projects', []):
            if p.get('name', '').startswith('L3_'):
                pid = p['projectId']
                await conn.call_tool('delete_project', {'model': {'projectId': pid}})
                print(json.dumps({"deleted": p['name'], "pid": pid}), flush=True)

asyncio.run(cleanup())
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = "qc_mcp"
        try:
            cleanup_proc = subprocess.run(
                ["qc_mcp/bin/python", "-c", _cleanup_script],
                capture_output=True, text=True, env=env, timeout=120,
            )
            for line in cleanup_proc.stdout.strip().split("\n"):
                if line.strip():
                    try:
                        evt = json.loads(line)
                        print(f"  Cleaned up orphaned project: {evt.get('deleted')} (pid={evt.get('pid')})")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"  Cleanup skipped: {type(e).__name__}: {e}")

        # Real deploy: codegen (system python) → QC deployment (MCP venv subprocess)
        def real_deploy(sid):
            s = passed.get(sid, {})
            if not s:
                return {"error": f"Strategy {sid} not in passed set"}
            try:
                # Step 1: Generate QC Python code (system python — has L2 packages)
                from layer3.codegen import generate_qc_algorithm, validate_generated_code
                # P1: validate each strategy on its fold's OOS window (after
                # train_end), not the fixed window that overlaps fold-3/4 training.
                _bt_start, _bt_end = oos_window_for_fold(s.get("fold_id", "0"))
                code = generate_qc_algorithm(
                    strategy_id=sid,
                    template_name=s.get("template_name", "bull_put_credit"),
                    entry_sexpr=s.get("entry_sexpr", ""),
                    exit_sexpr=s.get("exit_sexpr", ""),
                    size_sexpr=s.get("size_sexpr", "EphReal(0.5)"),
                    delta_sexpr=s.get("delta_sexpr"),
                    # PARITY: emit the evolved stop the strategy was selected on.
                    stop_mult=s.get("stop_mult"),
                    start_date=_bt_start,
                    end_date=_bt_end,
                    # P0-6: deploy with the per-fold MINUTE normalization the
                    # strategy was evolved under (when persisted on the record);
                    # falls back to frozen TERMINAL_NORM_STATS if absent.
                    norm_stats=s.get("fold_norm_stats"),
                )
                ok, err = validate_generated_code(code)
                if not ok:
                    return {"error": f"Codegen syntax error: {err}"}

                # Save code to temp file for MCP venv to read
                code_path = Path(f"generated_qc/l3_pipeline/L3_{sid}.py")
                code_path.parent.mkdir(parents=True, exist_ok=True)
                code_path.write_text(code)
                print(f"  [{sid}] Codegen OK ({len(code)} chars)")

                # Step 2: Deploy via MCP venv subprocess
                deploy_script = f'''
import asyncio, json, sys, os
sys.path.insert(0, 'qc_mcp')
from utils.mcp_connection import QCMCPConnection

async def deploy():
    code = open("{code_path}").read()
    project_name = "L3_{sid}"
    async with QCMCPConnection(tool_timeout=300) as conn:
        # Create project
        r = json.loads(await conn.call_tool('create_project',
            {{'model': {{'name': project_name, 'language': 'Py'}}}}))
        pid = r['projects'][0]['projectId']
        print(json.dumps({{"event": "project_created", "pid": pid}}), flush=True)

        # Upload
        await conn.call_tool('read_file',
            {{'model': {{'projectId': pid, 'name': 'main.py'}}}})
        await conn.call_tool('update_file_contents',
            {{'model': {{'projectId': pid, 'name': 'main.py', 'content': code}}}})
        print(json.dumps({{"event": "code_uploaded"}}), flush=True)

        # Compile
        cr = json.loads(await conn.call_tool('create_compile',
            {{'model': {{'projectId': pid}}}}))
        cid = cr.get('compileId', '')
        for attempt in range(30):
            await asyncio.sleep(10)
            try:
                raw = await conn.call_tool('read_compile',
                    {{'model': {{'projectId': pid, 'compileId': cid}}}})
                if not raw: continue
                cr2 = json.loads(raw)
            except Exception:
                continue
            if cr2.get('state') == 'BuildSuccess':
                print(json.dumps({{"event": "compile_success"}}), flush=True)
                break
            if cr2.get('state') == 'BuildError':
                errors = cr2.get('errors', [])
                await conn.call_tool('delete_project', {{'model': {{'projectId': pid}}}})
                print(json.dumps({{"event": "compile_error", "errors": errors[:5]}}), flush=True)
                return
        else:
            await conn.call_tool('delete_project', {{'model': {{'projectId': pid}}}})
            print(json.dumps({{"event": "compile_timeout"}}), flush=True)
            return

        # Backtest
        bt_raw = await conn.call_tool('create_backtest',
            {{'model': {{'projectId': pid, 'compileId': cid, 'backtestName': '{sid}_run'}}}})
        bt = json.loads(bt_raw)
        bt_id = bt.get('backtestId') or bt.get('backtest', {{}}).get('backtestId', '')
        print(json.dumps({{"event": "backtest_started", "bt_id": bt_id}}), flush=True)

        for i in range(360):  # 360 × 15s = 90 min (minute-res 0DTE-options backtests report progress=0 until near-done and can take 60+ min; do NOT infer "stuck" from 0%)
            await asyncio.sleep(15)
            try:
                raw = await conn.call_tool('read_backtest',
                    {{'model': {{'projectId': pid, 'backtestId': bt_id}}}})
                if not raw:
                    if i % 8 == 0:
                        print(json.dumps({{"event": "progress", "pct": 0, "note": "empty response"}}), flush=True)
                    continue
                poll = json.loads(raw)
            except (json.JSONDecodeError, Exception):
                continue

            # QC returns completed at top level OR nested under backtest
            completed = poll.get('completed', False) or poll.get('backtest', {{}}).get('completed', False)
            progress = poll.get('progress', 0) or poll.get('backtest', {{}}).get('progress', 0)

            if not completed:
                if i % 4 == 0:
                    print(json.dumps({{"event": "progress", "pct": progress}}), flush=True)
                continue

            # Stats are under backtest.statistics (MCP read_backtest response)
            _bt = poll.get('backtest', {{}})
            stats = _bt.get('statistics', {{}})
            # P0-5b: custom runtime statistics carry the vix_source_failed marker;
            # capture defensively (QC may key it runtimeStatistics or RuntimeStatistics).
            runtime_stats = (_bt.get('runtimeStatistics') or _bt.get('RuntimeStatistics') or {{}})
            # Read orders. QC read_backtest_orders is 1-INDEXED + paginated:
            # start=0 returns a progress stub with NO orders; paginate from
            # start=1, <=100/call, advance by batch size (verified 2026-05-31).
            orders = []
            try:
                _ostart = 1; _olen = None
                while True:
                    _oraw = await conn.call_tool('read_backtest_orders',
                        {{'model': {{'projectId': pid, 'backtestId': bt_id, 'start': _ostart, 'end': _ostart + 99}}}})
                    _od = json.loads(_oraw) if _oraw else {{}}
                    if not isinstance(_od, dict):
                        break
                    _olen = _od.get('length', _olen)
                    _ob = _od.get('orders', []) or []
                    if not _ob:
                        break
                    orders.extend(_ob); _ostart += len(_ob)
                    if _olen is not None and len(orders) >= _olen:
                        break
            except Exception:
                orders = []

            await conn.call_tool('delete_project', {{'model': {{'projectId': pid}}}})
            print(json.dumps({{"event": "complete", "stats": stats, "runtime_stats": runtime_stats, "n_orders": len(orders) if isinstance(orders, list) else 0, "pid": pid, "bt_id": bt_id}}), flush=True)
            return

        await conn.call_tool('delete_project', {{'model': {{'projectId': pid}}}})
        print(json.dumps({{"event": "backtest_timeout"}}), flush=True)

asyncio.run(deploy())
'''
                env = os.environ.copy()
                env["PYTHONPATH"] = "qc_mcp"
                proc = subprocess.run(
                    ["qc_mcp/bin/python", "-c", deploy_script],
                    capture_output=True, text=True, env=env,
                    timeout=6000,  # 100 min — must exceed the inner 90-min poll loop (minute-options are slow)
                )

                # Parse events from stdout
                events = []
                for line in proc.stdout.strip().split("\n"):
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"  [{sid}] non-JSON: {line}")

                for evt in events:
                    event_type = evt.get("event", "")
                    if event_type == "project_created":
                        print(f"  [{sid}] QC project created: {evt.get('pid')}")
                    elif event_type == "compile_success":
                        print(f"  [{sid}] Compile OK")
                    elif event_type == "compile_error":
                        return {"error": f"Compile error: {evt.get('errors', [])}"}
                    elif event_type == "compile_timeout":
                        return {"error": "Compile timeout (5 min)"}
                    elif event_type == "backtest_started":
                        print(f"  [{sid}] Backtest started: {evt.get('bt_id', '')[:16]}...")
                    elif event_type == "progress":
                        print(f"  [{sid}] Progress: {evt.get('pct', 0):.0%}")
                    elif event_type == "backtest_timeout":
                        return {"error": "Backtest timeout (90 min)"}
                    elif event_type == "complete":
                        stats = evt.get("stats", {})
                        orders = []  # orders already consumed by subprocess
                        # P0-5b: a dead CBOE VIX subscription makes the VIX terminals
                        # silently use ATM_IV*100, so the reported (proxy-scale) Sharpe
                        # is invalid. Reject rather than accept it into calibration.
                        if vix_source_failed(stats, evt.get("runtime_stats", {})):
                            print(f"  [{sid}] INVALID: P0-5b VIX source failure "
                                  f"(real CBOE VIX never populated; VIX terminals used ATM_IV*100)")
                            return {"error": "P0-5b VIX source failure: real CBOE VIX never "
                                             "populated; backtest INVALID for proxy->QC calibration"}
                        print(f"  [{sid}] COMPLETE: Sharpe={stats.get('Sharpe Ratio', 'N/A')}, "
                              f"Trades={stats.get('Total Orders', stats.get('Total Trades', 'N/A'))}")
                        return {
                            "sharpe": float(stats.get("Sharpe Ratio", 0)),
                            "total_trades": int(stats.get("Total Orders", stats.get("Total Trades", 0))),
                            "win_rate": float(stats.get("Win Rate", "0").replace("%", "")) / 100
                                if stats.get("Win Rate") else 0.0,
                            "max_drawdown": abs(float(stats.get("Drawdown", "0").replace("%", ""))) / 100
                                if stats.get("Drawdown") else 0.0,
                            "net_profit": float(stats.get("Net Profit", "0").replace("%", "").replace("$", "").replace(",", ""))
                                if stats.get("Net Profit") else 0.0,
                            "total_fees": float(stats.get("Total Fees", "$0").replace("$", "").replace(",", ""))
                                if stats.get("Total Fees") else 0.0,
                            "trade_log": [],
                            "project_id": evt.get("pid"),
                            "backtest_id": evt.get("bt_id"),
                        }

                # No complete event found
                if proc.returncode != 0:
                    return {"error": f"Deploy subprocess failed: {proc.stderr[:500]}"}
                return {"error": "Deploy produced no results"}
            except subprocess.TimeoutExpired:
                return {"error": "Deploy subprocess timed out (100 min)"}
            except Exception as e:
                return {"error": f"Deploy failed: {type(e).__name__}: {e}"}

        def hitl_callback(sid, assessment):
            request_approval(batch_id, sid, {
                "strategy_id": sid,
                "template_name": passed[sid]["template_name"],
                "interpretation": interpretations.get(sid, ""),
                "proxy_sharpe": passed[sid].get("val_sharpe", 0),
                "assessment": assessment.model_dump() if assessment else {},
            })
            # Check for existing decision (resume case)
            dec = check_existing_decision(batch_id, sid)
            if dec:
                return dec.get("decision") == "approve"
            # Block for decision (with short timeout for testing)
            dec = wait_for_decision(batch_id, sid, timeout_hours=HITL_TIMEOUT_HOURS)
            if dec:
                return dec.get("decision") == "approve"
            return False

        results = run_batch(
            state, passed,
            deploy_fn=real_deploy,
            interpret_fn=_interpret,
            hitl_fn=hitl_callback,
        )
        save_state(state)

        # Step 6: Portfolio assessment
        if state.approved:
            print(f"\n[6/6] Portfolio assessment ({len(state.approved)} approved)...")
            try:
                import numpy as np
                from layer3.portfolio import build_portfolio_assessment
                from layer3.agent_portfolio_manager import assess_portfolio as llm_assess_portfolio

                strategy_returns = {}
                strategy_terminals = {}
                strategy_templates = {}
                for sid in state.approved:
                    s = passed.get(sid, {})
                    strategy_templates[sid] = s.get("template_name", "")
                    # Extract terminals from entry tree
                    from layer2.grammar import from_sexpr, TermNode
                    try:
                        tree = from_sexpr(s.get("entry_sexpr", "EphReal(0)"))
                        terms = []
                        def _collect(node):
                            if isinstance(node, TermNode) and node.name not in ("EphReal", "EphInt"):
                                terms.append(node.name)
                            elif hasattr(node, "children"):
                                for c in node.children:
                                    _collect(c)
                        _collect(tree)
                        strategy_terminals[sid] = terms
                    except Exception:
                        strategy_terminals[sid] = []
                    # Re-run proxy evaluator to get actual daily returns
                    strategy_returns[sid] = _get_proxy_returns(s)
                n_with_returns = sum(1 for r in strategy_returns.values() if len(r) > 0)
                print(f"  Proxy re-evaluation: {n_with_returns}/{len(state.approved)} have real returns")

                det_assessment = build_portfolio_assessment(
                    strategy_returns, strategy_terminals, strategy_templates)
                print(f"  Deterministic: Sharpe={det_assessment.portfolio_sharpe:.3f}, "
                      f"DD={det_assessment.portfolio_max_drawdown:.4f}")
                if det_assessment.diversification_warnings:
                    for w in det_assessment.diversification_warnings:
                        print(f"  WARNING: {w}")

                # LLM assessment (only in full mode with API key)
                if mode == "full":
                    try:
                        final_assessment, pm_meta = llm_assess_portfolio(det_assessment)
                        print(f"  LLM: proceed={final_assessment.proceed_recommendation}")
                        if final_assessment.kill_criteria:
                            for sid, crit in final_assessment.kill_criteria.items():
                                print(f"  Kill: {sid} → {crit}")
                    except Exception as e:
                        print(f"  LLM assessment skipped: {e}")
            except Exception as e:
                print(f"  Portfolio assessment failed: {type(e).__name__}: {e}")
        else:
            print(f"\n[6/6] No strategies approved — skipping portfolio assessment")

        print(f"\n{'='*60}")
        print(f"  BATCH COMPLETE: {batch_id}")
        print(f"  Loaded: {len(strategies)}")
        print(f"  Passed gates: {len(passed)}")
        print(f"  Approved: {len(state.approved)}")
        print(f"  Discarded: {len(state.discarded)}")
        print(f"  Cost: ${state.total_cost_usd:.2f}")
        print(f"{'='*60}\n")

        return {
            "batch_id": batch_id,
            "mode": mode,
            "n_loaded": len(strategies),
            "n_passed_gates": len(passed),
            "n_approved": len(state.approved),
            "n_discarded": len(state.discarded),
            "approved": state.approved,
            "discarded": state.discarded,
            "cost_usd": state.total_cost_usd,
        }

    raise ValueError(f"Unknown mode: {mode}. Must be dry, diagnose, or full.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="layer3.pipeline_v2",
        description="L3 multi-agent strategy validation pipeline",
    )
    parser.add_argument("--results-dir", required=True,
                        help="Path to GP walk-forward results directory")
    parser.add_argument("--mode", choices=["dry", "diagnose", "full"],
                        default="dry", help="Pipeline mode")
    parser.add_argument("--template", default=None,
                        help="Filter to a specific template")
    parser.add_argument("--max-per-template", type=int, default=10,
                        help="Top-N strategies per template")
    parser.add_argument("--batch-id", default=None,
                        help="Batch ID (default: timestamp)")
    parser.add_argument("--budget", type=float, default=COST_BUDGET_USD,
                        help=f"Max LLM API cost in USD (default: {COST_BUDGET_USD})")
    args = parser.parse_args()

    result = run_pipeline(
        results_dir=args.results_dir,
        mode=args.mode,
        template_filter=args.template,
        max_per_template=args.max_per_template,
        batch_id=args.batch_id,
        cost_budget=args.budget,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
