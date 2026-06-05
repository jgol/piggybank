"""Regression test for the GATE-ATTRIBUTION fix (2026-06-04): the exit-utilization
gate must grade its graded-infeasibility violation by SHARPE, not exit_util.

Root cause of the feasibility collapse (0-2/128): the dominant 0DTE-credit structure
is hold-to-EOD-settlement (exit_util=0). The old grading (0.20-exit_util)/0.20 was a
FLAT 1.0 for EVERY such strategy, so NSGA-III had no gradient and the population
couldn't climb — while churny SIGNAL-exit losers (exit_util>=0.20, Sharpe ~-24)
passed. Grading by -sharpe/3.0 gives a gradient toward profitability so a -0.38
hold-to-close dominates a -2.0.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.fitness import FAILED_FITNESS_SENTINEL  # noqa: E402


def _violation(sharpe):
    # the exact term in fitness.py's exit-util gate
    return min(1.0, max(0.0, -sharpe / 3.0))


def test_violation_monotonic_decreasing_in_sharpe():
    # less-negative Sharpe -> lower violation -> dominates (climbing gradient)
    assert _violation(-0.1) < _violation(-0.38) < _violation(-2.0) < _violation(-24)
    # contract: clipped to [0, 1] so a feasible individual always dominates
    assert _violation(0.0) == 0.0
    assert _violation(-5.0) == 1.0
    assert all(0.0 <= _violation(s) <= 1.0 for s in (-0.1, -0.38, -2.0, -24, -100))


def test_source_grades_by_sharpe_not_exit_util():
    import layer2.fitness as f
    src = open(f.__file__).read()
    # the exit-util gate's graded-infeasible call must use -result.sharpe
    assert "min(1.0, max(0.0, -result.sharpe / 3.0))" in src, (
        "exit-util gate must grade by Sharpe (gate-attribution fix 2026-06-04)")
    # and must NOT still grade by the flat exit_util term
    assert "(0.20 - result.exit_utilization) / 0.20)" not in src, (
        "old flat exit_util grading must be gone")


def test_hold_to_close_seed_violation_is_sharpe_graded_not_flat():
    """End-to-end: the hold-to-expiry VRP seed (exit_util=0, sharpe~-0.38) must get a
    Sharpe-graded violation (~0.13), NOT the old flat 1.0."""
    import pandas as pd
    P = "raw_data/local_store/l1_minute_scalars.parquet"
    if not os.path.exists(P):
        pytest.skip("production parquet absent")
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    from layer2.terminal_stats import compute_norm_stats_from_arrays
    from layer2.templates import bull_put_credit_base
    from layer2.experiment import derive_vrp_compound_seed
    from layer2.grammar import from_sexpr
    from layer2.fitness import VectorizedFitnessEvaluator
    df = pd.read_parquet(P)
    tr = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                       embargo_days=0)["train"]
    fold = compute_norm_stats_from_arrays(
        prepare_terminal_data(tr, normalize_terminals=False), robust=True)
    td = prepare_terminal_data(tr, norm_stats_override=fold)
    entry = derive_vrp_compound_seed(td)
    tmpl = bull_put_credit_base()
    fe = VectorizedFitnessEvaluator(template=tmpl, data=tr, terminal_data=td,
                                    min_trades=20, warmup_bars=30,
                                    regime_gate_enabled=True)
    fv = fe.evaluate(entry, from_sexpr("LT(MinutesToClose, EphReal(-5.0))"),
                     from_sexpr("EphReal(0.5)"), delta_tree=tmpl.delta_seed,
                     stop_mult=0.0)
    assert fv[0] >= FAILED_FITNESS_SENTINEL, "seed is (still) infeasible (sharpe<=0)"
    violation = fv[0] - FAILED_FITNESS_SENTINEL
    # Sharpe ~-0.38 -> violation ~0.13; the old flat bug gave 1.0
    assert violation < 0.5, (
        f"violation {violation:.3f} should be Sharpe-graded (~0.13), not flat 1.0")
    assert violation > 0.0, "a sharpe<=0 hold-to-close is infeasible with positive violation"
