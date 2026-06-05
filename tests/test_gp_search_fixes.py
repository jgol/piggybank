"""Unit tests for the 2026-06-02 GP search-side winner-killer fixes.

#2  Max-hold gate: a >90%-max-hold (vestigial-exit) strategy is DEMOTED with a
    subtractive penalty, NOT hard-sentineled to -1e6 — so a genuinely profitable
    hold-to-close theta-harvest strategy survives on its real Sharpe.
#4  Graded infeasibility: an infeasible individual's fitness is the 1e6 sentinel
    base + a small graded violation in [0,1), preserving feasible-dominance EXACTLY
    while giving NSGA-III a gradient toward feasibility (vs the old flat 1e6 that
    drifted all-infeasible folds to 0/256 feasible).
"""
import numpy as np
import pytest

from layer2.fitness import (
    _graded_infeasible, FAILED_FITNESS_SENTINEL, _INFEASIBLE_GRADE_SPAN,
)


# --------------------------------------------------------------------------- #
# #4 — graded infeasibility helper
# --------------------------------------------------------------------------- #

def test_graded_infeasible_preserves_dominance_and_gives_gradient():
    """The graded sentinel must (a) stay strictly worse than ANY feasible individual
    and (b) order infeasibles by violation so NSGA has a climb-to-feasible gradient."""
    worst_feasible = np.full(3, 5.0)          # a bad-but-feasible Sharpe-space point
    barely = _graded_infeasible(3, 0.0)       # just outside a gate
    middling = _graded_infeasible(3, 0.5)
    maximal = _graded_infeasible(3, 1.0)      # maximally infeasible

    # (a) dominance: every infeasible is far above (worse than) the worst feasible,
    # on every objective — feasible ALWAYS dominates infeasible.
    for inf in (barely, middling, maximal):
        assert np.all(inf > worst_feasible), "feasible must dominate every infeasible"

    # (b) gradient: less violation strictly dominates more violation (all objectives).
    assert np.all(barely < middling) and np.all(middling < maximal), \
        "a less-violating infeasible must Pareto-dominate a more-violating one"

    # bounded in [SENTINEL, SENTINEL + SPAN]; clipping at the ends.
    assert np.allclose(barely, FAILED_FITNESS_SENTINEL)
    assert np.allclose(maximal, FAILED_FITNESS_SENTINEL + _INFEASIBLE_GRADE_SPAN)
    assert np.all(_graded_infeasible(3, 9.0) == maximal)   # violation > 1 clipped
    assert np.all(_graded_infeasible(3, -3.0) == barely)   # violation < 0 clipped

    # the graded span is many orders below the base — it can never flip dominance.
    assert _INFEASIBLE_GRADE_SPAN < 1e-3 * FAILED_FITNESS_SENTINEL


def test_graded_infeasible_wired_into_evaluate_for_too_few_trades():
    """An obviously-infeasible strategy (zero/near-zero trades on noise) comes back
    at the graded sentinel via the real VectorizedFitnessEvaluator.evaluate path —
    i.e. >= the 1e6 base and within one grade-span of it, not a flat exact 1e6."""
    from tests.test_gp_positive_control import build_planted_edge_fixture
    from layer2.grammar import from_sexpr
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.evaluator_vectorized import prepare_terminal_data
    from layer2.fitness import VectorizedFitnessEvaluator
    from layer2.templates import bull_put_credit_standard

    df, _ = build_planted_edge_fixture(n_days=30, bars_per_date=40, seed=3)
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    fe = VectorizedFitnessEvaluator(template=bull_put_credit_standard(), data=df,
                                    terminal_data=td, warmup_bars=15, min_trades=50)
    # An entry that essentially never fires -> total_trades << min_trades=50 -> the
    # graded min-trades sentinel.
    fit = fe.evaluate(from_sexpr("LT(RealizedVol30m, EphReal(-9.0))"),
                      from_sexpr("LT(MinutesToClose, EphReal(-5.0))"),
                      from_sexpr("EphReal(0.5)"))
    assert np.all(fit >= FAILED_FITNESS_SENTINEL), "infeasible must be at/above the base"
    assert np.all(fit <= FAILED_FITNESS_SENTINEL + _INFEASIBLE_GRADE_SPAN + 1e-9), \
        "graded sentinel must stay within one span of the base"


# --------------------------------------------------------------------------- #
# #2 — max-hold gate is a subtractive penalty, not a hard sentinel
# --------------------------------------------------------------------------- #

def test_max_hold_vestigial_uses_subtractive_penalty_not_sentinel():
    """Force a >90%-max-hold (vestigial-exit) strategy and confirm the evaluator no
    longer returns the -1e6 hard sentinel for it: the Sharpe is finite and far above
    -1e5, reflecting real performance minus the subtractive vestigial penalty."""
    from tests.test_gp_positive_control import build_planted_edge_fixture
    from layer2.grammar import from_sexpr
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.evaluator_vectorized import (
        prepare_terminal_data, vectorized_backtest, _MAX_HOLD_VESTIGIAL_PENALTY,
    )
    from layer2.templates import bull_put_credit_standard

    # Max-length days (the evaluator caps bars/day at 405) with a SMALL max_bars
    # so ~40 back-to-back max_bars trades per day overwhelmingly dominate the ~2
    # per-day overhead trades (the late tail + re-entry boundary) -> max-hold
    # fraction comfortably > 0.90, exercising the gate.
    df, is_vol = build_planted_edge_fixture(n_days=12, bars_per_date=405, seed=5)
    # Keep ONLY calm days: on a calm day the held credit spread never stop-losses,
    # so every position rides to max_bars (volatile days would flood the trade count
    # with early stop-loss exits and the max-hold fraction would never dominate).
    date_list = sorted(df["date"].unique())
    calm_dates = {date_list[i] for i in range(len(date_list)) if not is_vol[i]}
    df = df[df["date"].isin(calm_dates)].reset_index(drop=True)
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)

    # enter-always + never-exit + a SMALL max_bars so every position force-closes at
    # max_bars -> >90% max-hold exits (the vestigial-exit gate condition).
    res = vectorized_backtest(
        from_sexpr("GT(MinutesToClose, EphReal(-99.0))"),
        from_sexpr("LT(MinutesToClose, EphReal(-99.0))"),
        from_sexpr("EphReal(0.5)"),
        df, bull_put_credit_standard(), terminal_data=td,
        warmup_bars=15, min_bars_in_trade=3, max_bars_in_trade=8,
    )

    # the gate condition is actually exercised: >90% of trades hit max_bars.
    assert res.total_trades >= 10
    max_hold_frac = sum(1 for t in res.trades
                        if t.exit_bar - t.entry_bar >= 8 - 1) / len(res.trades)
    assert max_hold_frac > 0.90, f"test must trigger the max-hold gate; got {max_hold_frac:.2f}"

    # THE FIX: no -1e6 hard sentinel. Finite, far above -1e5.
    assert np.isfinite(res.sharpe), "max-hold strategy must not be NaN/inf"
    assert res.sharpe > -1e5, (
        f"max-hold strategy must NOT be hard-sentineled to -1e6; got {res.sharpe}")
    # sanity: the constant is a modest subtractive penalty, not a kill switch.
    assert 0.0 < _MAX_HOLD_VESTIGIAL_PENALTY <= 3.0
