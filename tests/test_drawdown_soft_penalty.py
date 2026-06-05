"""Drawdown enters as a SOFT PENALTY on the return axes, NOT a 4th objective.

Adding max_drawdown as a 4th NSGA objective would recreate the documented
"zero-profit Pareto attractor" (a low-drawdown ~zero-Sharpe strategy becomes
non-dominated). Instead the penalty pushes high-drawdown strategies DOWN the
return axes. HIGH-4 (2026-06-01 SECOND audit, SUPERSEDES the earlier MH4 neg_sharpe-
only decision): the penalty is applied to BOTH return axes (neg_sharpe AND
neg_sortino). Penalizing neg_sharpe ONLY was Pareto-INCONSISTENT — a high-drawdown
strategy dodged the demotion on the unpenalized neg_sortino axis and re-entered the
non-dominated set (the penalty created NEW non-dominated points). Applying it to both
is dominance-PRESERVING (a penalized point is dominated on both return axes by an
otherwise-equal low-drawdown sibling); the ~2x effective weight is accepted as the
price of consistency. These tests pin that (a) the penalty is applied above the free
level and zero below it, (b) it hits BOTH neg_sharpe AND neg_sortino equally, and
(c) the objective count stays 3.
"""
import numpy as np

from layer2.evaluator import BacktestResult
from layer2.fitness import (
    VectorizedFitnessEvaluator, DEFAULT_OBJECTIVES,
    DRAWDOWN_FREE_LEVEL, DRAWDOWN_PENALTY_WEIGHT,
)


def _bare_evaluator():
    # _fitness_from_result only reads self.objectives — skip the heavy __init__.
    ev = VectorizedFitnessEvaluator.__new__(VectorizedFitnessEvaluator)
    ev.objectives = DEFAULT_OBJECTIVES
    return ev


def _result(sharpe, sortino, dd_uncapped, avg_size=0.5, n_trades=100, n_days=120):
    return BacktestResult(
        returns=np.array([0.01, -0.005] * 50),
        trades=[], equity_curve=np.zeros(10),
        max_drawdown=min(dd_uncapped, 1.0), sharpe=sharpe, sortino=sortino,
        total_trades=n_trades, n_days=n_days, avg_position_size=avg_size,
        max_drawdown_uncapped=dd_uncapped,
    )


def test_no_fourth_objective_axis():
    """The soft gate must NOT add a Pareto axis (attractor avoidance)."""
    assert len(DEFAULT_OBJECTIVES) == 3
    assert "max_drawdown" not in DEFAULT_OBJECTIVES
    f = _bare_evaluator()._fitness_from_result(_result(0.5, 0.5, 0.3))
    assert f.shape == (3,)


def test_low_drawdown_unpenalized():
    """adj_dd below the free level -> no penalty; neg_sharpe == -sharpe."""
    # adj_dd = 0.3 / 0.5 = 0.6 < 1.0 (free level)
    f = _bare_evaluator()._fitness_from_result(_result(0.5, 0.5, 0.3))
    assert abs(f[0] - (-0.5)) < 1e-9  # neg_sharpe, no penalty
    assert abs(f[1] - (-0.5)) < 1e-9  # neg_sortino, no penalty


def test_high_drawdown_penalizes_both_return_axes():
    """HIGH-4 (2026-06-01 SECOND audit): adj_dd above the free level worsens BOTH
    neg_sharpe AND neg_sortino by exactly DRAWDOWN_PENALTY_WEIGHT*(adj_dd - free).
    Penalizing both is dominance-preserving (Pareto-consistent); penalizing only
    neg_sharpe let high-drawdown points survive on the unpenalized sortino axis.
    trade_count axis untouched."""
    ev = _bare_evaluator()
    f_low = ev._fitness_from_result(_result(0.5, 0.5, 0.3))   # adj_dd 0.6 -> 0 penalty
    f_high = ev._fitness_from_result(_result(0.5, 0.5, 3.0))  # adj_dd 6.0 -> penalty
    expected = DRAWDOWN_PENALTY_WEIGHT * (6.0 - DRAWDOWN_FREE_LEVEL)
    assert abs(f_high[0] - (-0.5 + expected)) < 1e-9, "neg_sharpe penalized"
    assert abs(f_high[1] - (-0.5 + expected)) < 1e-9, "neg_sortino ALSO penalized (HIGH-4)"
    assert f_high[0] > f_low[0], "higher drawdown must be WORSE on neg_sharpe"
    assert f_high[1] > f_low[1], "higher drawdown must be WORSE on neg_sortino too"
    assert abs(f_high[2] - f_low[2]) < 1e-9, "trade_count axis must be unaffected"


def test_drawdown_penalty_is_pareto_consistent():
    """The penalty must DEMOTE on every return axis simultaneously, so a high-drawdown
    strategy cannot dominate an otherwise-identical low-drawdown sibling via an
    unpenalized axis (the bug HIGH-4 fixes). With equal raw Sharpe/Sortino, the
    low-drawdown sibling dominates the high-drawdown one on BOTH return axes."""
    ev = _bare_evaluator()
    low_dd = ev._fitness_from_result(_result(0.5, 0.5, 0.3))   # no penalty
    high_dd = ev._fitness_from_result(_result(0.5, 0.5, 3.0))  # penalized both axes
    # pymoo MINIMIZES: low_dd must be <= high_dd on BOTH return objectives (dominates)
    assert low_dd[0] <= high_dd[0] and low_dd[1] <= high_dd[1], \
        "low-drawdown sibling must dominate on BOTH return axes (Pareto consistency)"
    assert low_dd[0] < high_dd[0] or low_dd[1] < high_dd[1], \
        "and strictly better on at least one"


def test_penalty_is_sizing_exploit_proof():
    """Halving avg_position_size doubles adj_dd (dd per unit size), so the GP
    cannot dodge the penalty by sizing down — same uncapped dd, smaller size,
    LARGER penalty."""
    ev = _bare_evaluator()
    f_big = ev._fitness_from_result(_result(0.5, 0.5, 2.0, avg_size=0.5))  # adj 4.0
    f_small = ev._fitness_from_result(_result(0.5, 0.5, 2.0, avg_size=0.25))  # adj 8.0
    assert f_small[0] > f_big[0], "smaller size -> larger adj_dd -> larger penalty"
