"""Torpedo fixes from the 2026-06-02 DSR/fitness adversarial audit.

Covers the fixes confined to layer2/fitness.py and layer2/pbo.py:

  B2 (BLOCKER): the DSR evolution-trial recorder must count trials from gen 0,
      NOT gated by the gp_engine min_trades ramp (the ramp gate undercounted N
      ~30x, leaving SR* far too lenient).  [Primary B2 recorder-guard test lives
      in tests/test_fitness_gate_fixes.py::test_record_trial_not_gated_by_min_trades_ramp;
      here we pin the N-counting consequence at the (N, V) derivation layer.]

  H1 (HIGH): a single near-flat-variance Sharpe artifact must be EXCLUDED from
      the DSR V estimate (not clipped-in at ±1.0 daily), and the plausible-Sharpe
      bound is a genuine 0DTE daily bound (~0.35) instead of the "absurd" 1.0.

  L2 (LOW): the random-entry null gate's fire-rate surcharge must use
      entry_fire_rate_flat (the selectivity measure the #6 tautology gate uses),
      NOT the raw bar-level entry_fire_rate.
"""
import math

import numpy as np
import pandas as pd
import pytest

from layer2.pbo import (
    TRADING_DAYS_PER_YEAR,
    _MAX_PLAUSIBLE_DAILY_SHARPE,
    evolution_n_v_from_records,
)

_SQRT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)


# ===========================================================================
# H1 — flat-variance artifact excluded from V (not clipped-in)
# ===========================================================================

def _records(sharpes_ann, n_days=125):
    """Build an evolution trial-records dict {sig: (ann_sharpe, n_days)}."""
    return {f"sig{i}": (float(s), int(n_days)) for i, s in enumerate(sharpes_ann)}


class TestH1FlatVarianceArtifactExcludedFromV:
    def test_plausible_bound_is_realistic_not_absurd(self):
        # The bound must be a genuinely plausible 0DTE DAILY Sharpe (~0.35,
        # ann ~5), not the prior 1.0 (ann ~15.9) the comment itself called absurd.
        assert _MAX_PLAUSIBLE_DAILY_SHARPE == pytest.approx(0.35)
        assert _MAX_PLAUSIBLE_DAILY_SHARPE < 0.5  # genuinely plausible band

    def test_single_artifact_does_not_inflate_v(self):
        """A population of plausible Sharpes + ONE near-flat-variance artifact
        (ann 5000 == daily ~315, from the returns_std 1e-9 floor). V must equal
        the variance of ONLY the plausible members — the artifact is EXCLUDED,
        not clipped to ±1.0 and kept."""
        plausible_ann = [0.5, 1.0, 1.5, 2.0, 0.8, 1.2]
        recs = _records(plausible_ann)
        recs["flat_artifact"] = (5000.0, 125)  # variance artifact, valid length

        n, v = evolution_n_v_from_records(recs, min_days=20)

        # V is the population variance of the PLAUSIBLE daily Sharpes only.
        plausible_daily = np.array([s / _SQRT_YEAR for s in plausible_ann])
        v_expected = float(np.var(plausible_daily, ddof=0))
        assert v == pytest.approx(v_expected, rel=1e-9), (
            "H1: the flat-variance artifact must be EXCLUDED from V"
        )

        # The OLD behaviour (clip the artifact to +1.0 daily and KEEP it in the
        # sample) would have produced a dramatically larger V. Prove the fix
        # actually moved the number.
        clipped_daily = np.append(
            plausible_daily, _MAX_PLAUSIBLE_DAILY_SHARPE)  # old clip-in value
        # also include the pre-H1 cap of 1.0 to mirror the exact old code:
        clipped_daily_old_cap = np.append(plausible_daily, 1.0)
        v_old = float(np.var(clipped_daily_old_cap, ddof=0))
        assert v_old > 10 * v_expected, (
            "sanity: clipping-in an artifact at the old 1.0 cap inflates V >10x"
        )
        assert v < v_old, "H1: excluding the artifact yields a much smaller V"

    def test_n_still_counts_the_artifact_trial(self):
        """B2 invariant preserved: every trial that RAN counts toward N (a
        backtest is a multiple-testing trial regardless of its Sharpe). Only V's
        sample drops the artifact; N is unchanged."""
        plausible_ann = [0.5, 1.0, 1.5]
        recs = _records(plausible_ann)
        recs["flat_artifact"] = (5000.0, 125)
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 4, "N counts ALL valid-length finite-Sharpe trials (B2)"

    def test_all_artifacts_leaves_zero_variance(self):
        """If every member is a variance artifact, excluding them leaves < 2
        valid V samples -> V = 0.0 (no dispersion to deflate). N still counts
        them (they ran)."""
        recs = _records([5000.0, 6000.0, 7000.0])
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 3, "every trial that ran counts toward N"
        assert v == 0.0, "H1: with no plausible members there is no V to inflate"

    def test_member_exactly_at_bound_is_kept(self):
        """A member EXACTLY at the plausible bound is kept (<=, not <)."""
        at_bound_ann = _MAX_PLAUSIBLE_DAILY_SHARPE * _SQRT_YEAR
        recs = _records([0.5, 1.0, at_bound_ann])
        n, v = evolution_n_v_from_records(recs, min_days=20)
        daily = np.array([0.5 / _SQRT_YEAR, 1.0 / _SQRT_YEAR,
                          _MAX_PLAUSIBLE_DAILY_SHARPE])
        assert v == pytest.approx(float(np.var(daily, ddof=0)), rel=1e-9)

    def test_short_sample_filter_for_v_still_applies(self):
        """The pre-existing n_days >= min_days validity filter for V is kept
        (H1 only changes the implausible-Sharpe handling, not the length filter)."""
        recs = {**_records([0.5, 1.0, 1.5], n_days=125)}
        recs["short"] = (2.0, 5)  # short sample -> excluded from N and V
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 3
        daily = np.array([0.5 / _SQRT_YEAR, 1.0 / _SQRT_YEAR, 1.5 / _SQRT_YEAR])
        assert v == pytest.approx(float(np.var(daily, ddof=0)), rel=1e-9)


# ===========================================================================
# L2 — random-entry null gate uses entry_fire_rate_flat, not raw fire rate
# ===========================================================================

from layer2.evaluator import BacktestResult           # noqa: E402
from layer2.fitness import FAILED_FITNESS_SENTINEL, VectorizedFitnessEvaluator  # noqa: E402
from layer2.grammar import from_sexpr                  # noqa: E402
from layer2.io import (                                # noqa: E402
    PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor               # noqa: E402


def _make_data(n_dates=4, bars_per_date=40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for di, date in enumerate([f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]):
        spot = 4500.0
        iv = 0.12 + 0.04 * di
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.0)
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: spot, "ATM_IV": iv,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18,
                "PredRV15": 0.30, "PredRV30": 0.30,
                "PredRegime": 0, "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = np.zeros(384, dtype=np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = 0.25
            rows.append(row)
    return pd.DataFrame(rows)


def _result(*, sharpe, entry_fire_rate, entry_fire_rate_flat):
    """A passing-gate backtest result that varies only the two fire rates."""
    return BacktestResult(
        returns=np.array([0.01, -0.005] * 50), trades=[],
        equity_curve=np.zeros(10), max_drawdown=0.05, sharpe=sharpe,
        sortino=sharpe + 0.1, total_trades=50, win_rate=0.6, n_days=4,
        exit_utilization=0.5, entry_fire_rate=entry_fire_rate,
        entry_fire_rate_flat=entry_fire_rate_flat, conditional_sharpe_gap=0.2,
        avg_position_size=0.5, max_drawdown_uncapped=0.05,
    )


def _make_evaluator():
    from layer2.evaluator_vectorized import prepare_terminal_data
    data = _make_data()
    td = prepare_terminal_data(data)
    fe = VectorizedFitnessEvaluator(
        template=iron_condor(), data=data, terminal_data=td,
        regime_gate_enabled=False,
    )
    # Pin a deterministic random-entry baseline + an active margin so the gate
    # is exercised (it is skipped when margin == 0 or baseline is None).
    fe._random_entry_sharpe = 0.0
    fe.random_entry_margin = 0.30
    return fe


class TestL2RandomEntryNullUsesFlatFireRate:
    def test_threshold_responds_to_flat_not_raw_fire_rate(self):
        """The null threshold surcharge is 0.50 * entry_fire_rate_flat. A
        day-selective hold with HIGH raw fire rate but LOW flat fire rate must
        get a LOW surcharge (so it can pass), while the SAME strategy scored
        under the old raw-rate surcharge would face a HIGH bar and be gated.

        threshold = baseline(0.0) + margin(0.30) + 0.50 * <fire-rate>.
        With flat=0.04 -> thr = 0.32 ; with raw=0.42 -> thr = 0.51.
        A Sharpe of 0.40 PASSES the flat bar (0.40 > 0.32) but FAILS the raw bar
        (0.40 <= 0.51) -> proves the gate now keys on the FLAT rate.
        """
        import layer2.evaluator_vectorized as _EV
        fe = _make_evaluator()
        entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
        exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
        size = from_sexpr("EphReal(0.5)")

        _orig = _EV.vectorized_backtest
        try:
            # Day-selective hold: raw rate high (0.42), flat rate low (0.04).
            _EV.vectorized_backtest = lambda *a, **kw: _result(
                sharpe=0.40, entry_fire_rate=0.42, entry_fire_rate_flat=0.04)
            f = fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
            assert f[0] < FAILED_FITNESS_SENTINEL, (
                "L2: gate uses FLAT fire rate (0.04 -> thr 0.32); Sharpe 0.40 "
                "must PASS — the old raw-rate bar (0.42 -> thr 0.51) wrongly "
                "sentineled this day-selective winner"
            )

            # Control: a genuine always-enter churner has BOTH rates high -> the
            # flat-rate surcharge is high too, so a sub-threshold Sharpe is gated.
            _EV.vectorized_backtest = lambda *a, **kw: _result(
                sharpe=0.40, entry_fire_rate=0.42, entry_fire_rate_flat=0.42)
            f2 = fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
            assert f2[0] >= FAILED_FITNESS_SENTINEL, (
                "L2: an unconditional churner (flat rate 0.42 -> thr 0.51) with "
                "Sharpe 0.40 must still be gated by the null"
            )
        finally:
            _EV.vectorized_backtest = _orig

    def test_graded_violation_uses_flat_threshold(self):
        """The graded-infeasibility magnitude for the null gate is keyed off the
        flat-rate threshold (_rand_thr), so two strategies differing only in
        their RAW fire rate (same flat rate) get the SAME violation grade."""
        import layer2.evaluator_vectorized as _EV
        fe = _make_evaluator()
        entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
        exit_ = from_sexpr("LT(MinutesToClose, EphReal(0.0))")
        size = from_sexpr("EphReal(0.5)")
        _orig = _EV.vectorized_backtest
        try:
            # Both below the flat-rate null bar; identical flat rate, different raw.
            _EV.vectorized_backtest = lambda *a, **kw: _result(
                sharpe=0.10, entry_fire_rate=0.30, entry_fire_rate_flat=0.20)
            f_a = fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
            _EV.vectorized_backtest = lambda *a, **kw: _result(
                sharpe=0.10, entry_fire_rate=0.05, entry_fire_rate_flat=0.20)
            f_b = fe._evaluate_on_data(entry, exit_, size, fe.data, fe.terminal_data)
            assert f_a[0] >= FAILED_FITNESS_SENTINEL
            assert f_b[0] >= FAILED_FITNESS_SENTINEL
            assert f_a[0] == pytest.approx(f_b[0]), (
                "L2: null-gate violation grade depends on the FLAT rate, so a "
                "differing RAW rate must not change it"
            )
        finally:
            _EV.vectorized_backtest = _orig
