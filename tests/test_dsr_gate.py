"""Tests for the P1-B Deflated Sharpe Ratio (DSR) gate — REDESIGNED 2026-05-30.

The gate deflates each champion's TRAINING Sharpe (the in-sample quantity
NSGA-III selected on, hence the one inflated by the N-trial search) against the
expected maximum Sharpe SR*_N of the WHOLE evolution's distinct trials (N, V),
NOT the tiny final Pareto front (whose own variance inverted the bar — Blocker
B). Deflating the training Sharpe against a training-trial SR* keeps the units
consistent (classic Bailey & Lopez de Prado in-sample deflation). The gate
FILTERS: champions failing DSR < threshold are removed from the front.

Functions under test (layer2/pbo.py — the SINGLE DSR implementation):
  - expected_max_sharpe(n_trials, sharpe_variance) -> SR*_N          (unchanged)
  - deflated_sharpe_ratio(sharpe_daily, n_days, skew, kurtosis, sr_star) (unchanged)
  - evolution_n_v_from_records(trial_records, min_days) -> (N, V)    (NEW)
  - dsr_gate_evolution(front, n_trials, sharpe_variance, ...) -> per-member DSR + pass (NEW)

UNITS are load-bearing: front members carry ANNUALIZED Sharpes
(BacktestResult.sharpe = x sqrt(252)); the DSR math is in PER-PERIOD (daily)
units. dsr_gate_evolution and evolution_n_v_from_records de-annualize internally.

Reference: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio,"
Journal of Portfolio Management 40(5), 94-107.
"""
import math

import numpy as np
import pytest
from scipy.stats import norm

from layer2.pbo import (
    TRADING_DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    dsr_gate_evolution,
    evolution_n_v_from_records,
    expected_max_sharpe,
)

_GAMMA = 0.5772156649
_SQRT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)

# Real fold length for this project's 0DTE data (~76-125 val trading days; the
# data starts mid-2022, so folds CANNOT be longer). Threshold calibration and
# the "champion passes at real T" test pin against this.
REAL_T_MIN = 76
REAL_T_TYPICAL = 125
PROPOSED_THRESHOLD = 0.5  # config default (pure-deflation bar at short T)


def _emax_ref(n, v):
    """Reference SR*_N independent of the implementation."""
    z = (1 - _GAMMA) * norm.ppf(1 - 1.0 / n) + _GAMMA * norm.ppf(1 - 1.0 / (n * math.e))
    return math.sqrt(v) * z


# ---------------------------------------------------------------------------
# expected_max_sharpe (unchanged core; kept under test)
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    def test_matches_hand_computed_value(self):
        val = expected_max_sharpe(10, 1.0)
        assert val == pytest.approx(1.5745983013449718, rel=1e-12)
        assert val == pytest.approx(_emax_ref(10, 1.0), rel=1e-12)

    def test_increases_with_n(self):
        v = 1.0
        vals = [expected_max_sharpe(n, v) for n in [2, 5, 10, 100, 1000, 10000]]
        for a, b in zip(vals, vals[1:]):
            assert a < b, f"SR* must increase with N: {a} !< {b}"

    def test_increases_with_variance(self):
        n = 50
        vals = [expected_max_sharpe(n, v) for v in [0.25, 0.5, 1.0, 2.0, 4.0]]
        for a, b in zip(vals, vals[1:]):
            assert a < b, f"SR* must increase with V: {a} !< {b}"

    def test_scales_as_sqrt_variance(self):
        n = 25
        base = expected_max_sharpe(n, 1.0)
        quad = expected_max_sharpe(n, 4.0)
        assert quad == pytest.approx(2.0 * base, rel=1e-12)

    def test_zero_variance_gives_zero(self):
        assert expected_max_sharpe(100, 0.0) == 0.0

    def test_n_clamped_to_two(self):
        v1 = expected_max_sharpe(1, 1.0)
        v0 = expected_max_sharpe(0, 1.0)
        v2 = expected_max_sharpe(2, 1.0)
        assert math.isfinite(v1) and math.isfinite(v0)
        assert v1 == pytest.approx(v2, rel=1e-12)
        assert v0 == pytest.approx(v2, rel=1e-12)

    def test_production_scale_n(self):
        # Production N is in the thousands (distinct individuals across a run),
        # NOT N=10. SR* must stay finite and increase past the small-N regime.
        small = expected_max_sharpe(10, 0.0024)
        prod = expected_max_sharpe(8000, 0.0024)
        assert math.isfinite(prod) and prod > small


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio (unchanged core; kept under test)
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatio:
    def test_sr_equals_sr_star_gives_half(self):
        sr_star = expected_max_sharpe(10, 1.0)
        dsr = deflated_sharpe_ratio(sr_star, 125, 0.0, 3.0, sr_star)
        assert dsr == pytest.approx(0.5, abs=1e-12)

    def test_sr_far_above_goes_to_one(self):
        sr_star = expected_max_sharpe(10, 1.0)
        dsr = deflated_sharpe_ratio(sr_star + 2.0, 125, 0.0, 3.0, sr_star)
        assert dsr > 0.999

    def test_sr_far_below_goes_to_zero(self):
        sr_star = expected_max_sharpe(10, 1.0)
        dsr = deflated_sharpe_ratio(sr_star - 2.0, 125, 0.0, 3.0, sr_star)
        assert dsr < 1e-6

    def test_monotonic_in_sharpe(self):
        sr_star = 0.3
        prev = -1.0
        for sr in np.linspace(-0.5, 1.5, 25):
            d = deflated_sharpe_ratio(sr, 125, 0.0, 3.0, sr_star)
            assert d >= prev - 1e-12, "DSR must be non-decreasing in SR"
            prev = d

    def test_negative_skew_lowers_dsr(self):
        sr_star, sr = 0.05, 0.20
        d_symmetric = deflated_sharpe_ratio(sr, 125, 0.0, 3.0, sr_star)
        d_negskew = deflated_sharpe_ratio(sr, 125, -1.5, 3.0, sr_star)
        assert d_negskew < d_symmetric

    def test_excess_kurtosis_lowers_dsr(self):
        sr_star, sr = 0.05, 0.20
        d_normal = deflated_sharpe_ratio(sr, 125, 0.0, 3.0, sr_star)
        d_heavy = deflated_sharpe_ratio(sr, 125, 0.0, 9.0, sr_star)
        assert d_heavy < d_normal

    def test_min_days_guard(self):
        assert deflated_sharpe_ratio(1.0, 1, 0.0, 3.0, 0.1) == 0.0
        assert deflated_sharpe_ratio(1.0, 0, 0.0, 3.0, 0.1) == 0.0
        assert deflated_sharpe_ratio(1.0, 2, 0.0, 3.0, 0.1) > 0.0

    def test_sqrt_argument_clamped(self):
        d = deflated_sharpe_ratio(5.0, 125, 5.0, 0.0, 0.1)
        assert math.isfinite(d) and 0.0 <= d <= 1.0

    def test_matches_reference_formula(self):
        sr, T, skew, kurt_ne, sr_star = 0.30, 125, -0.8, 4.0, 0.12
        denom = math.sqrt(max(1.0 - skew * sr + (kurt_ne - 1) / 4.0 * sr * sr, 1e-12))
        ref = norm.cdf((sr - sr_star) * math.sqrt(T - 1) / denom)
        got = deflated_sharpe_ratio(sr, T, skew, kurt_ne, sr_star)
        assert got == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# evolution_n_v_from_records — the (N, V) source (NEW)
# ---------------------------------------------------------------------------

def _records(sharpes_ann, n_days=125):
    """Build an evolution trial-records dict {sig: (sharpe_ann, n_days)}."""
    return {f"sig{i}": (float(s), int(n_days)) for i, s in enumerate(sharpes_ann)}


class TestEvolutionNVFromRecords:
    def test_n_is_distinct_count(self):
        recs = _records([0.1, 0.2, 0.3, 0.4, 0.5])
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 5

    def test_v_is_population_variance_of_daily_sharpes(self):
        ann = [0.5, 1.0, 1.5, 2.0, 0.8, 1.2]
        recs = _records(ann)
        n, v = evolution_n_v_from_records(recs, min_days=20)
        daily = np.array([s / _SQRT_YEAR for s in ann])
        assert v == pytest.approx(float(np.var(daily, ddof=0)), rel=1e-12)

    def test_short_sample_members_excluded_from_N_and_V(self):
        # Finding: V/N must exclude members failing min_days. A short-sample
        # member with a huge Sharpe must NOT pad N or inflate V.
        recs = {**_records([0.5, 1.0, 1.5], n_days=125)}
        recs["short_spike"] = (50.0, 5)  # n_days=5 < min_days
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 3, "short-sample member must be excluded from N"
        daily = np.array([0.5 / _SQRT_YEAR, 1.0 / _SQRT_YEAR, 1.5 / _SQRT_YEAR])
        assert v == pytest.approx(float(np.var(daily, ddof=0)), rel=1e-12), \
            "short-sample member must be excluded from V"

    def test_fewer_than_two_valid_gives_zero_variance(self):
        recs = {"a": (1.0, 5), "b": (2.0, 5)}  # all short
        n, v = evolution_n_v_from_records(recs, min_days=20)
        assert n == 0 and v == 0.0
        recs2 = {"a": (1.0, 125)}  # single valid
        n2, v2 = evolution_n_v_from_records(recs2, min_days=20)
        assert n2 == 1 and v2 == 0.0

    def test_units_daily_not_annualized(self):
        # A units mixup (using annualized Sharpes for V) would be 252x too big.
        ann = [1.0, 2.0, 3.0, 4.0]
        recs = _records(ann)
        _, v_daily = evolution_n_v_from_records(recs, min_days=20)
        v_ann = float(np.var(np.array(ann), ddof=0))
        assert v_daily == pytest.approx(v_ann / TRADING_DAYS_PER_YEAR, rel=1e-12)


# ---------------------------------------------------------------------------
# dsr_gate_evolution — the FILTER gate using evolution-level (N, V) (NEW)
# ---------------------------------------------------------------------------

def _member(sharpe_ann, n_days=125, skew=0.0, kurtosis_excess=0.0):
    """Front member dict with an ANNUALIZED (training) Sharpe."""
    return {
        "sharpe": sharpe_ann,
        "n_days": n_days,
        "skew": skew,
        "kurtosis": kurtosis_excess,  # excess (scipy default)
    }


class TestDsrGateEvolution:
    def test_sr_star_uses_supplied_N_and_V_once(self):
        # SR* must come from the SUPPLIED (N, V), computed once — NOT the
        # front's own variance. Two different fronts with the SAME (N, V) get
        # the SAME SR*.
        N, V = 8000, 0.0024
        f1 = [_member(s) for s in [0.5, 1.0, 1.5]]
        f2 = [_member(s) for s in [0.6, 9.4, 0.2]]  # very different spread
        r1 = dsr_gate_evolution(f1, n_trials=N, sharpe_variance=V, threshold=0.5)
        r2 = dsr_gate_evolution(f2, n_trials=N, sharpe_variance=V, threshold=0.5)
        assert r1["sr_star"] == pytest.approx(r2["sr_star"], rel=1e-12)
        assert r1["sr_star"] == pytest.approx(
            expected_max_sharpe(N, V), rel=1e-12)
        assert r1["n_trials"] == N and r1["sharpe_variance"] == V

    def test_units_daily_de_annualized(self):
        N, V = 5000, 0.0024
        front = [_member(2.0, n_days=125, skew=-1.0, kurtosis_excess=3.0)]
        res = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                 threshold=0.5)
        sr_star = expected_max_sharpe(N, V)
        # Member SR is de-annualized; kurtosis excess -> non-excess (+3).
        expected = deflated_sharpe_ratio(2.0 / _SQRT_YEAR, 125, -1.0, 6.0, sr_star)
        assert res["dsr"][0] == pytest.approx(expected, rel=1e-12)
        assert res["sr_star_annualized"] == pytest.approx(
            sr_star * _SQRT_YEAR, rel=1e-12)

    def test_inflated_member_must_still_clear_sr_star(self):
        # FIXED (was backwards): a champion's TRAINING Sharpe is DEFLATED
        # AGAINST THE POPULATION SR* (computed once from the whole evolution's
        # N trials and their variance V) — it can no longer inflate the bar with
        # its own variance to clear it. A weak training Sharpe (0.8), below the
        # luck-implied SR*, is REJECTED; a robust training Sharpe (3.0) clears
        # the same population SR* and passes; a modest one (1.2) is rejected. No
        # member gets special treatment; (N, V) come from the whole evolution.
        N, V = evolution_n_v_from_records(
            _records([0.3] * 600 + [9.4] + list(np.linspace(-1, 2, 600))),
            min_days=20,
        )
        front = [
            _member(0.8, n_days=REAL_T_TYPICAL, skew=-1.0, kurtosis_excess=3.0),  # weak train SR — below SR*
            _member(3.0, n_days=REAL_T_TYPICAL, skew=-1.0, kurtosis_excess=3.0),  # robust champion
            _member(1.2, n_days=REAL_T_TYPICAL, skew=-1.0, kurtosis_excess=3.0),  # modest
        ]
        res = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                 threshold=PROPOSED_THRESHOLD,
                                 min_days=20)
        assert res["passed"][0] is False, "weak-train-Sharpe member must be REJECTED"
        assert res["passed"][1] is True, "robust champion must pass"
        assert res["passed"][2] is False, "modest member must be rejected"
        assert res["dsr"][0] < res["dsr"][1]

    def test_robust_champion_passes_at_proposed_threshold_and_real_T(self):
        # A genuinely-strong champion (ann train Sharpe ~3.0) over the REAL fold
        # length passes at the proposed threshold against a realistic
        # production-scale (N, V).
        N, V = evolution_n_v_from_records(
            _records(list(np.random.default_rng(1).normal(0.3, 0.75, 4000))),
            min_days=20,
        )
        assert N >= 1000, "this test must exercise the production N regime"
        for T in (REAL_T_MIN, REAL_T_TYPICAL):
            front = [_member(3.0, n_days=T, skew=-1.0, kurtosis_excess=3.0)]
            res = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                     threshold=PROPOSED_THRESHOLD, min_days=20)
            assert res["passed"][0] is True, (
                f"robust champion (ann-Sharpe 3.0) must pass at T={T}, "
                f"SR*_ann={res['sr_star_annualized']:.2f}, DSR={res['dsr'][0]:.4f}"
            )

    def test_gate_filters_front_shrinks(self):
        # The gate's pass mask must drop sub-SR* champions. Mixed front: a few
        # strong, several weak -> n_passed < len(front).
        N, V = 8000, 0.0024
        front = (
            [_member(3.5, n_days=125) for _ in range(2)]    # strong
            + [_member(0.5, n_days=125) for _ in range(6)]  # weak -> deflated away
        )
        res = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                 threshold=0.5, min_days=20)
        assert 0 < res["n_passed"] < len(front), (
            f"gate must FILTER: {res['n_passed']} passed of {len(front)}"
        )
        assert sum(res["passed"]) == res["n_passed"]

    def test_higher_threshold_is_stricter(self):
        # At fixed (N, V, T), raising the threshold can only shrink the pass
        # set (adds short-sample significance the window may not supply).
        N, V = 8000, 0.0024
        front = [_member(s, n_days=125, skew=-1.0, kurtosis_excess=3.0)
                 for s in [2.5, 3.0, 3.5, 4.0, 5.0]]
        n_low = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                   threshold=0.5)["n_passed"]
        n_high = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                    threshold=0.95)["n_passed"]
        assert n_high <= n_low

    def test_min_days_guard_fails_short_members(self):
        N, V = 8000, 0.0024
        front = [
            _member(9.4, n_days=10),   # below min_days -> auto-fail
            _member(9.4, n_days=250),  # enough sample
        ]
        res = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V,
                                 threshold=0.5, min_days=20)
        assert res["dsr"][0] == 0.0
        assert res["passed"][0] is False
        assert res["passed"][1] is True

    def test_determinism(self):
        N, V = 8000, 0.0024
        front = [_member(s, skew=-0.5, kurtosis_excess=3.0)
                 for s in [0.5, 1.0, 1.5, 2.0, 9.4]]
        r1 = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V, threshold=0.5)
        r2 = dsr_gate_evolution(front, n_trials=N, sharpe_variance=V, threshold=0.5)
        assert r1["dsr"] == r2["dsr"]
        assert r1["passed"] == r2["passed"]
        assert r1["sr_star"] == r2["sr_star"]

    def test_empty_front(self):
        res = dsr_gate_evolution([], n_trials=8000, sharpe_variance=0.0024,
                                 threshold=0.5)
        assert res["n_trials"] == 8000  # supplied N is preserved
        assert res["dsr"] == []
        assert res["passed"] == []
        assert res["n_passed"] == 0

    def test_zero_variance_population(self):
        # V=0 -> SR*=0 -> a member with positive (de-annualized) train Sharpe and
        # enough data passes purely on (SR - 0) significance.
        front = [_member(2.0, n_days=125)]
        res = dsr_gate_evolution(front, n_trials=8000, sharpe_variance=0.0,
                                 threshold=0.5, min_days=20)
        assert res["sr_star"] == 0.0
        assert res["dsr"][0] > 0.5

    def test_kurtosis_excess_flag(self):
        N, V = 8000, 0.0024
        front_excess = [_member(2.0, kurtosis_excess=2.0)]
        res_excess = dsr_gate_evolution(front_excess, n_trials=N,
                                        sharpe_variance=V, kurtosis_is_excess=True)
        front_nonexcess = [{"sharpe": 2.0, "n_days": 125, "skew": 0.0,
                            "kurtosis": 5.0}]  # 2.0 excess == 5.0 non-excess
        res_nonexcess = dsr_gate_evolution(front_nonexcess, n_trials=N,
                                           sharpe_variance=V,
                                           kurtosis_is_excess=False)
        assert res_excess["dsr"][0] == pytest.approx(
            res_nonexcess["dsr"][0], rel=1e-12)


# ---------------------------------------------------------------------------
# Threshold calibration sanity (documents the proposed default)
# ---------------------------------------------------------------------------

class TestThresholdCalibration:
    def test_pure_deflation_bar_at_tau_half(self):
        # At τ=0.5 the implied ANNUALIZED champion bar equals SR*_ann exactly
        # (DSR=0.5 <=> SR_daily == SR*). For realistic production (N, V) this
        # bar is a strong, defensible ~2.5-3.0 annualized Sharpe — the
        # justification for τ=0.5 as the calibrated default at short fold T.
        N, V = evolution_n_v_from_records(
            _records(list(np.random.default_rng(2).normal(0.3, 0.75, 8000))),
            min_days=20,
        )
        sr_star_ann = expected_max_sharpe(N, V) * _SQRT_YEAR
        assert 2.0 < sr_star_ann < 4.0, (
            f"pure-deflation bar SR*_ann={sr_star_ann:.2f} should be a strong, "
            f"defensible champion Sharpe (~2.5-3.0) at production (N, V)"
        )
        # A champion exactly at the bar gets DSR≈0.5 at the same T.
        d = deflated_sharpe_ratio(sr_star_ann / _SQRT_YEAR, REAL_T_TYPICAL,
                                  0.0, 3.0, expected_max_sharpe(N, V))
        assert d == pytest.approx(0.5, abs=1e-9)

    def test_095_bar_is_unreachable_at_short_T(self):
        # The retired 0.95 default demands an ann-Sharpe far above the pure-
        # deflation bar at T≈76-125 — confirming why it is uncalibrated here.
        N, V = 8000, 0.0024
        sr_star = expected_max_sharpe(N, V)
        # Find the ann-Sharpe needed for DSR=0.95 at T=125 (symmetric returns).
        z = norm.ppf(0.95)
        # DSR=Φ[(sr-sr*)√(T-1)/√(1+sr²/2)] ; solve by fixed point.
        sr = sr_star
        for _ in range(200):
            denom = math.sqrt(1 + sr * sr / 2.0)
            sr = sr_star + z * denom / math.sqrt(REAL_T_TYPICAL - 1)
        bar_095_ann = sr * _SQRT_YEAR
        bar_050_ann = sr_star * _SQRT_YEAR
        assert bar_095_ann > bar_050_ann + 2.0, (
            f"0.95 bar (ann {bar_095_ann:.2f}) demands much more than the "
            f"pure-deflation 0.5 bar (ann {bar_050_ann:.2f}) at T={REAL_T_TYPICAL}"
        )


# ---------------------------------------------------------------------------
# Legacy is gone — guard against re-introduction.
# ---------------------------------------------------------------------------

class TestLegacyRemoved:
    def test_front_variance_gate_removed(self):
        import layer2.pbo as pbo
        assert not hasattr(pbo, "dsr_gate_front"), (
            "the front-variance gate (variance-inversion bug) must be deleted"
        )

    def test_legacy_experiment_helpers_removed(self):
        import layer2.experiment as exp
        for name in ("_expected_max_sr", "deflated_sharpe_ratio",
                     "deflated_sharpe_threshold"):
            assert not hasattr(exp, name), (
                f"legacy units-buggy helper experiment.{name} must be deleted "
                f"(DSR math lives only in layer2/pbo.py now)"
            )
