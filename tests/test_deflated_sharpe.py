"""Tests for the Deflated Sharpe Ratio implementation (migrated 2026-05-30).

P1-B redesign: the DSR math now lives ONLY in layer2/pbo.py (single
implementation). The legacy in-module helpers in layer2.experiment
(_expected_max_sr / deflated_sharpe_ratio / deflated_sharpe_threshold) were
DELETED — they compared an ANNUALIZED Sharpe against a z-scale SR* (V≡1), a
units bug that was silently over-strict. These tests now exercise the pbo
functions (DAILY units) and the repointed cross-fold survival gate G1.

Validates:
1. pbo.expected_max_sharpe (SR*) monotonic in N and V
2. pbo.deflated_sharpe_ratio non-normality corrections (skew/kurtosis/T)
3. BacktestResult.return_skew / .return_kurtosis populated correctly
4. Cross-fold survival gate G1 (_apply_proxy_survival_gates) uses pbo DSR
"""
import math

import numpy as np
import pytest

from layer2.pbo import (
    TRADING_DAYS_PER_YEAR,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)

_SQRT_YEAR = math.sqrt(TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# pbo.expected_max_sharpe (SR* under null) — replaces legacy _expected_max_sr
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    def test_monotonically_increasing_in_n(self):
        # At fixed V, SR* increases with N.
        vals = [expected_max_sharpe(n, 1.0) for n in [10, 100, 1000, 10000]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_n2_gives_reasonable_value(self):
        # At V=1.0, SR* == the bracket term; for N=2 it is modest.
        val = expected_max_sharpe(2, 1.0)
        assert 0.3 < val < 1.5, f"SR*(N=2, V=1) = {val}"

    def test_large_n_matches_sqrt2logn(self):
        # For large N and V=1, SR* ≈ sqrt(2*log(N)) (leading term).
        n = 100000
        val = expected_max_sharpe(n, 1.0)
        leading = math.sqrt(2 * math.log(n))
        assert abs(val - leading) / leading < 0.15

    def test_scales_with_sqrt_v(self):
        assert expected_max_sharpe(100, 4.0) == pytest.approx(
            2.0 * expected_max_sharpe(100, 1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# pbo.deflated_sharpe_ratio (per-strategy DSR, DAILY units)
# ---------------------------------------------------------------------------

class TestDeflatedSharpeRatio:
    def test_high_sharpe_low_sr_star_passes(self):
        # Genuinely good daily Sharpe well above a small SR* -> DSR high.
        sr_star = expected_max_sharpe(10, 0.001)  # tight dispersion -> tiny SR*
        dsr = deflated_sharpe_ratio(0.30, 250, 0.0, 3.0, sr_star)
        assert dsr > 0.95

    def test_low_sharpe_high_sr_star_fails(self):
        # Mediocre daily Sharpe below a large SR* (many high-V trials) -> low DSR.
        sr_star = expected_max_sharpe(100000, 1.0)  # huge SR*
        dsr = deflated_sharpe_ratio(0.05, 125, 0.0, 3.0, sr_star)
        assert dsr < 0.05

    def test_negative_sharpe_gives_near_zero(self):
        sr_star = expected_max_sharpe(100, 0.01)
        dsr = deflated_sharpe_ratio(-0.10, 125, 0.0, 3.0, sr_star)
        assert dsr < 0.01

    def test_heavy_tails_reduce_dsr(self):
        sr_star = 0.10
        dsr_normal = deflated_sharpe_ratio(0.20, 250, 0.0, 3.0, sr_star)
        dsr_heavy = deflated_sharpe_ratio(0.20, 250, 0.0, 9.0, sr_star)  # heavy
        assert dsr_heavy < dsr_normal

    def test_negative_skew_reduces_dsr(self):
        sr_star = 0.10
        dsr_symmetric = deflated_sharpe_ratio(0.20, 250, 0.0, 3.0, sr_star)
        dsr_negskew = deflated_sharpe_ratio(0.20, 250, -2.0, 3.0, sr_star)
        assert dsr_negskew < dsr_symmetric

    def test_more_data_increases_dsr(self):
        sr_star = 0.10
        dsr_short = deflated_sharpe_ratio(0.20, 50, 0.0, 3.0, sr_star)
        dsr_long = deflated_sharpe_ratio(0.20, 500, 0.0, 3.0, sr_star)
        assert dsr_long > dsr_short

    def test_returns_probability_in_01(self):
        for sr in [-0.5, -0.1, 0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for sr_star in [0.0, 0.1, 0.5]:
                dsr = deflated_sharpe_ratio(sr, 125, -1.0, 6.0, sr_star)
                assert 0.0 <= dsr <= 1.0

    def test_t_days_one_returns_zero(self):
        assert deflated_sharpe_ratio(0.2, 1, 0.0, 3.0, 0.1) == 0.0


# ---------------------------------------------------------------------------
# BacktestResult return moments (unchanged — still consumed by the gate)
# ---------------------------------------------------------------------------

class TestBacktestResultSkewKurtosis:
    def test_fields_exist_with_defaults(self):
        from layer2.evaluator import BacktestResult
        r = BacktestResult(
            returns=np.zeros(10), trades=[], equity_curve=np.zeros(10),
            max_drawdown=0.0, sharpe=0.0,
        )
        assert r.return_skew == 0.0
        assert r.return_kurtosis == 0.0

    def test_fields_set_correctly(self):
        from layer2.evaluator import BacktestResult
        r = BacktestResult(
            returns=np.zeros(10), trades=[], equity_curve=np.zeros(10),
            max_drawdown=0.0, sharpe=0.0,
            return_skew=-1.5, return_kurtosis=4.2,
        )
        assert r.return_skew == -1.5
        assert r.return_kurtosis == 4.2


# ---------------------------------------------------------------------------
# Cross-fold survival gate G1 — repointed to pbo (DAILY units)
# ---------------------------------------------------------------------------

class TestSurvivalGateDSR:
    def test_g1_rejects_on_dsr(self):
        """A clearly sub-bar candidate (negative Sharpe + heavy tails) is
        deflated away by G1 while strong candidates survive.

        H1 fix (review): G1's SR* now uses N_eff = the candidate-POOL size (the
        population whose variance V it measures), NOT the full-search N =
        pop×templates — `expected_max_sharpe(N, V)` is only valid when N and V
        describe ONE population, and the full-search multiple-testing inflation
        is already absorbed UPSTREAM by the per-template front DSR gate. So G1 is
        the LIGHT secondary cross-fold screen: with a small candidate pool its
        bar is legitimately mild (SR*_ann ≈ 1.3 for ~6 candidates), and it
        rejects only candidates clearly below that bar — a modest 0.8-ann Sharpe
        is no longer distinguishable from best-of-pool luck at the 0.05 level
        (that rejection, and OOS-collapse rejection generally, is carried by the
        front gate + G10 walk-forward efficiency). `n_trials` is now only the
        per-strategy-vs-static enable toggle; SR* uses N_eff = len(pool)."""
        from layer2.experiment import _apply_proxy_survival_gates
        border = {
            "val_sharpe": -1.5, "n_trades_val": 200, "folds_positive": 3,
            "structurally_similar": True,
            "val_return_skew": -3.0, "val_return_kurtosis": 10.0,
            "template_name": "border",
        }
        strong = [{
            "val_sharpe": v, "n_trades_val": 200, "folds_positive": 3,
            "structurally_similar": True,
            "val_return_skew": 0.0, "val_return_kurtosis": 0.0,
            "template_name": f"strong{i}",
        } for i, v in enumerate([3.0, 3.2, 3.4, 3.6, 3.8])]
        strategies = [border] + strong
        survivors = _apply_proxy_survival_gates(
            strategies, n_trials=100000, n_val_days=125)
        # ANNOTATE-DON'T-DESTROY (2026-06-01): the negative-Sharpe heavy-tailed
        # candidate is no longer DROPPED — it is KEPT in the front but TAGGED
        # non-viable. Intent preserved (a negative/insignificant Sharpe is not a
        # deployable edge), now expressed via tags: it is present in the returned
        # front, its DSR is still deflated below 0.05, and both positivity_passed
        # and the composite survival_passed are False.
        assert border in survivors, "annotate-don't-destroy keeps the border candidate"
        assert border["val_dsr"] < 0.05, "negative-Sharpe heavy-tailed must fail G1 DSR"
        assert border["positivity_passed"] is False, \
            "negative val_sharpe must fail the positivity tag"
        assert border["survival_passed"] is False, \
            "a negative-Sharpe candidate is not viable (survival_passed=False)"
        # The strong champions (positive Sharpe) remain viable and are kept.
        assert all(s in survivors for s in strong)
        assert all(s["survival_passed"] is True for s in strong)
        # The bar is the consistent-population one (N_eff = pool size = 6): the
        # strong 3.0-ann champion is viable (its tag is True), NOT deflated away.
        assert any(abs(s["val_sharpe"] - 3.0) < 1e-9 and s["survival_passed"]
                   for s in survivors), \
            "a 3.0-ann champion must remain viable under the light secondary G1 bar"

    def test_g1_passes_strong_strategy(self):
        """A strong strategy with normal returns survives G1."""
        from layer2.experiment import _apply_proxy_survival_gates
        strategies = [{
            "val_sharpe": 3.0,
            "n_trades_val": 200,
            "folds_positive": 3,
            "structurally_similar": True,
            "val_return_skew": 0.0,
            "val_return_kurtosis": 0.0,
        }]
        survivors = _apply_proxy_survival_gates(
            strategies, n_trials=1000, n_val_days=125)
        assert len(survivors) == 1

    def test_g1_attaches_dsr_value(self):
        from layer2.experiment import _apply_proxy_survival_gates
        strategies = [{
            "val_sharpe": 3.0, "n_trades_val": 200, "folds_positive": 3,
            "structurally_similar": True,
            "val_return_skew": 0.0, "val_return_kurtosis": 0.0,
        }]
        _apply_proxy_survival_gates(strategies, n_trials=1000, n_val_days=125)
        assert "val_dsr" in strategies[0]
        assert 0.0 <= strategies[0]["val_dsr"] <= 1.0

    def test_g1_fallback_without_skew_kurtosis(self):
        from layer2.experiment import _apply_proxy_survival_gates
        strategies = [{
            "val_sharpe": 3.0, "n_trades_val": 200, "folds_positive": 3,
            "structurally_similar": True,
        }]
        survivors = _apply_proxy_survival_gates(
            strategies, n_trials=1000, n_val_days=125)
        assert len(survivors) == 1

    def test_g1_static_fallback_when_n_trials_zero(self):
        from layer2.experiment import _apply_proxy_survival_gates, SURVIVAL_GATES
        strategies = [{
            "val_sharpe": SURVIVAL_GATES["G1_haircut_val_sharpe"] + 0.01,
            "n_trades_val": 200, "folds_positive": 3,
            "structurally_similar": True,
        }]
        survivors = _apply_proxy_survival_gates(
            strategies, n_trials=0, n_val_days=125)
        assert len(survivors) == 1


# ---------------------------------------------------------------------------
# Realistic 0DTE scenario (pbo, DAILY units)
# ---------------------------------------------------------------------------

class TestDSRRealisticScenario:
    def test_0dte_credit_spread_typical_fails(self):
        """Typical 0DTE credit spread: modest Sharpe + negative skew + heavy
        tails, deflated against a production-scale SR* -> fails the τ=0.5 bar.
        (SR*_ann ≈ 3.0; an ann-1.2 champion sits BELOW the pure-deflation bar.)"""
        sr_star = expected_max_sharpe(8000, 0.0024)  # production-scale SR*
        dsr = deflated_sharpe_ratio(
            sharpe_daily=1.2 / _SQRT_YEAR,  # ann 1.2 -> daily
            n_days=125, skew=-1.8, kurtosis=5.5 + 3.0, sr_star=sr_star,
        )
        assert dsr < 0.5, f"SR_ann=1.2 should fail τ=0.5 vs SR*_ann={sr_star*_SQRT_YEAR:.2f}"

    def test_genuinely_profitable_survives(self):
        """A genuinely strong champion (ann 3.5) clears the production SR* and
        the τ=0.5 bar even with negative skew + heavy tails."""
        sr_star = expected_max_sharpe(8000, 0.0024)
        dsr = deflated_sharpe_ratio(
            sharpe_daily=3.5 / _SQRT_YEAR,
            n_days=125, skew=-1.0, kurtosis=4.0 + 3.0, sr_star=sr_star,
        )
        assert dsr > 0.50, f"SR_ann=3.5 should survive vs SR*_ann={sr_star*_SQRT_YEAR:.2f}"
