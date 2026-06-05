"""Unit tests for GP-relevant forward targets computation.

Tests the target computation logic from scripts/build_gp_targets_and_refit_pls.py
using synthetic data (no corpus dependency). Validates:
  1. next_bar_ret is computed as a DIFFERENCE (not log ratio)
  2. MFE-5 is the MAX of 5 forward differences
  3. MAE-5 is the MIN of 5 forward differences
  4. Regime one-hot sums to 1 when IV is not NaN
  5. Targets at end-of-day (insufficient forward bars) are marked invalid
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Target computation extracted from build_gp_targets_and_refit_pls.py
# (standalone, no corpus or pipeline dependency)
# ---------------------------------------------------------------------------

def compute_single_window_targets(spx_series, iv_series, end_bar, n_bars):
    """Compute GP targets for a single window ending at end_bar.

    Replicates the logic from build_gp_targets_and_refit_pls.py lines 97-155
    exactly, using the same variable names and logic flow.

    Returns: (target_array_7d, is_valid)
    """
    forward_end = end_bar + 5

    if forward_end >= n_bars:
        return np.zeros(7, dtype=np.float32), False

    price_t = spx_series[end_bar]
    iv_t = iv_series[end_bar]

    price_forward = spx_series[end_bar + 1:forward_end + 1]
    iv_forward_5 = iv_series[forward_end]

    if np.isnan(price_t) or len(price_forward) < 5:
        return np.zeros(7, dtype=np.float32), False

    # v105 is session_log_return = ln(close/open), a cumulative intraday
    # log-return starting at 0. Forward targets are DIFFERENCES of this
    # series (incremental log-returns), NOT ratios.

    # Target 1: next-bar incremental log-return
    next_bar_ret = float(price_forward[0] - price_t)

    # Target 2: MFE-5
    excursions = price_forward - price_t
    mfe_5 = float(np.nanmax(excursions))

    # Target 3: MAE-5
    mae_5 = float(np.nanmin(excursions))

    # Target 4: IV change
    iv_change = float(iv_forward_5 - iv_t) if not np.isnan(iv_forward_5) else 0.0

    # Targets 5-7: regime one-hot
    regime_oh = np.zeros(3, dtype=np.float32)
    if not np.isnan(iv_t):
        if iv_t < 0.10:
            regime_oh[0] = 1.0
        elif iv_t < 0.16:
            regime_oh[1] = 1.0
        else:
            regime_oh[2] = 1.0

    target = np.array([next_bar_ret, mfe_5, mae_5, iv_change,
                        regime_oh[0], regime_oh[1], regime_oh[2]], dtype=np.float32)

    valid = not (np.isnan(target[:4]).any() or np.isinf(target[:4]).any())
    return target, valid


# ---------------------------------------------------------------------------
# Test 1: next_bar_ret is a DIFFERENCE (not log ratio)
# ---------------------------------------------------------------------------

class TestNextBarReturn:
    def test_next_bar_ret_is_difference(self):
        """next_bar_ret = price_forward[0] - price_t, NOT log(forward/current)."""
        # session_log_return at bars: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, ...]
        spx = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)

        # Window ends at bar 3 (price_t = 0.04)
        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid

        # next_bar_ret should be 0.05 - 0.04 = 0.01 (difference)
        # NOT log(0.05/0.04) = 0.223
        expected_diff = 0.05 - 0.04
        assert abs(target[0] - expected_diff) < 1e-6, (
            f"next_bar_ret={target[0]:.8f}, expected difference={expected_diff:.8f}"
        )

    def test_next_bar_ret_negative(self):
        """Verify negative returns work (price decreasing)."""
        spx = np.array([0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=2, n_bars=10)
        assert valid
        expected = 0.07 - 0.08  # = -0.01
        assert abs(target[0] - expected) < 1e-6

    def test_next_bar_ret_zero(self):
        """Zero return when price unchanged."""
        spx = np.full(10, 0.05, dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=2, n_bars=10)
        assert valid
        assert abs(target[0]) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: MFE-5 is the MAX of 5 forward differences
# ---------------------------------------------------------------------------

class TestMFE5:
    def test_mfe_is_max_of_forward_differences(self):
        """MFE-5 = max(forward[0:5] - price_t), computed as MAX of differences."""
        # Price series: starts at 1.0, rises then falls
        spx = np.array([1.0, 1.0, 1.0,   # bars 0-2
                         1.0,               # bar 3 = end_bar, price_t = 1.0
                         1.01, 1.03, 1.02, 0.99, 1.05,  # bars 4-8 (forward)
                         1.0],              # bar 9
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)

        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid

        # Forward differences: [1.01-1.0, 1.03-1.0, 1.02-1.0, 0.99-1.0, 1.05-1.0]
        # = [0.01, 0.03, 0.02, -0.01, 0.05]
        # MFE-5 = max = 0.05
        expected_mfe = 0.05
        assert abs(target[1] - expected_mfe) < 1e-6, (
            f"MFE-5={target[1]:.8f}, expected={expected_mfe:.8f}"
        )

    def test_mfe_all_negative_forward(self):
        """When all forward prices are below current, MFE is the least negative."""
        spx = np.array([0.0, 0.0, 0.0,
                         1.0,  # end_bar=3
                         0.99, 0.98, 0.97, 0.96, 0.95,  # all below
                         0.0],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid
        # Differences: [-0.01, -0.02, -0.03, -0.04, -0.05]
        # MFE = max = -0.01
        expected = -0.01
        assert abs(target[1] - expected) < 1e-6

    def test_mfe_is_not_ratio(self):
        """Verify MFE uses differences, not ratios (price_forward - price_t,
        NOT price_forward / price_t)."""
        spx = np.array([0.0, 0.0, 0.0,
                         2.0,  # end_bar=3
                         2.1, 2.2, 2.3, 2.4, 2.5,
                         0.0],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid
        # MFE = max(differences) = 2.5 - 2.0 = 0.5
        # If ratio: max(ratios) = 2.5/2.0 = 1.25 (very different)
        expected_diff = 0.5
        assert abs(target[1] - expected_diff) < 1e-6


# ---------------------------------------------------------------------------
# Test 3: MAE-5 is the MIN of 5 forward differences
# ---------------------------------------------------------------------------

class TestMAE5:
    def test_mae_is_min_of_forward_differences(self):
        """MAE-5 = min(forward[0:5] - price_t), computed as MIN of differences."""
        spx = np.array([1.0, 1.0, 1.0,
                         1.0,  # end_bar=3
                         1.01, 0.97, 1.02, 0.95, 1.05,
                         1.0],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)

        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid

        # Differences: [0.01, -0.03, 0.02, -0.05, 0.05]
        # MAE-5 = min = -0.05
        expected_mae = -0.05
        assert abs(target[2] - expected_mae) < 1e-6, (
            f"MAE-5={target[2]:.8f}, expected={expected_mae:.8f}"
        )

    def test_mae_all_positive_forward(self):
        """When all forward prices are above current, MAE is the smallest gain."""
        spx = np.array([0.0, 0.0, 0.0,
                         1.0,  # end_bar=3
                         1.01, 1.02, 1.03, 1.04, 1.05,
                         0.0],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid
        expected = 0.01  # min of [0.01, 0.02, 0.03, 0.04, 0.05]
        assert abs(target[2] - expected) < 1e-6

    def test_mae_is_not_ratio(self):
        """Verify MAE uses differences, not ratios."""
        spx = np.array([0.0, 0.0, 0.0,
                         2.0,
                         1.9, 1.8, 1.7, 1.6, 1.5,
                         0.0],
                        dtype=np.float64)
        iv = np.full(10, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=3, n_bars=10)
        assert valid
        # MAE = min(differences) = 1.5 - 2.0 = -0.5
        # If ratio: min(ratios) = 1.5/2.0 = 0.75 (very different)
        expected_diff = -0.5
        assert abs(target[2] - expected_diff) < 1e-6

    def test_mfe_gte_mae(self):
        """MFE should always be >= MAE (max >= min of same set)."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            spx = np.cumsum(rng.randn(15) * 0.01) + 5.0
            iv = np.full(15, 0.12, dtype=np.float64)
            target, valid = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
            if valid:
                assert target[1] >= target[2], (
                    f"MFE ({target[1]}) < MAE ({target[2]}) violates max >= min"
                )


# ---------------------------------------------------------------------------
# Test 4: Regime one-hot sums to 1 when IV is not NaN
# ---------------------------------------------------------------------------

class TestRegimeOneHot:
    def test_regime_sums_to_1_when_iv_valid(self):
        """When IV is not NaN, exactly one regime class should be 1.0."""
        spx = np.linspace(0.0, 0.1, 15, dtype=np.float64)
        for iv_val in [0.05, 0.09, 0.10, 0.12, 0.15, 0.16, 0.20, 0.30]:
            iv = np.full(15, iv_val, dtype=np.float64)
            target, valid = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
            if valid:
                regime_sum = target[4] + target[5] + target[6]
                assert abs(regime_sum - 1.0) < 1e-6, (
                    f"IV={iv_val}: regime one-hot sums to {regime_sum}, expected 1.0"
                )
                # Exactly one class is 1.0, others 0.0
                assert max(target[4], target[5], target[6]) == 1.0
                assert min(target[4], target[5], target[6]) == 0.0

    def test_regime_all_zero_when_iv_nan(self):
        """When IV is NaN, all regime one-hot values should be 0."""
        spx = np.linspace(0.0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, np.nan, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        # Note: this window is still valid because NaN IV only affects regime,
        # and iv_change gets 0.0 fallback
        regime_sum = target[4] + target[5] + target[6]
        assert regime_sum == 0.0, f"Regime sum should be 0 when IV=NaN, got {regime_sum}"

    def test_regime_low_iv(self):
        """IV < 0.10 maps to regime 0 (low)."""
        spx = np.linspace(0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.05, dtype=np.float64)
        target, _ = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert target[4] == 1.0  # low
        assert target[5] == 0.0  # mid
        assert target[6] == 0.0  # high

    def test_regime_mid_iv(self):
        """0.10 <= IV < 0.16 maps to regime 1 (mid)."""
        spx = np.linspace(0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.12, dtype=np.float64)
        target, _ = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert target[4] == 0.0
        assert target[5] == 1.0  # mid
        assert target[6] == 0.0

    def test_regime_high_iv(self):
        """IV >= 0.16 maps to regime 2 (high)."""
        spx = np.linspace(0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.25, dtype=np.float64)
        target, _ = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert target[4] == 0.0
        assert target[5] == 0.0
        assert target[6] == 1.0  # high

    def test_regime_boundary_at_010(self):
        """IV = 0.10 exactly should be mid (>= 0.10)."""
        spx = np.linspace(0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.10, dtype=np.float64)
        target, _ = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert target[4] == 0.0
        assert target[5] == 1.0  # mid (iv >= 0.10)
        assert target[6] == 0.0

    def test_regime_boundary_at_016(self):
        """IV = 0.16 exactly should be high (>= 0.16)."""
        spx = np.linspace(0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.16, dtype=np.float64)
        target, _ = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert target[4] == 0.0
        assert target[5] == 0.0
        assert target[6] == 1.0  # high (iv >= 0.16)


# ---------------------------------------------------------------------------
# Test 5: End-of-day targets (insufficient forward bars) marked invalid
# ---------------------------------------------------------------------------

class TestEndOfDayInvalid:
    def test_last_5_bars_invalid(self):
        """Windows ending within 5 bars of session end have invalid targets."""
        n_bars = 20
        spx = np.linspace(0.0, 0.1, n_bars, dtype=np.float64)
        iv = np.full(n_bars, 0.12, dtype=np.float64)

        # end_bar=15 needs forward_end=20, n_bars=20 -> forward_end >= n_bars -> invalid
        target, valid = compute_single_window_targets(spx, iv, end_bar=15, n_bars=n_bars)
        assert not valid, "end_bar=15 with n_bars=20 should be invalid (needs bar 20)"

        # end_bar=14 needs forward_end=19, n_bars=20 -> ok
        target, valid = compute_single_window_targets(spx, iv, end_bar=14, n_bars=n_bars)
        assert valid, "end_bar=14 with n_bars=20 should be valid (bar 19 exists)"

    def test_boundary_exactly_at_limit(self):
        """end_bar = n_bars - 6 is the last valid position (forward_end = n_bars - 1)."""
        n_bars = 30
        spx = np.linspace(0.0, 0.2, n_bars, dtype=np.float64)
        iv = np.full(n_bars, 0.12, dtype=np.float64)

        # Last valid: end_bar = 24, forward_end = 29 = n_bars - 1 -> ok
        _, valid = compute_single_window_targets(spx, iv, end_bar=24, n_bars=n_bars)
        assert valid

        # First invalid: end_bar = 25, forward_end = 30 = n_bars -> invalid
        _, valid = compute_single_window_targets(spx, iv, end_bar=25, n_bars=n_bars)
        assert not valid

    def test_invalid_returns_zero_target(self):
        """Invalid windows return a zero target array."""
        n_bars = 10
        spx = np.linspace(0.0, 0.1, n_bars, dtype=np.float64)
        iv = np.full(n_bars, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=8, n_bars=n_bars)
        assert not valid
        assert np.all(target == 0.0)

    def test_nan_price_also_invalid(self):
        """Window with NaN price at end_bar is also invalid."""
        n_bars = 15
        spx = np.linspace(0.0, 0.1, n_bars, dtype=np.float64)
        spx[5] = np.nan  # NaN at end_bar
        iv = np.full(n_bars, 0.12, dtype=np.float64)
        target, valid = compute_single_window_targets(spx, iv, end_bar=5, n_bars=n_bars)
        assert not valid

    def test_multiple_windows_end_of_day(self):
        """Sweep through all end_bar positions and verify validity boundary."""
        n_bars = 20
        spx = np.linspace(0.0, 0.1, n_bars, dtype=np.float64)
        iv = np.full(n_bars, 0.12, dtype=np.float64)
        for end_bar in range(n_bars):
            _, valid = compute_single_window_targets(spx, iv, end_bar=end_bar, n_bars=n_bars)
            expected_valid = (end_bar + 5) < n_bars
            assert valid == expected_valid, (
                f"end_bar={end_bar}: valid={valid}, expected={expected_valid}"
            )


# ---------------------------------------------------------------------------
# Cross-consistency: MFE/MAE/next_bar_ret relationships
# ---------------------------------------------------------------------------

class TestTargetConsistency:
    def test_next_bar_ret_between_mae_and_mfe(self):
        """next_bar_ret (1-bar excursion) must be between MAE-5 and MFE-5
        since it's one of the 5 excursions."""
        rng = np.random.RandomState(123)
        for _ in range(50):
            spx = np.cumsum(rng.randn(20) * 0.01) + 5.0
            iv = np.full(20, 0.12, dtype=np.float64)
            target, valid = compute_single_window_targets(spx, iv, end_bar=8, n_bars=20)
            if valid:
                nbr = target[0]  # next_bar_ret
                mfe = target[1]  # MFE-5
                mae = target[2]  # MAE-5
                assert mae <= nbr <= mfe, (
                    f"next_bar_ret={nbr:.6f} not in [{mae:.6f}, {mfe:.6f}]"
                )

    def test_iv_change_with_nan_forward_iv(self):
        """iv_change should be 0.0 when forward IV is NaN."""
        spx = np.linspace(0.0, 0.1, 15, dtype=np.float64)
        iv = np.full(15, 0.12, dtype=np.float64)
        iv[10] = np.nan  # forward_end bar has NaN IV
        target, valid = compute_single_window_targets(spx, iv, end_bar=5, n_bars=15)
        assert valid
        assert target[3] == 0.0, f"iv_change should be 0.0 when forward IV is NaN"
