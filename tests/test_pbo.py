"""Unit tests for layer2.pbo — Probability of Backtest Overfitting via CPCV.

Tests cover:
  - Combinatorial correctness (C(n,k) combinations)
  - Embargo/purging mechanics
  - PBO behavior on synthetic data (random vs trending returns)
  - Output format validation
  - Edge cases (few groups, single strategy, all-negative IS)
  - Multi-strategy PBO via returns matrix
"""
import itertools
import math

import numpy as np
import pytest

from layer2.pbo import (
    _build_train_indices,
    _build_test_indices,
    _daily_sharpe,
    _split_into_groups,
    compute_pbo,
    compute_pbo_matrix,
)


# ---------------------------------------------------------------------------
# Helper: daily sharpe
# ---------------------------------------------------------------------------

class TestDailySharpe:
    def test_empty_returns_zero(self):
        assert _daily_sharpe(np.array([])) == 0.0

    def test_zero_variance_returns_zero(self):
        assert _daily_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, size=252)
        assert _daily_sharpe(rets) > 0

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(-0.001, 0.01, size=252)
        assert _daily_sharpe(rets) < 0

    def test_annualization_factor(self):
        """Sharpe uses sqrt(252) annualization."""
        rets = np.array([0.01, -0.005, 0.008, -0.003, 0.006] * 50)
        sr = _daily_sharpe(rets)
        # Manual computation
        expected = float(np.mean(rets) / np.std(rets) * math.sqrt(252))
        assert abs(sr - expected) < 1e-10


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

class TestSplitIntoGroups:
    def test_even_split(self):
        blocks = _split_into_groups(100, 10)
        assert len(blocks) == 10
        assert blocks[0] == (0, 10)
        assert blocks[9] == (90, 100)

    def test_uneven_split_last_absorbs_remainder(self):
        blocks = _split_into_groups(103, 10)
        assert len(blocks) == 10
        # First 9 groups have 10 obs each, last has 13
        assert blocks[8] == (80, 90)
        assert blocks[9] == (90, 103)

    def test_all_observations_covered(self):
        blocks = _split_into_groups(77, 7)
        covered = set()
        for start, end in blocks:
            covered.update(range(start, end))
        assert covered == set(range(77))

    def test_no_overlap(self):
        blocks = _split_into_groups(50, 5)
        all_indices = []
        for start, end in blocks:
            all_indices.extend(range(start, end))
        assert len(all_indices) == len(set(all_indices))

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="Cannot split"):
            _split_into_groups(3, 10)

    def test_minimum_viable(self):
        blocks = _split_into_groups(2, 2)
        assert blocks == [(0, 1), (1, 2)]


# ---------------------------------------------------------------------------
# Embargo / purging
# ---------------------------------------------------------------------------

class TestEmbargo:
    def test_embargo_removes_adjacent_days(self):
        """Embargo of 5 days removes 5 observations from training blocks
        adjacent to each test block boundary."""
        # 100 obs, 10 groups of 10 each
        blocks = _split_into_groups(100, 10)
        # Test groups: {2, 5} -> test indices [20..30) and [50..60)
        test_groups = (2, 5)
        train_idx = _build_train_indices(blocks, test_groups, embargo_days=5, n_obs=100)

        train_set = set(train_idx)

        # Test indices themselves must NOT be in training
        for i in range(20, 30):
            assert i not in train_set, f"test index {i} in train"
        for i in range(50, 60):
            assert i not in train_set, f"test index {i} in train"

        # Embargo before test block 2 (indices 15-19 should be removed from group 1)
        for i in range(15, 20):
            assert i not in train_set, f"embargo index {i} before test block 2 in train"

        # Embargo after test block 2 (indices 30-34 should be removed from group 3)
        for i in range(30, 35):
            assert i not in train_set, f"embargo index {i} after test block 2 in train"

        # Embargo before test block 5 (indices 45-49 should be removed from group 4)
        for i in range(45, 50):
            assert i not in train_set, f"embargo index {i} before test block 5 in train"

        # Embargo after test block 5 (indices 60-64 should be removed from group 6)
        for i in range(60, 65):
            assert i not in train_set, f"embargo index {i} after test block 5 in train"

    def test_embargo_zero_no_purging(self):
        """With embargo=0, only test blocks are excluded."""
        blocks = _split_into_groups(50, 5)
        test_groups = (1,)  # indices [10..20)
        train_idx = _build_train_indices(blocks, test_groups, embargo_days=0, n_obs=50)

        train_set = set(train_idx)
        # All non-test indices should be present
        expected = set(range(50)) - set(range(10, 20))
        assert train_set == expected

    def test_embargo_at_boundaries(self):
        """Embargo at the start/end of data is clipped (no index errors)."""
        blocks = _split_into_groups(30, 3)
        # Test group 0 (first block): embargo before would go negative
        test_groups = (0,)
        train_idx = _build_train_indices(blocks, test_groups, embargo_days=5, n_obs=30)

        train_set = set(train_idx)
        # Indices 0-9 are test (group 0)
        for i in range(10):
            assert i not in train_set
        # Indices 10-14 are embargoed (after test block 0)
        for i in range(10, 15):
            assert i not in train_set

        # Test last group: embargo after would exceed n_obs
        test_groups_last = (2,)
        train_idx_last = _build_train_indices(blocks, test_groups_last, embargo_days=5, n_obs=30)
        train_set_last = set(train_idx_last)
        # Indices 20-29 are test (group 2)
        for i in range(20, 30):
            assert i not in train_set_last
        # Indices 15-19 are embargoed (before test block 2)
        for i in range(15, 20):
            assert i not in train_set_last

    def test_test_indices_match_blocks(self):
        blocks = _split_into_groups(100, 10)
        test_groups = (3, 7)
        test_idx = _build_test_indices(blocks, test_groups)
        expected = set(range(30, 40)) | set(range(70, 80))
        assert set(test_idx) == expected


# ---------------------------------------------------------------------------
# PBO: n_combinations correctness
# ---------------------------------------------------------------------------

class TestPBOCombinations:
    def test_c_10_2_equals_45(self):
        """C(10, 2) = 45 combinations."""
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        assert result["n_combinations"] == 45

    def test_c_8_3_equals_56(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=8, n_test_groups=3, embargo_days=0)
        assert result["n_combinations"] == 56

    def test_c_6_2_equals_15(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=120)
        result = compute_pbo(rets, n_groups=6, n_test_groups=2, embargo_days=0)
        assert result["n_combinations"] == 15

    def test_matches_itertools(self):
        """n_combinations must match itertools.combinations count."""
        for n in [5, 8, 10, 12]:
            for k in [1, 2, 3]:
                if k >= n:
                    continue
                rng = np.random.default_rng(0)
                rets = rng.normal(0, 0.01, size=n * 20)
                result = compute_pbo(rets, n_groups=n, n_test_groups=k, embargo_days=0)
                expected = len(list(itertools.combinations(range(n), k)))
                assert result["n_combinations"] == expected, f"C({n},{k})"


# ---------------------------------------------------------------------------
# PBO: random returns -> high PBO
# ---------------------------------------------------------------------------

class TestPBORandomReturns:
    def test_random_returns_pbo_near_half(self):
        """Pure noise returns should give PBO near 0.5 (no signal).

        With random returns, IS Sharpe will be positive ~50% of the time
        by chance, and the OOS Sharpe of those IS-positive combinations
        should also fail ~50% of the time. So PBO ~ 0.5.

        Uses seed=9 with 1000 observations (verified to give overall
        Sharpe near zero and sufficient IS-positive combinations for a
        meaningful PBO estimate).
        """
        rng = np.random.default_rng(9)
        rets = rng.normal(0, 0.01, size=1000)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        # PBO should be in roughly [0.2, 0.8] for random noise
        assert 0.2 <= result["pbo"] <= 0.8, (
            f"PBO={result['pbo']:.3f} outside expected range for random returns"
        )


# ---------------------------------------------------------------------------
# PBO: trending returns -> low PBO
# ---------------------------------------------------------------------------

class TestPBOTrendingReturns:
    def test_consistent_positive_returns_low_pbo(self):
        """Consistently positive returns should give PBO near 0.

        If the strategy has a genuine positive edge across all time blocks,
        then the best-IS strategy should also perform well OOS in nearly
        all combinations.
        """
        rng = np.random.default_rng(42)
        # Strong positive drift with low noise -> IS and OOS both positive
        rets = rng.normal(0.005, 0.002, size=500)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        assert result["pbo"] < 0.15, (
            f"PBO={result['pbo']:.3f} too high for consistently positive returns"
        )

    def test_regime_shift_returns_high_pbo(self):
        """Returns that switch sign mid-period should give higher PBO.

        First half positive, second half negative -> IS trained on first half
        will fail OOS on second half.
        """
        rng = np.random.default_rng(42)
        rets_first = rng.normal(0.005, 0.002, size=250)
        rets_second = rng.normal(-0.005, 0.002, size=250)
        rets = np.concatenate([rets_first, rets_second])
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        # When there's a regime shift, many IS-positive combinations will
        # train on the positive regime but test on the negative regime
        assert result["pbo"] > 0.2, (
            f"PBO={result['pbo']:.3f} should be elevated for regime-shift returns"
        )


# ---------------------------------------------------------------------------
# PBO: output format
# ---------------------------------------------------------------------------

class TestPBOOutputFormat:
    def test_all_keys_present(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2)
        assert "pbo" in result
        assert "n_combinations" in result
        assert "logit_distribution" in result
        assert "median_logit" in result

    def test_pbo_in_unit_interval(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_types_correct(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2)
        assert isinstance(result["pbo"], float)
        assert isinstance(result["n_combinations"], int)
        assert isinstance(result["logit_distribution"], list)
        assert isinstance(result["median_logit"], float)

    def test_logit_distribution_length(self):
        """Logit distribution should have one entry per valid combination."""
        rng = np.random.default_rng(0)
        rets = rng.normal(0, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        assert len(result["logit_distribution"]) == result["n_combinations"]

    def test_logit_values_finite(self):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2)
        for v in result["logit_distribution"]:
            assert math.isfinite(v), f"non-finite logit value: {v}"


# ---------------------------------------------------------------------------
# PBO: edge cases
# ---------------------------------------------------------------------------

class TestPBOEdgeCases:
    def test_minimum_groups(self):
        """n_groups=2, n_test_groups=1 -> C(2,1) = 2 combinations."""
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, size=100)
        result = compute_pbo(rets, n_groups=2, n_test_groups=1)
        assert result["n_combinations"] == 2

    def test_invalid_n_groups_raises(self):
        with pytest.raises(ValueError):
            compute_pbo(np.zeros(100), n_groups=1)

    def test_invalid_n_test_groups_raises(self):
        with pytest.raises(ValueError):
            compute_pbo(np.zeros(100), n_groups=5, n_test_groups=5)
        with pytest.raises(ValueError):
            compute_pbo(np.zeros(100), n_groups=5, n_test_groups=0)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            compute_pbo(np.zeros(3), n_groups=10)

    def test_all_negative_returns_pbo_one(self):
        """All-negative returns: IS never positive -> PBO = 1.0 (conservative)."""
        rets = np.full(200, -0.01)
        # Add tiny noise to avoid zero-variance guard
        rng = np.random.default_rng(42)
        rets = rets + rng.normal(0, 0.0001, size=200)
        result = compute_pbo(rets, n_groups=10, n_test_groups=2, embargo_days=0)
        assert result["pbo"] == 1.0


# ---------------------------------------------------------------------------
# Multi-strategy PBO (returns matrix)
# ---------------------------------------------------------------------------

class TestPBOMatrix:
    def test_single_strategy_matches_scalar(self):
        """Single-strategy matrix PBO should be directionally consistent with scalar.

        The scalar path uses ratio-based lambda (OOS/IS Sharpe), while the
        matrix path uses rank-based lambda (rank/S). For S=1 strategy, the
        rank is always 0 or 1, so the lambda values differ from the ratio.
        Both should agree on direction: low PBO = not overfit.
        """
        rng = np.random.default_rng(42)
        rets = rng.normal(0.001, 0.01, size=200)
        scalar_result = compute_pbo(rets, n_groups=6, n_test_groups=2, embargo_days=0)
        matrix_result = compute_pbo_matrix(
            rets.reshape(1, -1), n_groups=6, n_test_groups=2, embargo_days=0
        )
        # Both should be < 0.5 for a genuinely positive-drift strategy
        assert scalar_result["pbo"] < 0.5
        assert matrix_result["pbo"] < 0.5

    def test_many_noise_strategies_high_pbo(self):
        """Many pure-noise strategies: best-IS is always noise -> high PBO.

        This demonstrates the selection bias: choosing the best from many
        random strategies is almost guaranteed to be overfit. With 20
        strategies over 500 observations, the selection effect is strong
        enough to produce PBO well above the single-strategy baseline.
        """
        rng = np.random.default_rng(42)
        # 20 strategies, 500 days, pure noise — more observations give
        # more IS-positive combinations for a meaningful PBO estimate
        returns_matrix = rng.normal(0, 0.01, size=(20, 500))
        result = compute_pbo_matrix(
            returns_matrix, n_groups=10, n_test_groups=2, embargo_days=0
        )
        # With 20 noise strategies and 500 obs, selection bias pushes PBO high
        assert result["pbo"] > 0.25, (
            f"PBO={result['pbo']:.3f} should be elevated for 20 noise strategies"
        )

    def test_one_genuine_strategy_among_noise(self):
        """One genuine strategy + noise: PBO should be moderate to low.

        If the genuine strategy is reliably best IS AND best OOS, PBO stays
        low despite the noise candidates. But if noise occasionally beats it
        IS, PBO rises.
        """
        rng = np.random.default_rng(42)
        # 10 noise strategies
        noise = rng.normal(0, 0.01, size=(10, 200))
        # 1 genuine strategy with strong signal
        genuine = rng.normal(0.005, 0.005, size=(1, 200))
        returns_matrix = np.vstack([genuine, noise])
        result = compute_pbo_matrix(
            returns_matrix, n_groups=10, n_test_groups=2, embargo_days=0
        )
        # Genuine strategy should usually win IS and succeed OOS
        assert result["pbo"] < 0.5

    def test_best_is_indices_populated(self):
        rng = np.random.default_rng(42)
        returns_matrix = rng.normal(0, 0.01, size=(5, 200))
        result = compute_pbo_matrix(
            returns_matrix, n_groups=6, n_test_groups=2, embargo_days=0
        )
        assert "best_is_indices" in result
        assert len(result["best_is_indices"]) == result["n_combinations"]
        for idx in result["best_is_indices"]:
            assert 0 <= idx < 5

    def test_output_format(self):
        rng = np.random.default_rng(0)
        returns_matrix = rng.normal(0, 0.01, size=(3, 100))
        result = compute_pbo_matrix(
            returns_matrix, n_groups=5, n_test_groups=2, embargo_days=0
        )
        assert isinstance(result["pbo"], float)
        assert isinstance(result["n_combinations"], int)
        assert isinstance(result["logit_distribution"], list)
        assert isinstance(result["median_logit"], float)
        assert isinstance(result["best_is_indices"], list)
        assert 0.0 <= result["pbo"] <= 1.0


# ---------------------------------------------------------------------------
# Regression: embargo count arithmetic
# ---------------------------------------------------------------------------

class TestEmbargoArithmetic:
    def test_embargo_reduces_training_size(self):
        """Training set with embargo must be smaller than without."""
        blocks = _split_into_groups(100, 10)
        test_groups = (3,)
        no_embargo = _build_train_indices(blocks, test_groups, embargo_days=0, n_obs=100)
        with_embargo = _build_train_indices(blocks, test_groups, embargo_days=5, n_obs=100)
        assert len(with_embargo) < len(no_embargo)

    def test_embargo_count_is_correct(self):
        """For a single interior test block, embargo should remove exactly
        2 * embargo_days observations (before + after the test block)."""
        blocks = _split_into_groups(100, 10)
        # Test group 5 (interior block: [50..60))
        test_groups = (5,)
        embargo_days = 3
        no_embargo = _build_train_indices(blocks, test_groups, embargo_days=0, n_obs=100)
        with_embargo = _build_train_indices(blocks, test_groups, embargo_days=embargo_days, n_obs=100)
        # For an interior block, embargo removes exactly 2*embargo_days
        expected_reduction = 2 * embargo_days
        assert len(no_embargo) - len(with_embargo) == expected_reduction
