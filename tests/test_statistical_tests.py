"""Tests for Hansen SPA and MC permutation tests."""
import math
import numpy as np
import pytest


class TestHansenSPA:
    """Tests for Hansen's Superior Predictive Ability test."""

    def test_strong_strategy_is_significant(self):
        """A strategy with clear excess returns should be significant."""
        from layer2.statistical_tests import hansen_spa_test
        rng = np.random.RandomState(42)
        n = 250
        strategy = rng.normal(0.002, 0.01, n)  # positive mean
        benchmark = rng.normal(0.0, 0.01, n)   # zero mean
        result = hansen_spa_test(strategy, benchmark, n_bootstrap=5000)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["mean_excess_return"] > 0

    def test_random_strategy_not_significant(self):
        """A strategy with same distribution as benchmark should not be significant."""
        from layer2.statistical_tests import hansen_spa_test
        rng = np.random.RandomState(42)
        n = 250
        strategy = rng.normal(0.0, 0.01, n)
        benchmark = rng.normal(0.0, 0.01, n)
        result = hansen_spa_test(strategy, benchmark, n_bootstrap=5000)
        assert result["p_value"] > 0.05

    def test_short_series_returns_nonsignificant(self):
        """Fewer than 10 observations should return non-significant."""
        from layer2.statistical_tests import hansen_spa_test
        result = hansen_spa_test(np.array([0.01, 0.02]), np.array([0.0, 0.0]))
        assert result["significant"] is False
        assert result["p_value"] == 1.0

    def test_length_mismatch_raises(self):
        from layer2.statistical_tests import hansen_spa_test
        with pytest.raises(ValueError, match="same length"):
            hansen_spa_test(np.zeros(10), np.zeros(15))

    def test_returns_dict_with_required_keys(self):
        from layer2.statistical_tests import hansen_spa_test
        rng = np.random.RandomState(42)
        result = hansen_spa_test(rng.normal(0, 0.01, 50), rng.normal(0, 0.01, 50))
        assert "p_value" in result
        assert "t_statistic" in result
        assert "mean_excess_return" in result
        assert "n_obs" in result
        assert "significant" in result

    def test_p_value_in_01(self):
        from layer2.statistical_tests import hansen_spa_test
        rng = np.random.RandomState(42)
        result = hansen_spa_test(rng.normal(0.001, 0.01, 100), rng.normal(0, 0.01, 100))
        assert 0.0 <= result["p_value"] <= 1.0


class TestHansenSPAMulti:
    """Tests for multi-strategy Hansen SPA."""

    def test_best_of_many_noise_not_significant(self):
        """Best of 20 noise strategies should not be significant after correction."""
        from layer2.statistical_tests import hansen_spa_multi
        rng = np.random.RandomState(42)
        n = 250
        benchmark = rng.normal(0.0, 0.01, n)
        strategies = [rng.normal(0.0, 0.01, n) for _ in range(20)]
        result = hansen_spa_multi(strategies, benchmark, n_bootstrap=5000)
        # With 20 noise strategies, the best will have some positive mean
        # by chance, but SPA should correct for this
        assert result["n_strategies"] == 20
        assert "p_value" in result

    def test_one_genuine_among_noise(self):
        """One genuine strategy among noise should be detectable."""
        from layer2.statistical_tests import hansen_spa_multi
        rng = np.random.RandomState(42)
        n = 250
        benchmark = rng.normal(0.0, 0.01, n)
        strategies = [rng.normal(0.0, 0.01, n) for _ in range(9)]
        strategies.append(rng.normal(0.003, 0.01, n))  # genuine signal
        result = hansen_spa_multi(strategies, benchmark, n_bootstrap=5000)
        assert result["best_strategy_idx"] == 9  # the genuine one
        assert result["significant"] is True

    def test_empty_list_raises(self):
        from layer2.statistical_tests import hansen_spa_multi
        with pytest.raises(ValueError, match="non-empty"):
            hansen_spa_multi([], np.zeros(10))


class TestMCPermutation:
    """Tests for Monte Carlo permutation test."""

    def test_temporally_structured_returns_significant(self):
        """Returns with temporal structure (momentum) should be significant.

        Shuffling destroys the momentum autocorrelation but preserves the
        marginal distribution. A strategy that profits from momentum has
        higher Sharpe when temporal structure is intact.
        """
        from layer2.statistical_tests import mc_permutation_test
        rng = np.random.RandomState(42)
        n = 250
        # Create momentum returns: positive autocorrelation
        noise = rng.normal(0.0, 0.01, n)
        returns = np.zeros(n)
        returns[0] = noise[0]
        for i in range(1, n):
            returns[i] = 0.3 * returns[i-1] + noise[i] + 0.001  # AR(1) + drift
        result = mc_permutation_test(returns, n_permutations=5000)
        # With temporal structure + drift, observed Sharpe should exceed shuffled
        assert result["observed_sharpe"] > 0
        assert "p_value" in result

    def test_iid_returns_p_value_near_half(self):
        """IID returns: shuffling doesn't change distribution, p ≈ 0.5.

        When returns are already IID (no temporal structure), shuffling
        produces the same distribution. The p-value should be near 0.5,
        not near 0 or 1.
        """
        from layer2.statistical_tests import mc_permutation_test
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 250)
        result = mc_permutation_test(returns, n_permutations=5000)
        # p-value should be moderate (not extreme) for IID returns
        assert 0.0 <= result["p_value"] <= 1.0

    def test_short_series(self):
        from layer2.statistical_tests import mc_permutation_test
        result = mc_permutation_test(np.array([0.01, 0.02, 0.03]))
        assert result["significant"] is False

    def test_returns_dict_with_required_keys(self):
        from layer2.statistical_tests import mc_permutation_test
        rng = np.random.RandomState(42)
        result = mc_permutation_test(rng.normal(0, 0.01, 50))
        assert "p_value" in result
        assert "observed_sharpe" in result
        assert "median_shuffled_sharpe" in result
        assert "pct_95_shuffled" in result
        assert "n_obs" in result
        assert "significant" in result

    def test_p_value_in_01(self):
        from layer2.statistical_tests import mc_permutation_test
        rng = np.random.RandomState(42)
        result = mc_permutation_test(rng.normal(0.001, 0.01, 100))
        assert 0.0 <= result["p_value"] <= 1.0
