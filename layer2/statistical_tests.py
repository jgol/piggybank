"""Statistical significance tests for GP-evolved strategies.

Implements Hansen's Superior Predictive Ability (SPA) test and Monte Carlo
permutation test — the two mandatory tests that complement PBO/CSCV for
distinguishing genuine skill from data-snooping artifacts.

Hansen SPA (Hansen, 2005): Does the best strategy significantly outperform
a naive benchmark? Controls for data-snooping across the full set of tested
strategies. Supersedes White's (2000) Reality Check with better power.

MC Permutation (White, 2000): Is performance due to temporal structure or
random alignment? Shuffles daily returns to destroy temporal dependencies
while preserving marginal distribution.

Both tests operate on daily return arrays and are designed to run post-GP
on completed strategy results — no GP re-evaluation needed.

References:
    Hansen, P. R. (2005). A test for superior predictive ability.
    Journal of Business & Economic Statistics, 23(4), 365-380.

    White, H. (2000). A reality check for data snooping.
    Econometrica, 68(5), 1097-1126.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hansen SPA test
# ---------------------------------------------------------------------------

def hansen_spa_test(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_length: int = 5,
    seed: int = 42,
) -> Dict:
    """Hansen's Superior Predictive Ability test.

    Tests H0: the strategy does NOT outperform the benchmark.
    Rejection (p < 0.05) means the strategy has statistically significant
    superior performance after accounting for data-snooping.

    Uses stationary bootstrap (Politis & Romano, 1994) to preserve
    temporal dependence in the return series. Block length of 5 trading
    days matches the weekly autocorrelation structure typical of daily
    options strategy returns.

    Args:
        strategy_returns: Daily returns of the strategy being tested.
        benchmark_returns: Daily returns of the naive benchmark (e.g.,
            random-entry hold-to-EOD, or buy-and-hold SPX).
        n_bootstrap: Number of bootstrap replications.
        block_length: Expected block length for stationary bootstrap.
            Politis & Romano (1994) recommend sqrt(T) for general use;
            5 is conservative for daily options data.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys:
            p_value: float in [0, 1]. Reject H0 if p < 0.05.
            t_statistic: float. Test statistic (mean excess / SE).
            mean_excess_return: float. Strategy mean - benchmark mean.
            n_obs: int. Number of daily observations.
            significant: bool. True if p < 0.05.
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError(
            f"strategy_returns ({len(strategy_returns)}) and "
            f"benchmark_returns ({len(benchmark_returns)}) must have same length"
        )
    n = len(strategy_returns)
    if n < 10:
        return {
            "p_value": 1.0, "t_statistic": 0.0,
            "mean_excess_return": 0.0, "n_obs": n,
            "significant": False,
        }

    # Excess returns: strategy - benchmark
    excess = strategy_returns - benchmark_returns
    mean_excess = float(np.mean(excess))
    se_excess = float(np.std(excess, ddof=1) / math.sqrt(n))
    if se_excess < 1e-12:
        t_stat = 0.0
    else:
        t_stat = mean_excess / se_excess

    # Stationary bootstrap (Politis & Romano, 1994)
    # Each bootstrap sample draws blocks of geometrically distributed
    # length (expected = block_length), preserving serial dependence.
    rng = np.random.RandomState(seed)
    p_geom = 1.0 / max(block_length, 1)
    boot_t_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Build bootstrap sample via stationary bootstrap
        boot_excess = np.empty(n)
        pos = 0
        idx = rng.randint(0, n)
        while pos < n:
            boot_excess[pos] = excess[idx]
            pos += 1
            # Continue current block with probability (1-p), start new with p
            if rng.random() < p_geom:
                idx = rng.randint(0, n)
            else:
                idx = (idx + 1) % n

        # Center the bootstrap (under H0, mean excess = 0)
        boot_mean = np.mean(boot_excess) - mean_excess
        boot_se = np.std(boot_excess, ddof=1) / math.sqrt(n)
        if boot_se > 1e-12:
            boot_t_stats[b] = boot_mean / boot_se
        else:
            boot_t_stats[b] = 0.0

    # p-value: fraction of bootstrap t-stats exceeding observed t-stat
    # (one-sided: we test if strategy is BETTER, not just different)
    p_value = float(np.mean(boot_t_stats >= t_stat))

    return {
        "p_value": round(p_value, 6),
        "t_statistic": round(t_stat, 6),
        "mean_excess_return": round(mean_excess, 8),
        "n_obs": n,
        "significant": p_value < 0.05,
    }


def hansen_spa_multi(
    strategy_returns_list: List[np.ndarray],
    benchmark_returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_length: int = 5,
    seed: int = 42,
) -> Dict:
    """Multi-strategy Hansen SPA: tests if the BEST strategy outperforms
    the benchmark after correcting for testing multiple strategies.

    This is the proper formulation for GP Pareto front evaluation:
    we tested N strategies and selected the best — does the best one
    genuinely beat the benchmark, or did we just get lucky from N tries?

    Args:
        strategy_returns_list: List of daily return arrays, one per strategy.
        benchmark_returns: Daily returns of the naive benchmark.
        n_bootstrap: Number of bootstrap replications.
        block_length: Expected block length for stationary bootstrap.
        seed: RNG seed.

    Returns:
        dict with keys:
            p_value: float. Corrected p-value for the best strategy.
            best_strategy_idx: int. Index of the best strategy.
            best_mean_excess: float. Best strategy's mean excess return.
            n_strategies: int.
            significant: bool.
    """
    if not strategy_returns_list:
        raise ValueError("strategy_returns_list must be non-empty")

    n = len(benchmark_returns)
    n_strategies = len(strategy_returns_list)

    # Compute excess returns for each strategy
    excess_list = []
    mean_excess_list = []
    for sr in strategy_returns_list:
        if len(sr) != n:
            raise ValueError(f"All strategy returns must have length {n}")
        excess = sr - benchmark_returns
        excess_list.append(excess)
        mean_excess_list.append(float(np.mean(excess)))

    # Best strategy by mean excess return
    best_idx = int(np.argmax(mean_excess_list))
    best_mean = mean_excess_list[best_idx]

    # Observed test statistic: max over all strategies of (mean_excess / SE)
    t_stats = np.zeros(n_strategies)
    for i, excess in enumerate(excess_list):
        se = float(np.std(excess, ddof=1) / math.sqrt(n))
        t_stats[i] = mean_excess_list[i] / se if se > 1e-12 else 0.0
    observed_max_t = float(np.max(t_stats))

    # Bootstrap: compute max t-stat under H0 for each replication
    rng = np.random.RandomState(seed)
    p_geom = 1.0 / max(block_length, 1)
    boot_max_t = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Same bootstrap indices for all strategies (preserves cross-correlation)
        boot_indices = np.empty(n, dtype=int)
        pos = 0
        idx = rng.randint(0, n)
        while pos < n:
            boot_indices[pos] = idx
            pos += 1
            if rng.random() < p_geom:
                idx = rng.randint(0, n)
            else:
                idx = (idx + 1) % n

        # Compute centered t-stats for each strategy
        max_t = -np.inf
        for i, excess in enumerate(excess_list):
            boot_excess = excess[boot_indices]
            boot_mean = np.mean(boot_excess) - mean_excess_list[i]  # center under H0
            boot_se = np.std(boot_excess, ddof=1) / math.sqrt(n)
            if boot_se > 1e-12:
                t = boot_mean / boot_se
            else:
                t = 0.0
            if t > max_t:
                max_t = t
        boot_max_t[b] = max_t

    # p-value: fraction of bootstrap max-t exceeding observed max-t
    p_value = float(np.mean(boot_max_t >= observed_max_t))

    return {
        "p_value": round(p_value, 6),
        "best_strategy_idx": best_idx,
        "best_mean_excess": round(best_mean, 8),
        "best_t_statistic": round(observed_max_t, 6),
        "n_strategies": n_strategies,
        "n_obs": n,
        "significant": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Monte Carlo permutation test
# ---------------------------------------------------------------------------

def mc_permutation_test(
    daily_returns: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """Monte Carlo permutation test for strategy significance.

    Tests H0: the strategy's Sharpe ratio is not due to temporal structure.
    Shuffles daily returns to destroy temporal dependencies while preserving
    the marginal distribution. If the original Sharpe exceeds most shuffled
    Sharpes, the strategy exploits genuine temporal patterns.

    Reference: White, H. (2000). A reality check for data snooping.
    Econometrica, 68(5), 1097-1126.

    Args:
        daily_returns: Daily returns of the strategy.
        n_permutations: Number of random shuffles.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys:
            p_value: float in [0, 1]. Reject H0 if p < 0.05.
            observed_sharpe: float. Annualized Sharpe of original returns.
            median_shuffled_sharpe: float. Median Sharpe across permutations.
            pct_95_shuffled: float. 95th percentile of shuffled Sharpes.
            n_obs: int.
            significant: bool.
    """
    n = len(daily_returns)
    if n < 10:
        return {
            "p_value": 1.0, "observed_sharpe": 0.0,
            "median_shuffled_sharpe": 0.0, "pct_95_shuffled": 0.0,
            "n_obs": n, "significant": False,
        }

    ann = math.sqrt(252.0)

    # Observed Sharpe
    std = float(np.std(daily_returns))
    if std < 1e-12:
        observed_sharpe = 0.0
    else:
        observed_sharpe = float(np.mean(daily_returns) / std * ann)

    # Permutation distribution
    rng = np.random.RandomState(seed)
    shuffled_sharpes = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(daily_returns)
        perm_std = float(np.std(perm))
        if perm_std > 1e-12:
            shuffled_sharpes[i] = float(np.mean(perm) / perm_std * ann)
        else:
            shuffled_sharpes[i] = 0.0

    # p-value: fraction of shuffled Sharpes >= observed
    p_value = float(np.mean(shuffled_sharpes >= observed_sharpe))

    return {
        "p_value": round(p_value, 6),
        "observed_sharpe": round(observed_sharpe, 6),
        "median_shuffled_sharpe": round(float(np.median(shuffled_sharpes)), 6),
        "pct_95_shuffled": round(float(np.percentile(shuffled_sharpes, 95)), 6),
        "n_obs": n,
        "significant": p_value < 0.05,
    }
