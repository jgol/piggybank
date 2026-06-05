"""Probability of Backtest Overfitting (PBO) via Combinatorial Purged Cross-Validation.

Implements the methodology from Bailey, Borwein, Lopez de Prado & Zhu (2014),
"The probability of backtest overfitting," Journal of Computational Finance.

PBO quantifies the probability that GP-discovered strategies are overfit to
training data. The core idea: split data into N sequential blocks, enumerate
all C(N, k) train/test partitions, and measure how often the best in-sample
strategy fails out-of-sample.

Two entry points:
  - compute_pbo(): single-strategy PBO from a returns array
  - compute_pbo_from_pareto_front(): multi-strategy PBO that accounts for
    selection bias from GP search (the proper Bailey et al. formulation)

Both use embargo/purging to prevent information leakage at train/test
boundaries.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Sharpe helper (matches evaluator.py convention: annualized, sqrt(252))
# ---------------------------------------------------------------------------

def _daily_sharpe(daily_returns: np.ndarray) -> float:
    """Annualized Sharpe from daily returns. sqrt(252) convention.

    Uses population std (ddof=0) to match evaluator_vectorized.py. Standard
    finance uses ddof=1 (sample std); the difference is ~2.5% for n=20.
    Consistency with the proxy evaluator is prioritized over textbook convention.

    Returns 0.0 for empty/zero-variance input (matches evaluator.py guard).
    """
    if len(daily_returns) == 0 or np.std(daily_returns) < 1e-10:
        return 0.0
    return float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

def _split_into_groups(n_obs: int, n_groups: int) -> List[Tuple[int, int]]:
    """Split n_obs observations into n_groups sequential blocks.

    Returns list of (start, end) index pairs. The last group absorbs any
    remainder so no observations are silently dropped.

    Raises ValueError if n_obs < n_groups (cannot form non-empty blocks).
    """
    if n_obs < n_groups:
        raise ValueError(
            f"Cannot split {n_obs} observations into {n_groups} groups "
            f"(need at least 1 observation per group)."
        )
    base_size = n_obs // n_groups
    blocks: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_groups):
        if i < n_groups - 1:
            end = start + base_size
        else:
            end = n_obs  # last block absorbs remainder
        blocks.append((start, end))
        start = end
    return blocks


# ---------------------------------------------------------------------------
# Embargo / purging
# ---------------------------------------------------------------------------

def _build_train_indices(
    blocks: List[Tuple[int, int]],
    test_group_indices: Tuple[int, ...],
    embargo_days: int,
    n_obs: int,
) -> np.ndarray:
    """Build training indices with purging around test block boundaries.

    For each test block, removes `embargo_days` observations from adjacent
    training blocks on both sides of the boundary:
      - Before a test block: remove the last `embargo_days` of the preceding
        training block.
      - After a test block: remove the first `embargo_days` of the following
        training block.

    Returns a sorted 1-D array of valid training observation indices.
    """
    test_set = set(test_group_indices)
    train_groups = [i for i in range(len(blocks)) if i not in test_set]

    # Collect all test boundaries (start, end indices) for purging
    test_boundaries: List[Tuple[int, int]] = []
    for gi in test_group_indices:
        test_boundaries.append(blocks[gi])

    # Build exclusion set: embargo zones around test boundaries
    excluded = set()
    for t_start, t_end in test_boundaries:
        # Purge before test block: indices [t_start - embargo_days, t_start)
        for idx in range(max(0, t_start - embargo_days), t_start):
            excluded.add(idx)
        # Purge after test block: indices [t_end, t_end + embargo_days)
        for idx in range(t_end, min(n_obs, t_end + embargo_days)):
            excluded.add(idx)

    # Collect train indices minus exclusions
    train_indices: List[int] = []
    for gi in train_groups:
        start, end = blocks[gi]
        for idx in range(start, end):
            if idx not in excluded:
                train_indices.append(idx)

    return np.array(sorted(train_indices), dtype=np.intp)


def _build_test_indices(
    blocks: List[Tuple[int, int]],
    test_group_indices: Tuple[int, ...],
) -> np.ndarray:
    """Build test indices from selected block groups."""
    indices: List[int] = []
    for gi in test_group_indices:
        start, end = blocks[gi]
        indices.extend(range(start, end))
    return np.array(sorted(indices), dtype=np.intp)


# ---------------------------------------------------------------------------
# Single-strategy PBO
# ---------------------------------------------------------------------------

def compute_pbo(
    strategy_returns: np.ndarray,
    n_groups: int = 10,
    n_test_groups: int = 2,
    embargo_days: int = 5,
) -> dict:
    """Compute PBO via Combinatorial Purged Cross-Validation (single strategy).

    For a single strategy, PBO reduces to: fraction of CPCV combinations
    where the in-sample Sharpe is positive but the out-of-sample Sharpe is
    non-positive. This captures the overfitting scenario where the strategy
    *looks good* in-sample but fails out-of-sample.

    Combinations where IS Sharpe <= 0 are excluded from the denominator
    (the strategy never looked good in-sample, so there is nothing to
    overfit).

    The logit distribution is computed as logit(lambda) where lambda is the
    normalized rank of the OOS performance relative to the IS performance.
    For the single-strategy case, lambda = 1 if OOS >= IS, else 0, but we
    use the continuous formulation: lambda = OOS_sharpe / IS_sharpe when
    both are positive, clipped to [0, 1], then logit-transformed.

    Args:
        strategy_returns: (n_days,) array of daily returns.
        n_groups: number of sequential blocks to divide the data into.
        n_test_groups: number of blocks per test partition.
        embargo_days: observations to remove from training blocks adjacent
            to test block boundaries (purging / embargo).

    Returns:
        dict with keys:
            pbo: float in [0, 1], probability of backtest overfitting
            n_combinations: int, total CPCV combinations evaluated
            logit_distribution: list of float, logit(lambda) per combination
            median_logit: float, median of the logit distribution
    """
    strategy_returns = np.asarray(strategy_returns, dtype=np.float64)
    n_obs = len(strategy_returns)

    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError(
            f"n_test_groups must be in [1, n_groups-1], got {n_test_groups} "
            f"with n_groups={n_groups}"
        )

    blocks = _split_into_groups(n_obs, n_groups)
    combinations = list(itertools.combinations(range(n_groups), n_test_groups))
    n_combinations = len(combinations)

    logit_dist: List[float] = []
    n_is_positive = 0

    for test_indices_tuple in combinations:
        train_idx = _build_train_indices(blocks, test_indices_tuple, embargo_days, n_obs)
        test_idx = _build_test_indices(blocks, test_indices_tuple)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        is_sharpe = _daily_sharpe(strategy_returns[train_idx])
        oos_sharpe = _daily_sharpe(strategy_returns[test_idx])

        # Only count combinations where IS performance looks good
        if is_sharpe > 0:
            n_is_positive += 1

        # Logit(lambda) computation:
        # lambda = relative rank of OOS vs IS. For single strategy:
        # lambda = 0.5 if OOS_sharpe >= 0 (strategy "works"), else 0
        # We use the continuous formulation for a richer distribution.
        if is_sharpe > 1e-10:
            lam = max(0.0, min(1.0, oos_sharpe / is_sharpe))
        elif is_sharpe < -1e-10:
            # IS negative: strategy never looked good. lambda=1 (no overfit
            # evidence, but also no selection to begin with).
            lam = 1.0
        else:
            # IS ~= 0: indeterminate. Use 0.5.
            lam = 0.5

        # Logit transform with clipping to avoid inf
        lam_clipped = max(1e-6, min(1.0 - 1e-6, lam))
        logit_val = math.log(lam_clipped / (1.0 - lam_clipped))
        logit_dist.append(logit_val)

    # PBO = P(lambda < 0.5 | IS > 0) per Bailey et al. (2014).
    # logit(lambda) < 0 ↔ lambda < 0.5.
    # For scalar PBO, IS-negative combos have lambda=1.0 (logit >> 0),
    # so they never count as lambda < 0.5. Safe to count over all logits.
    if n_is_positive > 0:
        n_lambda_below_half = sum(1 for lv in logit_dist if lv < 0.0)
        pbo = n_lambda_below_half / n_is_positive
    else:
        pbo = 1.0  # no IS-positive combos → conservative bound

    median_logit = float(np.median(logit_dist)) if logit_dist else 0.0

    return {
        "pbo": float(pbo),
        "n_combinations": n_combinations,
        "logit_distribution": logit_dist,
        "median_logit": median_logit,
    }


# ---------------------------------------------------------------------------
# Multi-strategy PBO (Pareto front)
# ---------------------------------------------------------------------------

def compute_pbo_matrix(
    returns_matrix: np.ndarray,
    n_groups: int = 10,
    n_test_groups: int = 2,
    embargo_days: int = 5,
) -> dict:
    """Compute PBO from a matrix of strategy returns (Bailey et al. proper formulation).

    This is the core multi-strategy PBO algorithm. Given S strategies each with
    T daily returns, it:
      1. Splits T into N sequential blocks
      2. For each of C(N, k) CPCV combinations:
         a. Computes IS Sharpe for all S strategies on training blocks
         b. Picks the best IS strategy (the one GP selection would choose)
         c. Checks if that strategy's OOS Sharpe is positive
      3. PBO = fraction of combinations where the best-IS strategy fails OOS

    This accounts for selection bias: even if individual strategies aren't
    overfit, choosing the best one from many candidates introduces overfitting.

    Args:
        returns_matrix: (S, T) array where S = number of strategies,
            T = number of daily returns. All strategies must span the same
            time period.
        n_groups: number of sequential blocks.
        n_test_groups: blocks per test partition.
        embargo_days: purging gap at train/test boundaries.

    Returns:
        dict with keys:
            pbo: float in [0, 1]
            n_combinations: int
            logit_distribution: list of float
            median_logit: float
            best_is_indices: list of int, which strategy was best-IS per combo
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(1, -1)

    n_strategies, n_obs = returns_matrix.shape

    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}")
    if n_test_groups < 1 or n_test_groups >= n_groups:
        raise ValueError(
            f"n_test_groups must be in [1, n_groups-1], got {n_test_groups}"
        )

    blocks = _split_into_groups(n_obs, n_groups)
    combinations = list(itertools.combinations(range(n_groups), n_test_groups))
    n_combinations = len(combinations)

    logit_dist: List[float] = []
    best_is_indices: List[int] = []
    # Track per-combination IS/OOS Sharpe of the selected (best-IS) strategy
    # to avoid recomputing them in the final PBO tally.
    combo_best_is_sharpe: List[float] = []
    combo_best_oos_sharpe: List[float] = []

    for test_indices_tuple in combinations:
        train_idx = _build_train_indices(blocks, test_indices_tuple, embargo_days, n_obs)
        test_idx = _build_test_indices(blocks, test_indices_tuple)

        if len(train_idx) == 0 or len(test_idx) == 0:
            best_is_indices.append(-1)
            combo_best_is_sharpe.append(0.0)
            combo_best_oos_sharpe.append(0.0)
            continue

        # Compute IS Sharpe for all strategies
        is_sharpes = np.array([
            _daily_sharpe(returns_matrix[s, train_idx])
            for s in range(n_strategies)
        ])

        # Pick best IS strategy
        best_is = int(np.argmax(is_sharpes))
        best_is_indices.append(best_is)
        best_is_sharpe = is_sharpes[best_is]

        # OOS Sharpe of the selected strategy
        oos_sharpe = _daily_sharpe(returns_matrix[best_is, test_idx])

        combo_best_is_sharpe.append(best_is_sharpe)
        combo_best_oos_sharpe.append(oos_sharpe)

        # Logit(lambda): rank of OOS performance of best-IS strategy
        # among all strategies' OOS performances
        oos_sharpes = np.array([
            _daily_sharpe(returns_matrix[s, test_idx])
            for s in range(n_strategies)
        ])

        if n_strategies > 1:
            # Lambda per Bailey et al. (2014): rank of best-IS strategy's OOS
            # performance among all strategies. Lambda near 0 = best IS is worst
            # OOS (overfit). Lambda near 1 = best IS also best OOS (robust).
            # PBO = P(lambda < 0.5).
            lam = np.sum(oos_sharpes <= oos_sharpe) / n_strategies
        else:
            # Single strategy: lambda = 1 if OOS positive, 0 if negative
            if best_is_sharpe > 1e-10:
                lam = 1.0 if oos_sharpe > 0 else 0.0
            else:
                lam = 0.5

        lam_clipped = max(1e-6, min(1.0 - 1e-6, lam))
        logit_val = math.log(lam_clipped / (1.0 - lam_clipped))
        logit_dist.append(logit_val)

    # PBO = P(lambda < 0.5 | IS > 0) per Bailey et al. (2014).
    # For rank-based lambda (multi-strategy), IS-negative combos can have
    # lambda < 0.5, so we must filter to IS-positive combos only.
    n_is_positive = sum(
        1 for ci in range(n_combinations)
        if best_is_indices[ci] >= 0 and combo_best_is_sharpe[ci] > 0
    )
    if n_is_positive > 0:
        n_lambda_below_half = sum(
            1 for ci in range(len(logit_dist))
            if ci < n_combinations
            and best_is_indices[ci] >= 0
            and combo_best_is_sharpe[ci] > 0
            and logit_dist[ci] < 0.0
        )
        pbo = n_lambda_below_half / n_is_positive
    else:
        pbo = 1.0

    median_logit = float(np.median(logit_dist)) if logit_dist else 0.0

    return {
        "pbo": float(pbo),
        "n_combinations": n_combinations,
        "logit_distribution": logit_dist,
        "median_logit": median_logit,
        "best_is_indices": best_is_indices,
    }


def compute_pbo_from_pareto_front(
    pareto_strategies: list,
    data: "pd.DataFrame",
    template: "Template",
    n_groups: int = 10,
    n_test_groups: int = 2,
    embargo_days: int = 5,
    fold_recenter_stats: "Optional[Dict]" = None,
) -> dict:
    """Compute multi-strategy PBO by re-evaluating Pareto front strategies via CPCV.

    This is the full Bailey et al. formulation adapted for GP-evolved option
    strategies. For each CPCV train/test combination, ALL Pareto front
    strategies are re-evaluated on both splits using the vectorized backtester.
    The best in-sample strategy is identified, and its out-of-sample
    performance determines whether that combination counts as overfit.

    This properly accounts for the selection bias introduced by GP search:
    even if individual strategies aren't overfit, choosing the "winner" from
    a population of evolved candidates introduces overfitting proportional
    to the number of candidates and the degree of IS optimization.

    Args:
        pareto_strategies: list of dicts, each with keys "entry_tree",
            "exit_tree", "size_tree" (Node objects from the GP grammar).
        data: minute-resolution DataFrame (same format as evaluator_vectorized
            expects). Must have a "date" column for block splitting.
        template: Template object for backtesting.
        n_groups: number of sequential date blocks.
        n_test_groups: blocks per test partition.
        embargo_days: trading days to purge at train/test boundaries.

    Returns:
        dict with keys:
            pbo: float in [0, 1]
            n_combinations: int
            logit_distribution: list of float
            median_logit: float
            n_strategies: int
            best_is_indices: list of int
    """
    import pandas as pd
    from layer2.evaluator_vectorized import vectorized_backtest

    if not pareto_strategies:
        raise ValueError("pareto_strategies must be non-empty")
    if "date" not in data.columns:
        raise ValueError("data must have a 'date' column for temporal splitting")

    # Get unique dates for block splitting
    dates = sorted(data["date"].unique())
    n_dates = len(dates)

    if n_dates < n_groups:
        raise ValueError(
            f"Only {n_dates} unique dates but n_groups={n_groups}. "
            f"Reduce n_groups or provide more data."
        )

    # Split dates into groups
    date_blocks = _split_into_groups(n_dates, n_groups)
    combinations = list(itertools.combinations(range(n_groups), n_test_groups))
    n_combinations = len(combinations)
    n_strategies = len(pareto_strategies)

    logit_dist: List[float] = []
    best_is_indices: List[int] = []
    combo_best_is_sharpe: List[float] = []
    n_is_positive = 0
    for test_indices_tuple in combinations:
        # Build train/test date sets with embargo
        train_date_idx = _build_train_indices(
            date_blocks, test_indices_tuple, embargo_days, n_dates
        )
        test_date_idx = _build_test_indices(date_blocks, test_indices_tuple)

        if len(train_date_idx) == 0 or len(test_date_idx) == 0:
            best_is_indices.append(-1)
            continue

        train_dates = set(dates[i] for i in train_date_idx)
        test_dates = set(dates[i] for i in test_date_idx)

        train_data = data[data["date"].isin(train_dates)].reset_index(drop=True)
        test_data = data[data["date"].isin(test_dates)].reset_index(drop=True)

        if len(train_data) == 0 or len(test_data) == 0:
            best_is_indices.append(-1)
            continue

        # Resolve fold_ids once per CSCV combination (not per strategy)
        _train_fid, _test_fid = None, None
        if fold_recenter_stats is not None:
            try:
                from layer2.evaluator import (
                    _resolve_fold_ids_for_data, _dominant_fold_id,
                )
                _train_fid = _dominant_fold_id(
                    _resolve_fold_ids_for_data(train_data))
                _test_fid = _dominant_fold_id(
                    _resolve_fold_ids_for_data(test_data))
            except Exception as _exc:
                import sys
                print(f"[PBO] fold_id resolution failed: {_exc}",
                      file=sys.stderr)

        # Evaluate all strategies on train and test
        is_sharpes = np.zeros(n_strategies)
        oos_sharpes = np.zeros(n_strategies)

        for s_idx, strat in enumerate(pareto_strategies):
            # Auto-deserialize s-expression strings (JSONL output format)
            from layer2.grammar import from_sexpr as _from_sexpr
            _entry = strat["entry_tree"]
            _exit = strat["exit_tree"]
            _size = strat.get("size_tree")
            _delta = strat.get("delta_tree")
            if isinstance(_entry, str):
                _entry = _from_sexpr(_entry)
            if isinstance(_exit, str):
                _exit = _from_sexpr(_exit)
            if isinstance(_size, str):
                _size = _from_sexpr(_size)
            if isinstance(_delta, str):
                _delta = _from_sexpr(_delta)
            # PBO must re-score with the strategy's evolved stop (else IS/OOS
            # Sharpe diverges from the fitness it was selected on). Absent ⇒
            # backtester default.
            _stop_kw = ({"stop_loss_credit_multiple": float(strat["stop_mult"])}
                        if strat.get("stop_mult") is not None else {})
            # Train evaluation
            try:
                train_result = vectorized_backtest(
                    entry_tree=_entry,
                    exit_tree=_exit,
                    size_tree=_size,
                    template=template,
                    data=train_data,
                    delta_tree=_delta,
                    fold_recenter_stats=fold_recenter_stats,
                    fold_id=_train_fid,
                    **_stop_kw,
                )
                is_sharpes[s_idx] = train_result.sharpe
            except Exception:
                is_sharpes[s_idx] = -999.0

            # Test evaluation
            try:
                test_result = vectorized_backtest(
                    entry_tree=_entry,
                    exit_tree=_exit,
                    size_tree=_size,
                    template=template,
                    data=test_data,
                    delta_tree=_delta,
                    fold_recenter_stats=fold_recenter_stats,
                    fold_id=_test_fid,
                    **_stop_kw,
                )
                oos_sharpes[s_idx] = test_result.sharpe
            except Exception:
                oos_sharpes[s_idx] = -999.0

        # Pick best IS strategy
        best_is = int(np.argmax(is_sharpes))
        best_is_indices.append(best_is)
        best_is_sharpe = is_sharpes[best_is]
        combo_best_is_sharpe.append(best_is_sharpe)
        oos_sharpe = oos_sharpes[best_is]

        if best_is_sharpe > 0:
            n_is_positive += 1

        # Logit(lambda): rank of best-IS strategy's OOS among all OOS
        if n_strategies > 1:
            # Lambda per Bailey et al.: rank of best-IS OOS among all OOS.
            # High = robust, low = overfit. Matches compute_pbo_matrix convention.
            lam = np.sum(oos_sharpes <= oos_sharpe) / n_strategies
        else:
            if best_is_sharpe > 1e-10:
                lam = 1.0 if oos_sharpe > 0 else 0.0
            else:
                lam = 0.5

        lam_clipped = max(1e-6, min(1.0 - 1e-6, lam))
        logit_val = math.log(lam_clipped / (1.0 - lam_clipped))
        logit_dist.append(logit_val)

    # PBO = P(lambda < 0.5 | IS > 0) per Bailey et al. (2014).
    # Filter to IS-positive combos only (rank-based lambda can be < 0.5
    # even for IS-negative combos, which would inflate PBO > 1.0).
    if n_is_positive > 0:
        n_lambda_below_half = sum(
            1 for ci in range(len(logit_dist))
            if ci < len(combo_best_is_sharpe)
            and combo_best_is_sharpe[ci] > 0
            and logit_dist[ci] < 0.0
        )
        pbo = n_lambda_below_half / n_is_positive
    else:
        pbo = 1.0

    median_logit = float(np.median(logit_dist)) if logit_dist else 0.0

    return {
        "pbo": float(pbo),
        "n_combinations": n_combinations,
        "logit_distribution": logit_dist,
        "median_logit": median_logit,
        "n_strategies": n_strategies,
        "best_is_indices": best_is_indices,
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (DSR) — multiple-testing-significance gate (P1-B)
# ---------------------------------------------------------------------------
#
# Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
# Selection Bias, Backtest Overfitting, and Non-Normality," Journal of
# Portfolio Management 40(5), 94-107.
#
# The GP keeps the best-Sharpe individuals out of ~pop x gen trials, so the
# max observed Sharpe on the Pareto front is inflated by multiple testing.
# DSR deflates the significance threshold by the EXPECTED MAXIMUM Sharpe of
# N independent trials (SR*_N): a champion only "passes" if its Sharpe is
# significant GIVEN how many distinct strategies competed.
#
# This is a GATE at the Pareto-front level. It does NOT touch the NSGA-III
# objective vector (DEFAULT_OBJECTIVES) or the per-individual fitness; the
# Sharpe-based selection inside evolution is unchanged.
#
# ---------------------------------------------------------------------------
# UNITS (critical):
# ---------------------------------------------------------------------------
# BacktestResult.sharpe is ANNUALIZED (x sqrt(252), see _daily_sharpe above
# and evaluator_vectorized.py). Bailey & Lopez de Prado's SR* and the DSR
# z-statistic are defined in PER-PERIOD (here: DAILY) Sharpe units, because
# the variance estimate V (the dispersion of trial Sharpes) and the
# sqrt(T-1) sample-size scaling are both per-period quantities.
#
# Therefore EVERYTHING in this section works in DAILY Sharpe units:
# SR_daily = sharpe_annualized / sqrt(252)
# applied CONSISTENTLY to (a) each front member's SR and (b) the cross-front
# variance V = Var(daily Sharpes). Mixing an annualized member SR with a
# daily-scale SR* (a sqrt(252) ~ 15.87x inflation of the member) would make
# essentially any champion clear the bar -- the exact failure mode this gate
# exists to prevent. Callers pass ANNUALIZED Sharpes; de-annualization happens
# inside these functions so the call site cannot get the units wrong.

# Trading days per year — annualization factor used everywhere in layer2.
TRADING_DAYS_PER_YEAR = 252.0

# HIGH H1 (2026-06-02 audit) — plausible-Sharpe bound (DAILY units) for the DSR V
# population. A real 0DTE daily Sharpe is |·| ~ 0.3 (ann ~5); the prior 1.0-daily
# cap (== ann ~15.9, "absurd" by its own comment) merely CLIPPED variance
# artifacts to ±1.0 instead of removing them — a single near-flat-variance spike
# (from the backtester's returns_std 1e-9 floor) clipped to 1.0 still inflated V
# and could drive SR*_ann to ~17, rejecting the whole front. The bound is now set
# to a genuinely plausible 0DTE daily Sharpe (|daily| ~ 0.3, ann ~5 -> 0.35) AND
# any member that HITS the bound is EXCLUDED from V (not clipped-in): a variance
# artifact is not a valid trial Sharpe and must not set the dispersion bar.
_MAX_PLAUSIBLE_DAILY_SHARPE = 0.35

# Euler-Mascheroni constant (Bailey & Lopez de Prado 2014, eq. for E[max SR]).
_EULER_MASCHERONI = 0.5772156649


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum DAILY Sharpe of N independent trials under the null.

    Bailey & Lopez de Prado (2014):

        SR*_N = sqrt(V) * [ (1 - gamma) * Phi^-1(1 - 1/N)
                            + gamma     * Phi^-1(1 - 1/(N*e)) ]

    where
        V     = sharpe_variance = Var of the trial (DAILY) Sharpe ratios,
        gamma = Euler-Mascheroni constant (~0.5772),
        Phi^-1 = inverse standard-normal CDF (percent-point function),
        e     = Euler's number.

    The bracketed term is the expected maximum of N i.i.d. standard normals;
    multiplying by sqrt(V) rescales it to the dispersion of the actual trial
    Sharpes. SR*_N increases monotonically in BOTH N and V.

    Args:
        n_trials: number of distinct strategies tried (effective trial count).
            For the front-level gate this is the deduped Pareto-front size.
            Clamped to >= 2 (Phi^-1(1 - 1/1) = +inf is undefined).
        sharpe_variance: variance of the trial DAILY Sharpe ratios (V).
            Clamped to >= 0. If 0 (degenerate / single member), SR* = 0.

    Returns:
        Expected maximum DAILY Sharpe under the null (float, per-period units).
    """
    from scipy.stats import norm as _norm_dist

    n = max(int(n_trials), 2)
    v = max(float(sharpe_variance), 0.0)
    z_max = (
        (1.0 - _EULER_MASCHERONI) * _norm_dist.ppf(1.0 - 1.0 / n)
        + _EULER_MASCHERONI * _norm_dist.ppf(1.0 - 1.0 / (n * math.e))
    )
    return float(math.sqrt(v) * z_max)


def deflated_sharpe_ratio(
    sharpe_daily: float,
    n_days: int,
    skew: float,
    kurtosis: float,
    sr_star: float,
) -> float:
    """Per-strategy Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

        DSR = Phi[ (SR - SR*) * sqrt(T - 1)
                   / sqrt(1 - skew*SR + (kurtosis - 1)/4 * SR^2) ]

    with T = n_days, SR = sharpe_daily (DAILY Sharpe units), SR* the expected
    maximum DAILY Sharpe under the null (from `expected_max_sharpe`).

    The denominator is the non-normality-adjusted standard error of the Sharpe
    estimator (Lo 2002 / Mertens). NOTE the convention used here, matching the
    task spec and Bailey & Lopez de Prado (2014) Eq. (9):

        Var(SR_hat) ~ [1 - g3*SR + (g4 - 1)/4 * SR^2] / (T - 1)

    where g3 = skewness and g4 = NON-EXCESS (Pearson) kurtosis, so a Gaussian
    (g4 = 3) gives the (g4 - 1)/4 = 1/2 term. scipy.stats.kurtosis returns
    EXCESS kurtosis by default (Gaussian -> 0); callers must therefore pass
    g4 = excess_kurtosis + 3 here. (The cross-checked alternative form
    1 + (g4_excess + 2)/4 * SR^2 - skew*SR is algebraically identical.)

    DSR is the probability that the strategy's TRUE Sharpe exceeds SR* given
    the estimation error -- i.e. P(SR > SR*). DSR ~ 0.5 when SR == SR*,
    -> 1 as SR >> SR*, -> 0 as SR << SR*.

    Units: SR, SR*, skew, kurtosis are all PER-PERIOD (daily). Callers that
    hold annualized Sharpes must de-annualize (divide by sqrt(252)) BOTH the
    member SR and the variance feeding SR* before calling this.

    Args:
        sharpe_daily: strategy's DAILY (de-annualized) Sharpe ratio.
        n_days: number of independent daily return observations T. If < 2,
            sqrt(T - 1) is undefined -> returns 0.0 (cannot assess).
        skew: skewness of daily returns (scipy convention, g3).
        kurtosis: NON-EXCESS kurtosis of daily returns (Gaussian = 3.0).
            Pass scipy excess kurtosis + 3.0.
        sr_star: expected maximum DAILY Sharpe under null (from
            `expected_max_sharpe`).

    Returns:
        DSR in [0, 1] = P(true daily SR > SR*).
    """
    from scipy.stats import norm as _norm_dist

    T = int(n_days)
    if T < 2:
        # sqrt(T-1) undefined / non-positive: cannot form the statistic.
        return 0.0

    sr = float(sharpe_daily)
    # Non-normality-adjusted variance factor (per-period, before /(T-1)).
    # 1 - g3*SR + (g4 - 1)/4 * SR^2, with g4 = non-excess kurtosis.
    denom_var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    # Guard: heavy negative skew + large SR can drive this non-positive;
    # clamp to a tiny positive so sqrt() is finite (degenerate SE -> the
    # sign of (SR - SR*) dominates and DSR saturates to 0 or 1).
    denom = math.sqrt(max(denom_var, 1e-12))

    z = (sr - sr_star) * math.sqrt(T - 1) / denom
    return float(_norm_dist.cdf(z))


def evolution_n_v_from_records(
    trial_records: "Dict[str, Tuple[float, int]]",
    *,
    min_days: int = 20,
) -> "Tuple[int, float]":
    """Derive the DSR (N, V) from the whole evolution's distinct trials.

    `trial_records` maps a canonical tree signature -> (raw ANNUALIZED training
    Sharpe, n_days) for every genuinely-distinct individual evaluated across the
    run (produced by gp_engine.evolve via the evaluator's trial recorder, keyed
    by `grammar.canonical_key` so commutative twins are already collapsed).

    Procedure (DAILY units, matching the rest of this module):
      * Keep only members with n_days >= min_days. A member with too few
        observations is not a valid Sharpe estimate and must NOT set the bar
        (Finding: V/N include invalid members). This is the ONLY validity
        filter — a backtest that ran IS a multiple-testing trial regardless of
        whether it later passed the GP quality/regime gates.
      * N = number of surviving distinct members (effective trial count).
      * V = population variance (ddof=0) of their DAILY Sharpes
        (SR_daily = SR_annualized / sqrt(252)).

    Returns (N, V). With < 2 valid members V = 0.0 (no dispersion to deflate);
    N is the count of VALID (n_days >= min_days, finite-Sharpe) members
    (clamped to >= 2 inside expected_max_sharpe).

    This is the fix for Blocker B (variance-inversion): N and V come from the
    population, so a single overfit spike (e.g. val Sharpe 9.4) is deflated
    AGAINST the population's dispersion instead of inflating its own bar.
    """
    sqrt_year = math.sqrt(TRADING_DAYS_PER_YEAR)
    daily = [
        float(sr) / sqrt_year
        for (sr, nd) in trial_records.values()
        if int(nd) >= int(min_days) and np.isfinite(sr)
    ]
    n = len(daily)
    if n < 2:
        return n, 0.0
    # HIGH H1 (2026-06-02 audit): EXCLUDE implausible Sharpes from V (do not clip
    # them in). A near-flat strategy can hit the returns_std 1e-9 floor in the
    # backtester and produce an astronomical daily Sharpe; the prior code CLIPPED
    # such a spike to ±1.0 daily (ann ~15.9) and KEPT it in the V sample, so one
    # variance artifact still inflated V -> SR* (SR*_ann up to ~17) and rejected
    # the whole front. A variance artifact is not a valid trial Sharpe, so it is
    # now DROPPED from the V estimate entirely; the bound is a genuinely plausible
    # 0DTE daily Sharpe (|daily| ~ 0.3 == ann ~5 -> 0.35). If excluding the
    # artifacts leaves < 2 valid members there is no dispersion to deflate (V=0).
    #
    # N is UNCHANGED — every trial that ran is a multiple-testing trial (B2); only
    # V's sample drops the variance artifacts. (N is the count of all valid-length
    # finite-Sharpe trials, computed above; only the DISPERSION estimate is
    # robustified here.)
    arr_full = np.asarray(daily, dtype=np.float64)
    _plausible = np.abs(arr_full) <= _MAX_PLAUSIBLE_DAILY_SHARPE
    arr = arr_full[_plausible]
    if arr.size < 2:
        return n, 0.0
    v = float(np.var(arr, ddof=0))
    return n, v


def dsr_gate_evolution(
    front: "List[dict]",
    *,
    n_trials: int,
    sharpe_variance: float,
    threshold: float = 0.5,
    min_days: int = 20,
    sharpe_key: str = "sharpe",
    n_days_key: str = "n_days",
    skew_key: str = "skew",
    kurtosis_key: str = "kurtosis",
    kurtosis_is_excess: bool = True,
) -> dict:
    """Apply the DSR multiple-testing gate using EVOLUTION-level (N, V).

    Unlike the retired front-variance gate, the multiple-testing inflation
    (N, V) is supplied by the caller from the WHOLE evolution's distinct
    trials (see `evolution_n_v_from_records`), NOT recomputed from the tiny
    final front. SR* is therefore the expected maximum Sharpe of the entire
    search, computed ONCE, and every champion is deflated against it. An
    overfit spike that survived to the front no longer inflates its own bar.

    Procedure (all in DAILY Sharpe units):
      1. SR* = expected_max_sharpe(N, V)  -- ONE value for the whole front.
      2. For each front member: de-annualize its Sharpe (SR_daily = SR / √252),
         convert kurtosis to non-excess if needed, and compute
         DSR_i = deflated_sharpe_ratio(SR_daily_i, T_i, skew_i, kurt_i, SR*).
      3. pass_i = (DSR_i >= threshold) and (T_i >= min_days).

    Args:
        front: list of per-member dicts. Each provides an ANNUALIZED Sharpe
            (`sharpe_key`), observation count (`n_days_key`), and return
            moments (`skew_key`, `kurtosis_key`). Missing moment keys default
            to 0.0.
        n_trials: N -- distinct individuals evaluated across the evolution.
        sharpe_variance: V -- variance of the evolution's DAILY trial Sharpes.
        threshold: DSR pass level. Default 0.5 == "Sharpe beats the
            multiple-testing-deflated SR*" (pure deflation; at the short fold
            lengths this project's 0DTE data supports, 0.5 is the calibrated
            bar -- higher thresholds demand short-sample significance the
            window cannot supply). See docs/methodology L2 for the calibration.
        min_days: minimum daily observations to assess a member; fewer -> fail.
        sharpe_key/n_days_key/skew_key/kurtosis_key: dict key names.
        kurtosis_is_excess: if True (default), supplied kurtosis is EXCESS
            (scipy, Gaussian = 0) and is +3.0'd to non-excess for the Lo/Bailey
            SE form. Set False if non-excess (Pearson) kurtosis is supplied.

    Returns:
        dict with:
            n_trials:    int, N (evolution-level, as supplied)
            sharpe_variance: float, V (evolution-level, as supplied)
            sr_star:     float, SR*_N (DAILY units)
            sr_star_annualized: float, SR* * sqrt(252) (reporting only)
            dsr:         list[float], per-member DSR aligned with `front`
            passed:      list[bool], per-member gate result
            n_passed:    int, sum(passed)
            threshold:   float, the threshold used
            min_days:    int, the min_days used
    """
    sqrt_year = math.sqrt(TRADING_DAYS_PER_YEAR)
    n = len(front)

    # SR* ONCE from the evolution-level (N, V). Independent of the front's own
    # spread, so an inflated front member cannot raise (or clear) its own bar.
    v = max(float(sharpe_variance), 0.0)
    sr_star = expected_max_sharpe(int(n_trials), v)

    base = {
        "n_trials": int(n_trials),
        "sharpe_variance": v,
        "sr_star": float(sr_star),
        "sr_star_annualized": float(sr_star * sqrt_year),
        "threshold": float(threshold),
        "min_days": int(min_days),
    }
    if n == 0:
        return {**base, "dsr": [], "passed": [], "n_passed": 0}

    dsr_vals: List[float] = []
    passed: List[bool] = []
    for m in front:
        T = int(m.get(n_days_key, 0))
        if T < int(min_days):
            dsr_vals.append(0.0)
            passed.append(False)
            continue
        sr_daily = float(m.get(sharpe_key, 0.0)) / sqrt_year
        skew = float(m.get(skew_key, 0.0))
        kurt = float(m.get(kurtosis_key, 0.0))
        if kurtosis_is_excess:
            kurt = kurt + 3.0  # excess -> non-excess (Pearson) for the SE form
        d = deflated_sharpe_ratio(
            sharpe_daily=sr_daily,
            n_days=T,
            skew=skew,
            kurtosis=kurt,
            sr_star=sr_star,
        )
        dsr_vals.append(d)
        passed.append(bool(d >= float(threshold)))

    return {**base, "dsr": dsr_vals, "passed": passed,
            "n_passed": int(sum(passed))}
