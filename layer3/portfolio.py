"""L3 Portfolio Manager: assess approved strategies as a portfolio.

Deterministic computations:
- Return correlation matrix
- Terminal concentration
- Regime concentration
- Directional exposure
- Portfolio drawdown simulation

LLM assessment (via Portfolio Manager agent):
- Allocation rationale
- Kill criteria
- Diversification warnings
- Proceed/wait recommendation

Minimum N=2 for portfolio analysis. N=1 produces a degenerate report.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from layer3.schemas import PortfolioAssessment


def compute_correlation_matrix(
    strategy_returns: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise daily-return correlation between strategies.

    Args:
        strategy_returns: Dict of strategy_id → daily return array.
            All arrays must have the same length (common date range).

    Returns:
        Nested dict: {sid_i: {sid_j: correlation}}.
    """
    sids = sorted(strategy_returns.keys())
    if len(sids) < 2:
        return {s: {s: 1.0} for s in sids}

    matrix = {}
    for i, si in enumerate(sids):
        matrix[si] = {}
        for j, sj in enumerate(sids):
            if i == j:
                matrix[si][sj] = 1.0
            elif j < i:
                matrix[si][sj] = matrix[sj][si]  # symmetric
            else:
                ri, rj = strategy_returns[si], strategy_returns[sj]
                n = min(len(ri), len(rj))
                if n < 5:
                    matrix[si][sj] = 0.0
                else:
                    corr = float(np.corrcoef(ri[:n], rj[:n])[0, 1])
                    matrix[si][sj] = round(corr, 4) if np.isfinite(corr) else 0.0
    return matrix


def compute_terminal_concentration(
    strategy_terminals: Dict[str, List[str]],
) -> Dict[str, int]:
    """Count how many strategies use each terminal in their entry tree.

    Returns dict of terminal_name → count.
    """
    counts = {}
    for sid, terminals in strategy_terminals.items():
        for t in set(terminals):  # deduplicate within strategy
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def compute_directional_balance(
    strategy_templates: Dict[str, str],
) -> str:
    """Compute net directional exposure from template types."""
    direction_map = {
        "bull_put_credit": "bullish",
        "bear_call_credit": "bearish",
        "iron_condor": "neutral",
        "iron_butterfly": "neutral",
        "ratio_put_backspread": "bearish",
    }
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for sid, tmpl in strategy_templates.items():
        # Strip suffixes
        base = tmpl.replace("_standard", "").replace("_narrow", "").replace("_wide", "")
        direction = direction_map.get(base, "neutral")
        counts[direction] += 1

    total = sum(counts.values())
    if total == 0:
        return "empty"
    parts = []
    for d in ("bullish", "bearish", "neutral"):
        if counts[d] > 0:
            parts.append(f"{counts[d]} {d}")
    net = counts["bullish"] - counts["bearish"]
    if net > 0:
        parts.append(f"net bullish by {net}")
    elif net < 0:
        parts.append(f"net bearish by {abs(net)}")
    else:
        parts.append("balanced")
    return ", ".join(parts)


def simulate_portfolio_drawdown(
    strategy_returns: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Simulate portfolio performance with given weights (default equal weight).

    Returns dict with: sharpe, max_drawdown, calmar, longest_losing_streak, worst_week.
    """
    sids = sorted(strategy_returns.keys())
    if not sids:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "longest_losing_streak": 0, "worst_week": 0.0}

    # Default equal weight
    if weights is None:
        w = 1.0 / len(sids)
        weights = {s: w for s in sids}

    # Compute portfolio daily returns
    n = min(len(strategy_returns[s]) for s in sids)
    port_returns = np.zeros(n)
    for sid in sids:
        port_returns += weights.get(sid, 0.0) * strategy_returns[sid][:n]

    # Sharpe
    std = float(np.std(port_returns))
    sharpe = float(np.mean(port_returns) / max(std, 1e-12) * math.sqrt(252))

    # Max drawdown
    cum = np.cumsum(port_returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar
    annual_return = float(np.mean(port_returns) * 252)
    calmar = annual_return / max(max_dd, 1e-12) if max_dd > 0 else 0.0

    # Longest losing streak
    longest = 0
    current = 0
    for r in port_returns:
        if r < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    # Worst week (5-day rolling sum)
    worst_week = 0.0
    if n >= 5:
        for i in range(n - 4):
            week_ret = float(np.sum(port_returns[i:i+5]))
            worst_week = min(worst_week, week_ret)

    return {
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "longest_losing_streak": longest,
        "worst_week": round(worst_week, 6),
    }


def build_portfolio_assessment(
    strategy_returns: Dict[str, np.ndarray],
    strategy_terminals: Dict[str, List[str]],
    strategy_templates: Dict[str, str],
) -> PortfolioAssessment:
    """Build the deterministic portion of the portfolio assessment.

    The LLM component (allocation rationale, kill criteria) is added
    separately by the Portfolio Manager agent.
    """
    n = len(strategy_returns)
    if n == 0:
        return PortfolioAssessment(
            n_strategies=0,
            diversification_warnings=["No strategies approved"],
            proceed_recommendation=False,
        )

    corr = compute_correlation_matrix(strategy_returns)
    terminal_conc = compute_terminal_concentration(strategy_terminals)
    direction = compute_directional_balance(strategy_templates)
    sim = simulate_portfolio_drawdown(strategy_returns)

    warnings = []
    if n == 1:
        warnings.append("Single-strategy portfolio — no diversification possible")

    # Check for high correlation pairs
    if n >= 2:
        sids = sorted(strategy_returns.keys())
        for i, si in enumerate(sids):
            for j, sj in enumerate(sids):
                if j > i and abs(corr.get(si, {}).get(sj, 0)) > 0.8:
                    warnings.append(f"High correlation ({corr[si][sj]:.2f}) between {si} and {sj}")

    # Check terminal concentration
    if n >= 2:
        for term, count in terminal_conc.items():
            if count / n > 0.6:
                warnings.append(f"Terminal concentration: {count}/{n} strategies use {term}")

    return PortfolioAssessment(
        n_strategies=n,
        correlation_matrix=corr,
        terminal_concentration=terminal_conc,
        directional_balance=direction,
        portfolio_sharpe=sim["sharpe"],
        portfolio_max_drawdown=sim["max_drawdown"],
        portfolio_calmar=sim["calmar"],
        diversification_warnings=warnings,
        proceed_recommendation=n >= 1 and sim["sharpe"] > 0,
    )
