"""Trailing-rolling local-window robust normalization (plan #4, RevIN-style).

WHY (measured, docs/trailing_rolling_normalization_spec_2026_06_03.md §0): the
expanding per-fold robust norm sets each LEVEL terminal's (center, scale) over the
WHOLE multi-regime train span, dominated by the 2022 high-vol regime. So a single
75th-pct entry threshold fires ~98% of 2022 bars but ~2-8% of 2024 bars → IV-gated
strategies cannot reach `min_trades` in the late folds → feasibility collapses
(fold-1 pilot 2/128). This module normalizes each LEVEL terminal against a CAUSAL
trailing window of recent days, so the signal is "high vs RECENT context" (tradeable
in every regime). Validated offline: 2024 firing 2-8%→23%, monthly-firing CV
1.26→0.69, and the gate's premium-capture lift goes +0.023→+0.051 (it stops firing
on the wrong — high-absolute-IV-but-also-high-RV — days).

Granularity is DAILY (one (center, scale) per day, broadcast to that day's minute
bars), matching the daily-regime frequency and the QC online-update hook. Causality
is enforced by shift(1) — day d uses days [d-W, d-1] only, so there is no same-day or
future leak and val/test bars normalize against their own (past) context without
train->val leakage.

Robustness: center = trailing median, scale = trailing IQR/1.349 (Cont 2001 heavy
tails), then the existing [-clip, +clip] winsorization (the "Huber" robustness of
plan #4 is satisfied by median/IQR + clip; no Huber-mean is added, spec §4.5).

This is the PROXY side. The codegen (QC) side computes the IDENTICAL per-day stats
online from its daily buffers; a parity test (spec §5) is the gate before deployment.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Robust scale: IQR -> sigma (matches terminal_stats.ROBUST_IQR_TO_SIGMA).
ROBUST_IQR_TO_SIGMA = 1.349
_MIN_SCALE = 1e-6
DEFAULT_CLIP = 5.0

# Window length: chosen by the 2026-06-03 W-sweep (docs spec §4.2) — the knee of
# firing-balance (CV=0.69) and near-peak premium-capture lift (0.051) with no
# fully-starved month (min monthly firing 5%). W in {10,20,30,40,60,90} measured;
# 20 dominates (W>=30 reintroduces 0%-firing months for negligible lift gain;
# W<=10 loses discrimination). MEASURED, not assumed.
DEFAULT_TRAILING_WINDOW = 20  # trading days

# LEVEL terminals (regime-dependent levels) get trailing-rolling norm; every other
# terminal keeps its static/expanding (center, scale). Spec §4.4. Bounded / diff /
# time-of-day terminals (BarOfDay, MinutesToClose, SessionPosition, SessionReturn,
# OvernightGap, ThetaUrgency, GridReliability, DeltaSpread1/5, VIXChange,
# UnrealizedProfitPct) are intentionally EXCLUDED — a trailing window adds noise with
# no regime benefit for them.
LEVEL_TERMINALS = frozenset({
    "ATM_IV", "ATM_IV_5m",
    "IV_5Dp", "IV_10Dp", "IV_25Dp", "IV_40Dp", "IV_ATMp",
    "IV_ATM", "IV_ATMc", "IV_40Dc", "IV_25Dc", "IV_10Dc", "IV_5Dc",
    "VIXSpot", "VIXMean5d", "VIXTermSlope",
    "RealizedVol30m", "RealizedVol30m_5m", "RV5d", "IVRVGap5d",
    "RawSpread", "RawSpread_5m", "PutCallSkew", "SPXReturn3d",
})


def _min_periods_for(W: int) -> int:
    """Warmup floor: need at least this many prior days before a trailing stat is
    used; before that, the bar falls back to the static (expanding) stat."""
    return max(5, W // 2)


def compute_trailing_day_stats(
    raw_values: np.ndarray,
    dates,
    W: int = DEFAULT_TRAILING_WINDOW,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """Per-DAY causal trailing (center=median, scale=IQR/1.349) for one terminal.

    Args:
        raw_values: per-bar RAW (un-normalized) terminal values.
        dates: per-bar date (same length as raw_values).
        W: trailing window length in trading days.
        min_periods: minimum prior days before a stat is non-NaN (default _min_periods_for).

    Returns:
        DataFrame indexed by unique normalized day, columns ['center', 'scale'].
        Rows in the warmup region (< min_periods prior days) are NaN — the caller
        substitutes the static fallback there.
    """
    if min_periods is None:
        min_periods = _min_periods_for(W)
    day = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).normalize()
    # daily summary = median of the day's bars (robust, QC-reproducible; spec §4.3)
    daily = (pd.Series(np.asarray(raw_values, dtype=float), index=day)
             .groupby(level=0).median().sort_index())
    daily.index = pd.DatetimeIndex(daily.index)
    roll = daily.rolling(W, min_periods=min_periods)
    # shift(1): strictly PAST days (causal — no same-day leak)
    center = roll.median().shift(1)
    q75 = roll.quantile(0.75).shift(1)
    q25 = roll.quantile(0.25).shift(1)
    scale = ((q75 - q25) / ROBUST_IQR_TO_SIGMA).clip(lower=_MIN_SCALE)
    return pd.DataFrame({"center": center, "scale": scale})


def apply_trailing_norm(
    raw_dict: Dict[str, np.ndarray],
    dates,
    W: int = DEFAULT_TRAILING_WINDOW,
    static_stats: Optional[Dict[str, Tuple[float, float, str]]] = None,
    level_terminals=LEVEL_TERMINALS,
    min_periods: Optional[int] = None,
    clip: float = DEFAULT_CLIP,
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """Normalize a raw terminal-data dict: LEVEL terminals via causal per-day trailing
    robust stats; all others via `static_stats` (the per-fold expanding/frozen stats).

    Drop-in for the normalization step: feed the output of
    prepare_terminal_data(data, normalize_terminals=False) (raw synthesized arrays).

    Args:
        raw_dict: {name: raw per-bar array} (1-D arrays; non-1-D passed through).
        dates: per-bar dates aligned to the arrays.
        W: trailing window (trading days).
        static_stats: {name: (center, scale, kind)} — used for non-level terminals
            AND as the warmup fallback for level terminals. If a terminal is absent
            here and not a level terminal, it passes through RAW (e.g., the position-
            aware UnrealizedProfitPct, which the evaluator consumes un-normalized).
        level_terminals: which terminals get trailing-rolling.
        min_periods: warmup floor (default _min_periods_for(W)).
        clip: winsorization bound (matches the evaluator's [-5,+5] OOD clamp).

    Returns:
        (normalized_dict, per_day_stats) where per_day_stats[name] is the DataFrame
        from compute_trailing_day_stats (for the codegen parity test / inspection).
    """
    if min_periods is None:
        min_periods = _min_periods_for(W)
    static_stats = static_stats or {}
    day = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).normalize()
    out: Dict[str, np.ndarray] = {}
    per_day_stats: Dict[str, pd.DataFrame] = {}

    def _static_norm(arr: np.ndarray, name: str) -> Optional[np.ndarray]:
        fb = static_stats.get(name)
        if fb is None:
            return None
        c, s = float(fb[0]), max(float(fb[1]), _MIN_SCALE)
        return np.clip((arr - c) / s, -clip, clip)

    for name, arr in raw_dict.items():
        arr = np.asarray(arr, dtype=float)
        if name in level_terminals and arr.ndim == 1 and len(arr) == len(day):
            stats = compute_trailing_day_stats(arr, dates, W, min_periods)
            per_day_stats[name] = stats
            c = stats["center"].reindex(day).to_numpy()
            s = stats["scale"].reindex(day).to_numpy()
            warm = np.isnan(c) | np.isnan(s)
            z = np.empty_like(arr)
            ok = ~warm
            z[ok] = np.clip((arr[ok] - c[ok]) / s[ok], -clip, clip)
            if warm.any():
                fb_z = _static_norm(arr, name)
                # warmup days fall back to the static (expanding) stat; if no static
                # stat exists, leave neutral 0.0 (rare for a level terminal).
                z[warm] = fb_z[warm] if fb_z is not None else 0.0
            out[name] = z
        else:
            z = _static_norm(arr, name) if arr.ndim == 1 else None
            out[name] = z if z is not None else arr  # passthrough (e.g. UPP, vectors)
    return out, per_day_stats
