"""Shuffled-L1 control utility for H2 (real-L1 vs shuffled-L1 GP comparison).

The shuffled-L1 GP is the load-bearing experiment for H2:
  - Real-L1 condition: GP receives genuine L1 outputs (typed vectors + probes)
  - Shuffled-L1 condition: same shape and z-score statistics, but contents
    randomly permuted across windows
  - If real-L1 GP fitness > shuffled-L1 GP fitness by >1σ, the L1
    representation is transmitting signal that GP can use

Per AI Engineer's "load-bearing methodological detail" review:
  - Shuffle by WINDOW or DAY blocks (NOT row-level — destroys intra-window structure)
  - REGIME-STRATIFY the permutation so shuffled embeddings come from
    same-regime days; otherwise shuffled L1 is OOD-by-construction and
    we'd be measuring distributional mismatch, not signal content
  - PRESERVE temporal scalars (date, MinutesToClose, SPXClose, VIX*) — these
    are causally tied to time and shuffling them tests the wrong null
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.io import (
    PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)


# Columns to PERMUTE under shuffled-L1 (everything L1-derived):
# - All 7 typed vectors
# - All 8 probe scalars (V2: original 4 + GammaAccel/SmileConvexity/Jump/FlowToxicity)
# - All 4 regime probabilities
from layer2.io import PROBE_SCALAR_COLUMNS_V2
SHUFFLE_TARGET_COLUMNS: Tuple[str, ...] = (
    TYPED_VECTOR_COLUMNS + PROBE_SCALAR_COLUMNS_V2 + REGIME_PROB_COLUMNS
)

# Columns to PRESERVE (temporal/price scalars that aren't L1 outputs):
PRESERVE_COLUMNS: Tuple[str, ...] = (
    "date", "window_idx", "bar_position",
    PRICE_COLUMN, "MinutesToClose",
    "VIXSpot", "VIXTermSlope", "ATM_IV",
    "RawSpread", "DeltaSpread1", "DeltaSpread5",
    "RealizedVol30m",
)


def shuffle_l1(df: pd.DataFrame, seed: int = 42,
               block: str = "window",
               regime_stratify: bool = False,
               regime_column: str = "PredRegime") -> pd.DataFrame:
    """Permute L1-derived columns across windows; preserve temporal/price columns.

    Args:
        df: L1Output DataFrame (must have 'date' and the SHUFFLE_TARGET_COLUMNS)
        seed: random seed for reproducibility (separate from GP seed)
        block: 'window' (shuffle individual windows — preserves intra-window
                 structure) or 'day' (shuffle whole days — stronger null,
                 destroys day-level info)
        regime_stratify: if True, only permute within same-regime groups.
                 Default False (2026-05-09): unstratified is the stronger null.
                 The GP already has bypass scalars (ATM_IV, VIXSpot) for regime
                 context — it doesn't need the encoder's regime predictions
                 preserved in the shuffled arm. Stratified shuffle weakens
                 the null by giving the shuffled arm free regime information.
        regime_column: column to stratify by (default PredRegime, the
                 4-class L1 regime label)

    Returns:
        New DataFrame with SHUFFLE_TARGET_COLUMNS permuted; all other
        columns identical to input. Same row count, same row ordering by
        (date, window_idx) as input.
    """
    if block not in ("window", "day"):
        raise ValueError(f"block must be 'window' or 'day', got {block!r}")

    rng = np.random.default_rng(seed)
    df = df.copy()

    # Subset to columns we have (some may be missing in v2 corpus, e.g.
    # EMB_FLOW_AGG zeroed for v2 but still a column)
    target_cols = [c for c in SHUFFLE_TARGET_COLUMNS if c in df.columns]
    if not target_cols:
        raise ValueError(
            f"no shuffle-target columns found in DataFrame; "
            f"expected at least one of {SHUFFLE_TARGET_COLUMNS}"
        )

    if block == "day":
        return _shuffle_by_day(df, target_cols, rng, regime_stratify, regime_column)
    return _shuffle_by_window(df, target_cols, rng, regime_stratify, regime_column)


def _shuffle_by_window(df: pd.DataFrame, target_cols: List[str],
                       rng: np.random.Generator,
                       regime_stratify: bool, regime_column: str) -> pd.DataFrame:
    """Shuffle individual rows (windows) — preserves intra-window structure since
    each row IS one window in the L1Output schema.

    Stratification: rows with the same regime label get permuted only among
    themselves. Prevents shuffled-L1 from becoming OOD-by-construction.

    M3 invariant (Commit 6): `regime_column` is itself in SHUFFLE_TARGET_COLUMNS,
    but under `regime_stratify=True` the permutation is within-stratum, so each
    row receives a donor value from a row with the SAME regime label. The row-
    level regime assignment is therefore preserved even though the column is
    nominally "shuffled." We lock this invariant with a runtime assertion
    below — if a future refactor changes `perm` to cross strata, the assertion
    trips loudly instead of silently producing regime-contaminated shuffles.
    """
    n = len(df)
    perm = np.arange(n)

    # Snapshot the regime BEFORE shuffling so we can assert invariance after
    regime_values_before = (
        df[regime_column].astype(int).values
        if regime_stratify and regime_column in df.columns
        else None
    )

    if regime_stratify and regime_column in df.columns:
        unique_regimes = np.unique(regime_values_before)
        for r in unique_regimes:
            idx = np.where(regime_values_before == r)[0]
            if len(idx) > 1:
                shuffled_idx = rng.permutation(idx)
                perm[idx] = shuffled_idx
    else:
        perm = rng.permutation(n)

    # Apply permutation to the target columns; other columns stay aligned to row
    out = df.copy()
    for col in target_cols:
        out[col] = df[col].values[perm]

    # M3 lock: under stratified shuffle, row-level regime must be invariant.
    if regime_values_before is not None and regime_column in out.columns:
        regime_values_after = out[regime_column].astype(int).values
        if not np.array_equal(regime_values_before, regime_values_after):
            raise RuntimeError(
                "shuffled-L1 regime-stratification invariant violated: "
                "row-level PredRegime changed after shuffle. The permutation "
                "leaked across regime strata — this corrupts H2 because "
                "shuffled embeddings now come from wrong-regime days."
            )
    return out


def _shuffle_by_day(df: pd.DataFrame, target_cols: List[str],
                    rng: np.random.Generator,
                    regime_stratify: bool, regime_column: str) -> pd.DataFrame:
    """Shuffle whole days — stronger null than window-level shuffle.

    For each date, swap its target-column values with another date's values.
    Stratification: only shuffle within same-regime days (use the day's
    modal regime).
    """
    out = df.copy()
    dates = sorted(df["date"].unique())

    if regime_stratify and regime_column in df.columns:
        # Day-level regime = modal regime of that day
        day_regime = (
            df.groupby("date")[regime_column].agg(lambda s: s.mode().iloc[0])
        )
        regime_to_dates = {}
        for d in dates:
            r = int(day_regime.loc[d])
            regime_to_dates.setdefault(r, []).append(d)
        date_perm_map = {}
        for r, day_list in regime_to_dates.items():
            if len(day_list) > 1:
                permuted = rng.permutation(day_list).tolist()
                date_perm_map.update(dict(zip(day_list, permuted)))
            else:
                # Only one day in this regime — must map to itself
                date_perm_map[day_list[0]] = day_list[0]
    else:
        permuted = rng.permutation(dates).tolist()
        date_perm_map = dict(zip(dates, permuted))

    # Build the swapped DataFrame: for each row, look up its date's "swap partner"
    # and copy the partner's same-window-position target values into this row.
    # On uneven days, excess bars get the column's global mean (NOT original
    # values — keeping originals would leak unshuffled L1 signal).
    for original_date, swap_date in date_perm_map.items():
        if original_date == swap_date:
            continue
        original_idx = df.index[df["date"] == original_date].tolist()
        swap_idx = df.index[df["date"] == swap_date].tolist()
        n_swap = min(len(original_idx), len(swap_idx))
        for col in target_cols:
            out.loc[original_idx[:n_swap], col] = (
                df.loc[swap_idx[:n_swap], col].values
            )
            # Fill excess bars with column mean to prevent data leak
            if len(original_idx) > n_swap:
                col_mean = df[col].mean() if not hasattr(df[col].iloc[0], '__len__') else None
                for excess_idx in original_idx[n_swap:]:
                    if col_mean is not None:
                        out.at[excess_idx, col] = col_mean
                    # For typed-vector columns (ndarray), fill with zeros
                    else:
                        out.at[excess_idx, col] = np.zeros_like(df[col].iloc[0])
    return out
