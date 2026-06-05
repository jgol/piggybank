"""I/O adapters for L1Output Parquet → L2 GP consumption.

Per Software Architect review (Week 1 sprint): the Parquet→bar_data path
is the load-bearing friction point. Two issues this module resolves:

1. Typed-vector columns (EMB_GRID, EMB_VIX, etc.) round-trip from Parquet
   as `list[float]` per row, not `np.ndarray`. The evaluator's
   `EvaluationContext.update()` type-dispatches on `isinstance(val,
   np.ndarray)` to route vectors to the vector_buffer. Convert once at
   load time.

2. Naive `df.iloc[i].to_dict()` passes EVERY column to the evaluator,
   including `date`, `window_idx`, `bar_position` — which the evaluator
   tries to `_safe_real()` on a date string and silently inserts 0.0
   under key "date". Use `L1_TERMINAL_COLUMNS` whitelist to filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical column lists (must match grammar terminal names + L1Output schema)
# ---------------------------------------------------------------------------

# typed-vector columns — list[float] in Parquet, np.ndarray after load
TYPED_VECTOR_COLUMNS: Tuple[str, ...] = (
    "EMB_SHARED",
    "EMB_GRID",
    "EMB_STRIKE",
    "EMB_SPX",
    "EMB_VIX",
    "EMB_FLOW_RAW",
    "EMB_FLOW_AGG",
    # R2 per-variate embeddings (bypass mean-pool bottleneck)
    "EMB_VAR_ATM_IV",
    "EMB_VAR_ATM_SPREAD",
    "EMB_VAR_SPX_RET",
)

# Channel 2: probe forecast scalars (PascalCase per grammar terminal names)
# Original 4 probes (implemented):
PROBE_SCALAR_COLUMNS: Tuple[str, ...] = (
    "PredRV15",
    "PredRV30",
    "PredRegime",   # int 0-3, derived from argmax(RegimeProb*)
    "PredSpread",
)
# expansion — 4 new probes (pending implementation in encoder inference):
PROBE_SCALAR_COLUMNS_V2: Tuple[str, ...] = PROBE_SCALAR_COLUMNS + (
    "PredGammaAccel",      # gamma/theta ratio change at t+15
    "PredSmileConvexity",  # IV butterfly spread change at t+15
    "PredJump",            # P(|ret_30| > 1%) binary
    "PredFlowToxicity",    # order flow toxicity z-score at t+15
)

# Channel 3: raw bypass scalars
BYPASS_SCALAR_COLUMNS: Tuple[str, ...] = (
    "RawSpread",
    "DeltaSpread1",
    "DeltaSpread5",
    "ATM_IV",
    "MinutesToClose",
    "VIXSpot",
    "VIXTermSlope",
    "RealizedVol30m",
    # BarOfDay: synthesized at evaluation time from MinutesToClose (not in Parquet)
)

# Per-class regime probabilities (Parquet flat columns RegimeProb0..3)
REGIME_PROB_COLUMNS: Tuple[str, ...] = tuple(f"RegimeProb{k}" for k in range(4))

# Price column for the backtester (not a GP terminal, but needed for P&L)
PRICE_COLUMN = "SPXClose"

# Index columns — date-keyed, NOT GP-evaluable
INDEX_COLUMNS: Tuple[str, ...] = ("date", "window_idx", "bar_position")

# All columns that should reach EvaluationContext.update() as bar_data values
# (typed vectors + scalars; excludes index and price which are handled separately
# by the backtester loop, NOT by GP terminal lookup)
# TODO(): Switch PROBE_SCALAR_COLUMNS → PROBE_SCALAR_COLUMNS_V2 when
# encoder inference ships V2 columns (PredGammaAccel, PredSmileConvexity,
# PredJump, PredFlowToxicity). Until then, V2 probes will read as 0.0.
L1_TERMINAL_COLUMNS: Tuple[str, ...] = (
    TYPED_VECTOR_COLUMNS
    + PROBE_SCALAR_COLUMNS
    + BYPASS_SCALAR_COLUMNS
    + REGIME_PROB_COLUMNS
)


# ---------------------------------------------------------------------------
# Parquet load with typed-vector conversion
# ---------------------------------------------------------------------------

@dataclass
class L1ParquetSchema:
    """Validated metadata about a loaded L1Output Parquet."""
    n_rows: int
    n_dates: int
    typed_vector_dim: int            # D*n_layers (default 384 = 128*3)
    has_typed_vectors: bool
    has_probe_scalars: bool
    has_bypass_scalars: bool
    has_price_column: bool


def load_l1_parquet(path: str | Path,
                    validate: bool = True) -> Tuple[pd.DataFrame, L1ParquetSchema]:
    """Load L1Output Parquet, converting typed-vector list columns to np.ndarray.

    Args:
        path: path to the L1Output Parquet file
        validate: if True, assert schema consistency (all 7 typed vectors
                  present, all scalars present, no NaN in price column).

    Returns:
        (df, schema) — DataFrame with typed-vector cols as np.ndarray,
        and a schema metadata object for downstream consumers.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"L1 Parquet not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")

    # Convert typed-vector columns from list[float] → np.ndarray once at load.
    # Subsequent EvaluationContext.update() will route via isinstance(np.ndarray).
    typed_vec_dim = 0
    for col in TYPED_VECTOR_COLUMNS:
        if col not in df.columns:
            if validate:
                raise ValueError(
                    f"L1 Parquet at {path} missing typed-vector column {col!r}. "
                    f"Available columns: {list(df.columns)[:20]}..."
                )
            continue
        # Per-row coercion. Empty/None rows become zero-vectors; non-empty rows
        # become float32 ndarrays. Per-row dim must be consistent.
        first_non_null = next((v for v in df[col] if v is not None and len(v) > 0), None)
        if first_non_null is None:
            if validate:
                raise ValueError(f"typed-vector column {col!r} is all-None")
            continue
        col_dim = len(first_non_null)
        if typed_vec_dim == 0:
            typed_vec_dim = col_dim
        elif typed_vec_dim != col_dim:
            raise ValueError(
                f"typed-vector dimensions inconsistent: "
                f"first vector dim={typed_vec_dim}, {col!r} dim={col_dim}"
            )
        df[col] = df[col].apply(
            lambda x, _d=col_dim: np.asarray(x, dtype=np.float32) if x is not None
            else np.zeros(_d, dtype=np.float32)
        )

    if validate:
        # Probe scalars
        missing_probes = [c for c in PROBE_SCALAR_COLUMNS if c not in df.columns]
        if missing_probes:
            raise ValueError(f"L1 Parquet missing probe columns: {missing_probes}")
        # Bypass scalars
        missing_bypass = [c for c in BYPASS_SCALAR_COLUMNS if c not in df.columns]
        if missing_bypass:
            raise ValueError(f"L1 Parquet missing bypass columns: {missing_bypass}")
        # Price column
        if PRICE_COLUMN not in df.columns:
            raise ValueError(
                f"L1 Parquet missing price column {PRICE_COLUMN!r} — "
                f"backtester needs this for P&L"
            )
        if df[PRICE_COLUMN].isna().any():
            n_nan = int(df[PRICE_COLUMN].isna().sum())
            raise ValueError(
                f"L1 Parquet has {n_nan} NaN value(s) in price column "
                f"{PRICE_COLUMN!r} — backtester would silently produce zero returns"
            )

    schema = L1ParquetSchema(
        n_rows=len(df),
        n_dates=df["date"].nunique() if "date" in df.columns else 0,
        typed_vector_dim=typed_vec_dim,
        has_typed_vectors=any(c in df.columns for c in TYPED_VECTOR_COLUMNS),
        has_probe_scalars=any(c in df.columns for c in PROBE_SCALAR_COLUMNS),
        has_bypass_scalars=any(c in df.columns for c in BYPASS_SCALAR_COLUMNS),
        has_price_column=PRICE_COLUMN in df.columns,
    )
    return df, schema


def load_minute_parquet(path: str | Path) -> Tuple[pd.DataFrame, L1ParquetSchema]:
    """Load 1-minute Parquet for vectorized GP evaluation.

    Supports two variants:
    - Scalar-only (l1_minute_scalars.parquet): bypass scalars only, for Condition A.
    - Enriched (l1_minute_enriched.parquet): scalars + typed vectors + probes,
      for Conditions B/C/D. Typed vectors are stored as list[float] and
      converted to np.ndarray on load.

    Validation: requires date, SPXClose, and at least one bypass scalar column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Minute Parquet not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")

    # Basic validation
    if "date" not in df.columns:
        raise ValueError(f"Minute Parquet missing 'date' column: {list(df.columns)}")
    if PRICE_COLUMN not in df.columns:
        raise ValueError(f"Minute Parquet missing price column '{PRICE_COLUMN}'")
    if df[PRICE_COLUMN].isna().any():
        n_nan = int(df[PRICE_COLUMN].isna().sum())
        raise ValueError(
            f"Minute Parquet has {n_nan} NaN value(s) in '{PRICE_COLUMN}'"
        )
    # Check at least some bypass scalars are present
    present_bypass = [c for c in BYPASS_SCALAR_COLUMNS if c in df.columns]
    if not present_bypass:
        raise ValueError(
            f"Minute Parquet has no bypass scalar columns. "
            f"Expected some of: {BYPASS_SCALAR_COLUMNS}"
        )

    # Detect enriched minute Parquet (with typed vectors + probes)
    present_vecs = [c for c in TYPED_VECTOR_COLUMNS if c in df.columns]
    present_probes = [c for c in PROBE_SCALAR_COLUMNS if c in df.columns]
    vec_dim = 0
    if present_vecs:
        sample = df[present_vecs[0]].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            vec_dim = len(sample)
        # Convert list[float] → np.ndarray for typed-vector columns
        for col in present_vecs:
            df[col] = df[col].apply(
                lambda x, _d=vec_dim: np.asarray(x, dtype=np.float32)
                if isinstance(x, (list, np.ndarray)) else np.zeros(_d, dtype=np.float32)
            )

    schema = L1ParquetSchema(
        n_rows=len(df),
        n_dates=df["date"].nunique(),
        typed_vector_dim=vec_dim,
        has_typed_vectors=len(present_vecs) > 0,
        has_probe_scalars=len(present_probes) > 0,
        has_bypass_scalars=True,
        has_price_column=True,
    )
    return df, schema


# ---------------------------------------------------------------------------
# Date-based train/val/test split (walk-forward)
# ---------------------------------------------------------------------------

def split_by_date(df: pd.DataFrame,
                  train_end: str,
                  val_end: Optional[str] = None,
                  embargo_days: int = 5,
                  train_start: Optional[str] = None,
                  test_end: Optional[str] = None) -> dict:
    """Split a loaded L1Output DataFrame into train/val/test by trading date.

    Walk-forward (no shuffling) with embargo days between train and val/test
    per de Prado (2018) — prevents window-overlap leakage.

    Args:
        df: loaded L1Output DataFrame (must have 'date' column as YYYY-MM-DD str)
        train_end: last date INCLUDED in training (e.g., "2024-09-30")
        val_end: optional — last date in validation (e.g., "2025-01-31").
                 If None, val is empty and test starts after train + embargo.
        embargo_days: trading-day gap between train end and val/test start
        train_start: optional — first date INCLUDED in training. Default
                     = the earliest date in df. v8 OOR pilot uses this to
                     train on a 2022-Q4 bear-regime slice.
        test_end: optional — last date INCLUDED in test. Default = the
                  latest date in df. OOR pilot uses this to bound test
                  to 2023-Q1.

    Returns:
        dict with keys "train", "val" (may be empty), "test" — each a DataFrame.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame missing 'date' column for split_by_date")

    df = df.copy()
    df["date"] = df["date"].astype(str)

    # Sort all unique trading dates ascending
    all_dates = sorted(df["date"].unique())
    if not all_dates:
        return {"train": df.iloc[0:0], "val": df.iloc[0:0], "test": df.iloc[0:0]}

    train_end = str(train_end)
    if val_end is not None:
        val_end = str(val_end)
    if train_start is not None:
        train_start = str(train_start)
    if test_end is not None:
        test_end = str(test_end)

    # v8: optional train_start floor + test_end ceiling (for OOR pilots).
    # train_dates: [train_start (default earliest), train_end]
    train_dates = [
        d for d in all_dates
        if d <= train_end and (train_start is None or d >= train_start)
    ]
    if not train_dates:
        raise ValueError(
            f"no dates in [train_start={train_start}, train_end={train_end}] "
            f"in data; check date bounds + parquet coverage"
        )
    train_end_idx = all_dates.index(train_dates[-1])

    # Embargo gap → val start
    val_start_idx = train_end_idx + 1 + embargo_days
    val_dates: List[str] = []
    test_dates: List[str] = []

    if val_end is not None:
        # Val window: [val_start_idx, last_date ≤ val_end]
        val_candidates = [
            d for i, d in enumerate(all_dates)
            if i >= val_start_idx and d <= val_end
        ]
        val_dates = val_candidates
        if val_dates:
            test_start_idx = all_dates.index(val_dates[-1]) + 1 + embargo_days
        else:
            test_start_idx = val_start_idx
        test_candidates = [
            d for i, d in enumerate(all_dates) if i >= test_start_idx
        ]
    else:
        test_candidates = [d for i, d in enumerate(all_dates) if i >= val_start_idx]

    # v8: optional test_end ceiling for OOR
    if test_end is not None:
        test_dates = [d for d in test_candidates if d <= test_end]
    else:
        test_dates = test_candidates

    # Prior-session daily VIX for each slice's first day, taken from the FULL
    # corpus (the embargo-gap session that sits just before the slice). Passed to
    # prepare_terminal_data(vix_prior=...) so its VIX lookahead-lag uses the true
    # prior close on the first slice day instead of that day's own (lookahead)
    # close — closing the residual first-slice-day lookahead (code review
    # 2026-05-31). These are real, already-closed values, so no lookahead.
    def _prior_vix(slice_dates: List[str]) -> dict:
        if not slice_dates:
            return {}
        i = all_dates.index(slice_dates[0])
        if i == 0:
            return {}  # slice starts at corpus start — no prior session exists
        prev_rows = df[df["date"] == all_dates[i - 1]]
        out = {}
        for c in ("VIXSpot", "VIXTermSlope"):
            if c in df.columns and len(prev_rows):
                out[c] = float(prev_rows[c].values[-1])
        return out

    return {
        "train": df[df["date"].isin(train_dates)].reset_index(drop=True),
        "val": df[df["date"].isin(val_dates)].reset_index(drop=True),
        "test": df[df["date"].isin(test_dates)].reset_index(drop=True),
        "vix_prior": {
            "train": _prior_vix(train_dates),
            "val": _prior_vix(val_dates),
            "test": _prior_vix(test_dates),
        },
    }


# ---------------------------------------------------------------------------
# Bar-data iteration (the backtester consumes one row at a time)
# ---------------------------------------------------------------------------

def iter_bar_data(df: pd.DataFrame,
                  cols: Optional[Iterable[str]] = None) -> Iterable[dict]:
    """Yield dicts of {column_name → value} per row, restricted to GP terminal cols.

    Use the L1_TERMINAL_COLUMNS whitelist by default — keeps date, window_idx,
    bar_position OUT of EvaluationContext.update() bar_data, preventing the
    "date string silently coerced to 0.0" bug.

    Typed-vector columns are already np.ndarray (per load_l1_parquet), so
    the evaluator's update() method routes them to vector_buffer correctly.
    """
    if cols is None:
        cols = [c for c in L1_TERMINAL_COLUMNS if c in df.columns]
    cols = list(cols)
    for _, row in df.iterrows():
        yield {c: row[c] for c in cols if c in row.index}
