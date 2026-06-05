"""Unit tests for layer2.io — L1 Parquet adapter."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layer2.io import (
    L1_TERMINAL_COLUMNS,
    PRICE_COLUMN,
    PROBE_SCALAR_COLUMNS,
    REGIME_PROB_COLUMNS,
    TYPED_VECTOR_COLUMNS,
    BYPASS_SCALAR_COLUMNS,
    iter_bar_data,
    load_l1_parquet,
    split_by_date,
)


def _make_synthetic_l1_df(n_dates: int = 5, bars_per_date: int = 10,
                          emb_dim: int = 384) -> pd.DataFrame:
    """Build a synthetic L1Output DataFrame matching the expected schema."""
    rng = np.random.default_rng(42)
    rows = []
    dates = [f"2024-{m:02d}-01" for m in range(1, n_dates + 1)]
    for date in dates:
        for w in range(bars_per_date):
            row = {
                "date": date,
                "window_idx": w,
                "bar_position": w,
                PRICE_COLUMN: 4500.0 + rng.normal(0, 5),
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = rng.standard_normal(emb_dim).astype(np.float32).tolist()
            for col in PROBE_SCALAR_COLUMNS:
                row[col] = float(rng.standard_normal())
            for col in BYPASS_SCALAR_COLUMNS:
                row[col] = float(abs(rng.standard_normal()))
            for col in REGIME_PROB_COLUMNS:
                row[col] = float(rng.uniform(0, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_l1_parquet_round_trip(tmp_path):
    """Saving + loading roundtrips correctly with typed vectors as np.ndarray."""
    df = _make_synthetic_l1_df()
    parquet_path = tmp_path / "l1_test.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")

    loaded, schema = load_l1_parquet(parquet_path)
    assert schema.n_rows == len(df)
    assert schema.n_dates == 5
    assert schema.typed_vector_dim == 384
    assert schema.has_typed_vectors
    assert schema.has_probe_scalars
    assert schema.has_bypass_scalars
    assert schema.has_price_column

    # Typed vectors are np.ndarray after load (NOT list)
    for col in TYPED_VECTOR_COLUMNS:
        assert isinstance(loaded[col].iloc[0], np.ndarray), \
            f"{col} should be np.ndarray after load"
        assert loaded[col].iloc[0].dtype == np.float32
        assert loaded[col].iloc[0].shape == (384,)


def test_load_l1_parquet_validates_schema(tmp_path):
    """Missing typed-vector columns raise on validate=True."""
    df = _make_synthetic_l1_df()
    df = df.drop(columns=["EMB_GRID"])  # break schema
    p = tmp_path / "broken.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="EMB_GRID"):
        load_l1_parquet(p, validate=True)


def test_load_l1_parquet_validates_price(tmp_path):
    """NaN in price column raises (would silently zero P&L)."""
    df = _make_synthetic_l1_df()
    df.loc[0, PRICE_COLUMN] = np.nan
    p = tmp_path / "nan_price.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="NaN"):
        load_l1_parquet(p, validate=True)


def test_split_by_date_walk_forward():
    df = _make_synthetic_l1_df(n_dates=10, bars_per_date=2)
    splits = split_by_date(df, train_end="2024-05-01", val_end="2024-08-01",
                           embargo_days=0)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    # No date overlap
    train_dates = set(splits["train"]["date"])
    val_dates = set(splits["val"]["date"])
    test_dates = set(splits["test"]["date"])
    assert train_dates.isdisjoint(val_dates)
    assert val_dates.isdisjoint(test_dates)
    # Train dates ≤ train_end; val dates > train_end
    assert all(d <= "2024-05-01" for d in train_dates)


def test_split_by_date_embargo_creates_gap():
    df = _make_synthetic_l1_df(n_dates=20, bars_per_date=1)
    splits = split_by_date(df, train_end="2024-10-01", embargo_days=3)
    train_dates = sorted(splits["train"]["date"].unique())
    test_dates = sorted(splits["test"]["date"].unique())
    if train_dates and test_dates:
        # The first 3 dates after train_end should NOT appear in test
        all_dates = sorted(df["date"].unique())
        train_end_idx = all_dates.index(train_dates[-1])
        embargoed = all_dates[train_end_idx + 1 : train_end_idx + 1 + 3]
        for d in embargoed:
            assert d not in test_dates, f"date {d} should be embargoed"


def test_iter_bar_data_excludes_index_columns():
    """iter_bar_data must NOT yield 'date', 'window_idx', 'bar_position'."""
    df = _make_synthetic_l1_df(n_dates=2, bars_per_date=2)
    # Convert typed vectors to ndarray for fair test (load_l1_parquet does this)
    for col in TYPED_VECTOR_COLUMNS:
        df[col] = df[col].apply(lambda x: np.asarray(x, dtype=np.float32))

    bars = list(iter_bar_data(df))
    assert len(bars) == 4
    for bar in bars:
        assert "date" not in bar
        assert "window_idx" not in bar
        assert "bar_position" not in bar
        # But all GP terminal columns should be present
        for col in TYPED_VECTOR_COLUMNS:
            assert col in bar
            assert isinstance(bar[col], np.ndarray)
        for col in PROBE_SCALAR_COLUMNS:
            assert col in bar
        for col in BYPASS_SCALAR_COLUMNS:
            assert col in bar


def test_iter_bar_data_typed_vectors_are_arrays():
    """The dispatched typed vectors must be np.ndarray for evaluator routing."""
    df = _make_synthetic_l1_df(n_dates=1, bars_per_date=1)
    for col in TYPED_VECTOR_COLUMNS:
        df[col] = df[col].apply(lambda x: np.asarray(x, dtype=np.float32))
    bars = list(iter_bar_data(df))
    for col in TYPED_VECTOR_COLUMNS:
        assert isinstance(bars[0][col], np.ndarray)
        assert bars[0][col].dtype == np.float32


def test_l1_terminal_columns_excludes_index_and_price():
    """Sanity: the canonical terminal column list doesn't include date/price."""
    assert "date" not in L1_TERMINAL_COLUMNS
    assert "window_idx" not in L1_TERMINAL_COLUMNS
    assert "bar_position" not in L1_TERMINAL_COLUMNS
    assert PRICE_COLUMN not in L1_TERMINAL_COLUMNS
