"""Unit tests for layer2.shuffle — H2 shuffled-L1 control."""
import numpy as np
import pandas as pd
import pytest

from layer2.io import (
    PRICE_COLUMN, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.shuffle import (
    PRESERVE_COLUMNS, SHUFFLE_TARGET_COLUMNS, shuffle_l1,
)


def _make_data(n_dates=4, bars_per_date=10, emb_dim=384) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for di, date in enumerate([f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]):
        spot = 4500.0
        regime = di % 4   # vary regime across days for stratification testing
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.0)
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: spot, "ATM_IV": 0.20,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18,
                "PredRV15": float(rng.standard_normal()),
                "PredRV30": float(rng.standard_normal()),
                "PredRegime": regime,
                "PredSpread": float(rng.standard_normal()),
            }
            for col in TYPED_VECTOR_COLUMNS:
                # Distinct vectors per row so we can detect permutation
                row[col] = (np.ones(emb_dim) * (di * 100 + w)).astype(np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = float(rng.uniform())
            rows.append(row)
    return pd.DataFrame(rows)


def test_shuffle_preserves_temporal_columns():
    df = _make_data()
    shuffled = shuffle_l1(df, seed=42, regime_stratify=False)
    for col in PRESERVE_COLUMNS:
        if col in df.columns:
            np.testing.assert_array_equal(
                df[col].values, shuffled[col].values,
                err_msg=f"preserve column {col} was modified by shuffle",
            )


def test_shuffle_permutes_target_columns():
    df = _make_data(n_dates=10, bars_per_date=5)
    shuffled = shuffle_l1(df, seed=42, regime_stratify=False, block="window")
    # At least one EMB_GRID value must differ post-shuffle
    # (vectors are numpy arrays so element-wise comparison)
    n_changed = sum(
        1 for orig, perm in zip(df["EMB_GRID"], shuffled["EMB_GRID"])
        if not np.array_equal(orig, perm)
    )
    assert n_changed > 0, "shuffle did not permute target columns"


def test_shuffle_reproducible_with_same_seed():
    df = _make_data()
    s1 = shuffle_l1(df, seed=42, regime_stratify=False)
    s2 = shuffle_l1(df, seed=42, regime_stratify=False)
    # Spot check a typed vector and a probe scalar
    for orig, perm in zip(s1["EMB_GRID"], s2["EMB_GRID"]):
        np.testing.assert_array_equal(orig, perm)
    np.testing.assert_array_equal(s1["PredRV15"].values, s2["PredRV15"].values)


def test_shuffle_different_seeds_differ():
    df = _make_data()
    s1 = shuffle_l1(df, seed=42, regime_stratify=False)
    s2 = shuffle_l1(df, seed=99, regime_stratify=False)
    n_diff = sum(
        1 for a, b in zip(s1["EMB_GRID"], s2["EMB_GRID"])
        if not np.array_equal(a, b)
    )
    assert n_diff > 0, "different seeds produced identical shuffles"


def test_regime_stratification_keeps_within_regime():
    """When stratified, a row with regime=0 must end up with target values
    that came from a regime=0 row originally."""
    df = _make_data(n_dates=8, bars_per_date=10)
    shuffled = shuffle_l1(df, seed=42, regime_stratify=True, block="window")
    # Tag each original row's regime via a "fingerprint" in the EMB vector
    # (the synthetic data uses (di*100+w) which encodes regime di%4)
    # Verify: for each shuffled row, the EMB fingerprint comes from a row
    # whose regime matched
    for i, row in shuffled.iterrows():
        post_first_value = row["EMB_GRID"][0]
        original_regime = int(row["PredRegime"])
        # Find which original row had this EMB value
        match_indices = [
            j for j, orig in enumerate(df["EMB_GRID"])
            if orig[0] == post_first_value
        ]
        assert match_indices, "shuffled value doesn't appear in original data"
        # All matches should have the same regime as the post-shuffle row
        for j in match_indices:
            j_regime = int(df.iloc[j]["PredRegime"])
            assert j_regime == original_regime, (
                f"row {i} (regime {original_regime}) got value from "
                f"row {j} (regime {j_regime}) — stratification violated"
            )


def test_shuffle_preserves_row_count_and_ordering():
    df = _make_data()
    shuffled = shuffle_l1(df, seed=42)
    assert len(shuffled) == len(df)
    # Index ordering preserved (sort key is (date, window_idx))
    np.testing.assert_array_equal(
        df["date"].values, shuffled["date"].values,
    )
    np.testing.assert_array_equal(
        df["window_idx"].values, shuffled["window_idx"].values,
    )


def test_shuffle_block_must_be_valid():
    df = _make_data()
    with pytest.raises(ValueError, match="block must be"):
        shuffle_l1(df, seed=42, block="hour")
