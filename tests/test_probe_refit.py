"""Tests for per-fold probe refitting (L2.62 look-ahead fix).

Covers standalone probe fitting/prediction, forward-fill, per-fold divergence,
and condition B/D enforcement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from layer1.inference.batch_forecast import (
    fit_probes_standalone,
    predict_probes_standalone,
)
from layer2.experiment import (
    ExperimentConfig,
    _forward_fill_probes,
    _refit_probes_for_fold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_emb(n: int = 300, d: int = 64, seed: int = 42):
    """Small synthetic emb_fine for fast tests (real is 3072-d)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


def _make_targets(n: int = 300, seed: int = 42):
    rng = np.random.RandomState(seed)
    return {
        "rv_15": rng.randn(n).astype(np.float32),
        "rv_30": rng.randn(n).astype(np.float32),
        "regime": rng.randint(0, 4, n),
        "spread": rng.randn(n).astype(np.float32),
    }


def _make_probe_bundle(n_windows: int = 120, n_dates: int = 10,
                        emb_dim: int = 64, seed: int = 42,
                        include_l2_62: bool = False):
    """Synthetic probe bundle matching the NPZ format."""
    rng = np.random.RandomState(seed)
    # 12 windows per date (matching real 5-min windows)
    windows_per_date = n_windows // n_dates
    dates = []
    bar_positions = []
    for d in range(n_dates):
        date_str = f"2024-{1 + d // 30:02d}-{1 + d % 30:02d}"
        for w in range(windows_per_date):
            dates.append(date_str)
            bar_positions.append(59 + w * 30)  # 59, 89, 119, ...

    n = len(dates)
    bundle = {
        "emb_fine": rng.randn(n, emb_dim).astype(np.float32),
        "window_dates": np.array(dates, dtype="U10"),
        "bar_positions": np.array(bar_positions, dtype=np.int32),
        "target_rv_15": rng.randn(n).astype(np.float32),
        "target_rv_30": rng.randn(n).astype(np.float32),
        "target_regime": rng.randint(0, 4, n).astype(np.int64),
        "target_spread": rng.randn(n).astype(np.float32),
        "valid_rv_15": np.ones(n, dtype=bool),
        "valid_rv_30": np.ones(n, dtype=bool),
        "valid_regime": np.ones(n, dtype=bool),
        "valid_spread": np.ones(n, dtype=bool),
    }
    if include_l2_62:
        gamma = rng.randn(n).astype(np.float32)
        gamma[:5] = np.nan  # some invalid
        bundle["target_gamma_accel"] = gamma
        bundle["valid_gamma_accel"] = ~np.isnan(gamma)
        bundle["target_smile_convex"] = rng.randn(n).astype(np.float32)
        bundle["valid_smile_convex"] = np.ones(n, dtype=bool)
        bundle["target_flow_tox"] = rng.randn(n).astype(np.float32)
        bundle["valid_flow_tox"] = np.ones(n, dtype=bool)
        jump = rng.randint(0, 2, n).astype(np.float32)
        bundle["target_jump_30"] = jump
        bundle["valid_jump_30"] = np.ones(n, dtype=bool)
    return bundle


def _make_minute_df(dates, bars_per_date: int = 405):
    """Synthetic minute-resolution DataFrame."""
    rows = []
    for date in dates:
        for bp in range(bars_per_date):
            rows.append({"date": date, "bar_position": bp,
                         "ATM_IV": 0.2, "PredRV15": 99.0,
                         "PredRV30": 99.0, "PredRegime": 0,
                         "PredSpread": 99.0,
                         "RegimeProb0": 0.25, "RegimeProb1": 0.25,
                         "RegimeProb2": 0.25, "RegimeProb3": 0.25})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitProbesStandalone:
    """Test standalone probe fitting + prediction roundtrip."""

    def test_fit_and_predict_core_probes(self):
        emb = _make_synthetic_emb(200)
        targets = _make_targets(200)
        probes = fit_probes_standalone(emb, targets)
        assert "rv_15" in probes
        assert "rv_30" in probes
        assert "regime" in probes
        assert "spread" in probes

        preds = predict_probes_standalone(probes, emb)
        assert preds["rv_15"].shape == (200,)
        assert preds["rv_30"].shape == (200,)
        assert preds["regime_proba"].shape == (200, 4)
        assert preds["predicted_regime"].shape == (200,)
        assert preds["spread"].shape == (200,)

    def test_rv_predictions_are_positive(self):
        """RV goes through exp+sqrt — output must be positive."""
        emb = _make_synthetic_emb(200)
        targets = _make_targets(200)
        probes = fit_probes_standalone(emb, targets)
        preds = predict_probes_standalone(probes, emb)
        assert np.all(preds["rv_15"] > 0)
        assert np.all(preds["rv_30"] > 0)

    def test_regime_probabilities_sum_to_one(self):
        emb = _make_synthetic_emb(200)
        targets = _make_targets(200)
        probes = fit_probes_standalone(emb, targets)
        preds = predict_probes_standalone(probes, emb)
        sums = preds["regime_proba"].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_l2_62_probes_with_nan(self):
        """L2.62 probes should handle NaN in targets."""
        emb = _make_synthetic_emb(300)
        targets = _make_targets(300)
        # Add probes with some NaN
        rng = np.random.RandomState(99)
        gamma = rng.randn(300).astype(np.float32)
        gamma[:50] = np.nan
        targets["gamma_accel"] = gamma
        targets["jump_30"] = rng.randint(0, 2, 300).astype(np.float32)
        targets["jump_30"][:30] = np.nan

        probes = fit_probes_standalone(emb, targets)
        assert "gamma_accel" in probes
        assert "jump_30" in probes

        preds = predict_probes_standalone(probes, emb)
        assert "gamma_accel" in preds
        assert "jump_proba" in preds


class TestPerFoldProbesDiffer:
    """Verify probes fitted on different train_ends produce different predictions."""

    def test_different_train_ends_produce_different_probes(self):
        rng = np.random.RandomState(42)
        n = 400
        emb = rng.randn(n, 64).astype(np.float32)
        # Make first half and second half structurally different
        emb[:200] += 2.0
        targets = {
            "rv_15": rng.randn(n).astype(np.float32),
            "rv_30": rng.randn(n).astype(np.float32),
            "regime": rng.randint(0, 4, n),
            "spread": rng.randn(n).astype(np.float32),
        }
        # Fold 1: fit on first 100
        probes_1 = fit_probes_standalone(emb[:100], {k: v[:100] for k, v in targets.items()})
        # Fold 2: fit on first 300
        probes_2 = fit_probes_standalone(emb[:300], {k: v[:300] for k, v in targets.items()})

        # Predict on same test data
        preds_1 = predict_probes_standalone(probes_1, emb[300:])
        preds_2 = predict_probes_standalone(probes_2, emb[300:])

        # Predictions must differ (different training data = different models)
        diff_rv = np.abs(preds_1["rv_15"] - preds_2["rv_15"]).mean()
        assert diff_rv > 1e-6, f"RV predictions should differ but diff={diff_rv}"


class TestForwardFillProbes:
    """Test forward-fill from 5-min to 1-min resolution."""

    def test_bars_before_first_window_get_zero(self):
        dates = ["2024-01-01"]
        minute_df = _make_minute_df(dates, bars_per_date=120)
        # Only one 5-min window at bar_position=59
        pred_5min = pd.DataFrame({
            "date": ["2024-01-01"],
            "bar_position": [59],
            "PredRV15": [0.42],
            "PredSpread": [0.10],
        })
        result = _forward_fill_probes(minute_df, pred_5min)
        # Bars 0-58 should get 0.0 (before first window)
        early = result[result["bar_position"] < 59]
        assert (early["PredRV15"] == 0.0).all()
        assert (early["PredSpread"] == 0.0).all()

    def test_bar_at_window_gets_window_value(self):
        dates = ["2024-01-01"]
        minute_df = _make_minute_df(dates, bars_per_date=120)
        pred_5min = pd.DataFrame({
            "date": ["2024-01-01"],
            "bar_position": [59],
            "PredRV15": [0.42],
        })
        result = _forward_fill_probes(minute_df, pred_5min)
        at_59 = result[result["bar_position"] == 59]["PredRV15"].iloc[0]
        assert at_59 == pytest.approx(0.42)

    def test_forward_fill_carries_value(self):
        dates = ["2024-01-01"]
        minute_df = _make_minute_df(dates, bars_per_date=120)
        pred_5min = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01"],
            "bar_position": [59, 89],
            "PredRV15": [0.42, 0.55],
        })
        result = _forward_fill_probes(minute_df, pred_5min)
        # Bar 60-88 should carry forward from window at 59
        at_70 = result[result["bar_position"] == 70]["PredRV15"].iloc[0]
        assert at_70 == pytest.approx(0.42)
        # Bar 89+ should have new value
        at_89 = result[result["bar_position"] == 89]["PredRV15"].iloc[0]
        assert at_89 == pytest.approx(0.55)

    def test_cross_date_independence(self):
        """Each date's forward-fill is independent."""
        dates = ["2024-01-01", "2024-01-02"]
        minute_df = _make_minute_df(dates, bars_per_date=120)
        pred_5min = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "bar_position": [59, 59],
            "PredRV15": [0.42, 0.99],
        })
        result = _forward_fill_probes(minute_df, pred_5min)
        day1 = result[(result["date"] == "2024-01-01") &
                       (result["bar_position"] == 59)]["PredRV15"].iloc[0]
        day2 = result[(result["date"] == "2024-01-02") &
                       (result["bar_position"] == 59)]["PredRV15"].iloc[0]
        assert day1 == pytest.approx(0.42)
        assert day2 == pytest.approx(0.99)


class TestRefitProbesForFold:
    """Integration test for _refit_probes_for_fold."""

    def test_replaces_all_core_probe_columns(self):
        bundle = _make_probe_bundle(n_windows=120, n_dates=10, emb_dim=64)
        dates = sorted(set(bundle["window_dates"]))
        minute_df = _make_minute_df(dates[:5], bars_per_date=120)

        result = _refit_probes_for_fold(minute_df, bundle, train_end=dates[4])

        enriched_bars = result[result["bar_position"] >= 59]
        # All 4 core probe columns should be replaced (not sentinel 99.0)
        for col in ["PredRV15", "PredRV30", "PredSpread"]:
            assert col in result.columns, f"{col} missing"
            assert not (enriched_bars[col] == 99.0).any(), \
                f"{col} should have been replaced"
        assert "PredRegime" in result.columns
        # Regime probabilities should be present and sum to 1
        for k in range(4):
            assert f"RegimeProb{k}" in result.columns

    def test_replaces_l2_62_probe_columns(self):
        # Need >100 valid training windows for probes (fit_probes_standalone
        # threshold). Use 360 windows / 30 dates so train half has ~180.
        bundle = _make_probe_bundle(
            n_windows=360, n_dates=30, emb_dim=64, include_l2_62=True)
        dates = sorted(set(bundle["window_dates"]))
        minute_df = _make_minute_df(dates[:15], bars_per_date=120)

        result = _refit_probes_for_fold(minute_df, bundle, train_end=dates[14])

        enriched_bars = result[result["bar_position"] >= 59]
        for col in ["PredGammaAccel", "PredSmileConvexity",
                     "PredFlowToxicity", "PredJump"]:
            assert col in result.columns, f"{col} missing from output"
            # Should have non-zero values for enriched bars
            assert enriched_bars[col].abs().sum() > 0, \
                f"{col} is all zeros — probe may not have been fitted"

    def test_row_count_preserved(self):
        bundle = _make_probe_bundle(n_windows=120, n_dates=10, emb_dim=64)
        dates = sorted(set(bundle["window_dates"]))
        minute_df = _make_minute_df(dates[:5], bars_per_date=120)

        result = _refit_probes_for_fold(minute_df, bundle, train_end=dates[4])
        assert len(result) == len(minute_df), \
            f"Row count changed: {len(minute_df)} → {len(result)}"

    def test_int64_regime_with_invalid_entries(self):
        """Regression: real bundle stores target_regime as int64 with -1 for
        invalid entries. NaN assignment to int64 crashed before the fix."""
        bundle = _make_probe_bundle(n_windows=120, n_dates=10, emb_dim=64)
        # Simulate production: int64 regime with -1 invalid entries
        n = len(bundle["window_dates"])
        regime = bundle["target_regime"].copy()
        regime = regime.astype(np.int64)
        regime[:10] = -1
        bundle["target_regime"] = regime
        valid = np.ones(n, dtype=bool)
        valid[:10] = False
        bundle["valid_regime"] = valid

        dates = sorted(set(bundle["window_dates"]))
        minute_df = _make_minute_df(dates[:5], bars_per_date=120)

        # Should not crash
        result = _refit_probes_for_fold(minute_df, bundle, train_end=dates[4])
        assert "PredRegime" in result.columns
        # Regime predictions should be valid classes (0-3), not -1
        enriched = result[result["bar_position"] >= 59]
        regime_vals = enriched["PredRegime"].unique()
        assert all(v in {0, 1, 2, 3} for v in regime_vals), \
            f"Regime predictions contain invalid values: {regime_vals}"


class TestRecalibrateSeedThresholds:
    """Test seed tree EphReal recalibration to per-fold norm stats."""

    def test_gt_threshold_recalibrated(self):
        from layer2.grammar import FuncNode, TermNode, TermDef, FuncDef, GType
        from layer2.terminal_stats import (
            recalibrate_seed_thresholds, TERMINAL_NORM_STATS, normalize_threshold,
        )
        GT = FuncDef("GT", [GType.REAL, GType.REAL], GType.BOOL)

        # Build: GT(ATM_IV, EphReal(frozen_normalized_0.18))
        frozen_thresh = normalize_threshold("ATM_IV", 0.18)
        tree = FuncNode(
            defn=GT,
            children=[
                TermNode(defn=TermDef("ATM_IV", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=frozen_thresh),
            ],
        )
        # Per-fold stats: shifted center and scale
        fold_stats = dict(TERMINAL_NORM_STATS)
        orig_center, orig_scale, method = TERMINAL_NORM_STATS["ATM_IV"]
        fold_stats["ATM_IV"] = (orig_center + 0.02, orig_scale * 1.5, method)

        recalibrate_seed_thresholds(tree, fold_stats)

        # The EphReal value should now differ from frozen
        new_val = tree.children[1].value
        assert new_val != pytest.approx(frozen_thresh, abs=1e-6), \
            "EphReal should have been recalibrated"
        # Verify it denormalizes to the same raw value
        fold_c, fold_s, _ = fold_stats["ATM_IV"]
        raw_recovered = new_val * fold_s + fold_c
        assert raw_recovered == pytest.approx(0.18, abs=1e-6), \
            f"Recalibrated threshold should denormalize to 0.18, got {raw_recovered}"

    def test_non_comparison_nodes_unchanged(self):
        from layer2.grammar import FuncNode, TermNode, TermDef, FuncDef, GType
        from layer2.terminal_stats import recalibrate_seed_thresholds, TERMINAL_NORM_STATS
        ADD = FuncDef("Add", [GType.REAL, GType.REAL], GType.REAL)

        # Build: Add(ATM_IV, EphReal(1.5)) — not GT/LT, should be untouched
        tree = FuncNode(
            defn=ADD,
            children=[
                TermNode(defn=TermDef("ATM_IV", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=1.5),
            ],
        )
        recalibrate_seed_thresholds(tree, dict(TERMINAL_NORM_STATS))
        assert tree.children[1].value == 1.5  # unchanged


class TestConditionBRequiresBundle:
    """Test that B/D conditions enforce probe_bundle_path for early folds."""

    def test_probes_only_without_bundle_raises_for_fold1(self):
        config = ExperimentConfig(
            l1_parquet_path="dummy.parquet",
            condition="probes-only",
            probe_bundle_path=None,
        )
        from layer2.experiment import run_walk_forward, WALK_FORWARD_FOLDS
        folds = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] == 1]
        with pytest.raises(ValueError, match="probe look-ahead leak"):
            run_walk_forward(config, folds=folds, output_base="/tmp/test")

    def test_real_l1_without_bundle_raises_for_fold1(self):
        config = ExperimentConfig(
            l1_parquet_path="dummy.parquet",
            condition="real-l1",
            probe_bundle_path=None,
        )
        from layer2.experiment import run_walk_forward, WALK_FORWARD_FOLDS
        folds = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] == 1]
        with pytest.raises(ValueError, match="probe look-ahead leak"):
            run_walk_forward(config, folds=folds, output_base="/tmp/test")

    def test_scalar_only_without_bundle_ok(self):
        """Scalar-only doesn't use probes — no bundle needed."""
        config = ExperimentConfig(
            l1_parquet_path="dummy.parquet",
            condition="scalar-only",
            probe_bundle_path=None,
        )
        from layer2.experiment import WALK_FORWARD_FOLDS
        folds = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] == 4]
        # Should not raise (fold 4's train_end >= cutoff)
        # Can't actually run because dummy.parquet doesn't exist,
        # but the assertion check happens before the Parquet load.
        # We just verify the assertion doesn't fire.
        # The test passes if we get past the assertion to the actual Parquet load.
        try:
            from layer2.experiment import run_walk_forward
            run_walk_forward(config, folds=folds, output_base="/tmp/test_scalar")
        except (FileNotFoundError, OSError):
            pass  # Expected — dummy.parquet doesn't exist
        except ValueError as e:
            if "probe look-ahead leak" in str(e):
                pytest.fail(f"Scalar-only should not require probe bundle: {e}")
            raise
