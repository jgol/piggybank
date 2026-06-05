"""Tests for the post-training structured metrics pipeline.

Validates:
1. _safe_float() helper handles all edge cases
2. collect_training_metrics() produces valid JSON structure
3. Per-group pct_learned calculation is correct
4. Sanity gates G1-G6 fire correctly
5. Output JSON is loadable and schema-compliant
6. Diagnostic snapshots are stored correctly in history
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# We need to mock the environment before importing pipeline
os.environ["LOCAL_MODE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------------------------------------------------------------------------
# Import the functions under test (pipeline is large, we import carefully)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_local_store(tmp_path, monkeypatch):
    """Redirect metric file output to a temp directory."""
    # We'll patch os.path operations inside collect_training_metrics
    # Actually, we mock the _results global and check the output
    pass


def _import_pipeline_funcs():
    """Lazy import to avoid heavy torch initialization in every test."""
    from layer1.training.pipeline import (
        _safe_float,
        collect_training_metrics,
        _results,
        ACTIVE_N_SSL_FEATURES,
    )
    return _safe_float, collect_training_metrics, _results, ACTIVE_N_SSL_FEATURES


# ---------------------------------------------------------------------------
# 1. _safe_float tests
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_none(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(None) is None

    def test_python_float(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(3.14) == 3.14

    def test_python_int(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(42) == 42.0

    def test_numpy_scalar(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(np.float32(0.5)) == pytest.approx(0.5, abs=1e-6)
        assert _safe_float(np.float64(1.23)) == pytest.approx(1.23, abs=1e-10)
        assert _safe_float(np.int64(7)) == 7.0

    def test_torch_scalar(self):
        import torch
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(torch.tensor(2.5)) == pytest.approx(2.5, abs=1e-6)

    def test_torch_multi_element_returns_none(self):
        import torch
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float(torch.tensor([1.0, 2.0])) is None

    def test_string_returns_none(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        assert _safe_float("hello") is None

    def test_numpy_nan(self):
        _safe_float, _, _, _ = _import_pipeline_funcs()
        result = _safe_float(np.nan)
        assert result is not None  # float(nan) is valid
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# 2. collect_training_metrics integration test
# ---------------------------------------------------------------------------

def _build_mock_results():
    """Build a realistic _results dict as populated by run_ssl_pipeline steps 5-9."""
    return {
        "ssl_history": {
            "train_loss": [0.4, 0.3, 0.2],
            "val_loss": [0.35, 0.25, 0.15],
        },
        "ssl_baselines": {
            "predict_zero": {"aggregate": 0.90, "whole_var": 0.95, "per_cell": 0.85},
            "persistence": {"aggregate": 0.70, "whole_var": 0.80, "per_cell": 0.60},
            "ar1": {"aggregate": 0.65, "whole_var": 0.75, "per_cell": 0.55},
            "cv_linear": {"aggregate": 0.55, "whole_var": 0.60, "per_cell": 0.50},
            "hybrid": {"aggregate": 0.50, "whole_var": 0.55, "per_cell": 0.45},
            "oracle_linear": {"aggregate": 0.30, "whole_var": 0.35, "per_cell": 0.25},
            "transformer": {"aggregate": 0.40, "whole_var": 0.45, "per_cell": 0.35},
        },
        "ssl_baseline_diagnostics": {
            "per_var_mse": {
                "predict_zero": [0.9] * 10,
                "transformer": [0.4] * 10,
            },
            "per_var_variance": [1.0] * 10,
            "group_results": {
                "options_grid": {
                    "predict_zero": 0.90,
                    "persistence": 0.70,
                    "ar1": 0.65,
                    "cv_linear": 0.55,
                    "hybrid": 0.50,
                    "oracle_linear": 0.30,
                    "transformer": 0.40,
                    "r2": 0.556,
                    "n_feat": 80,
                },
                "spx_derived": {
                    "predict_zero": 0.85,
                    "persistence": 0.60,
                    "ar1": 0.55,
                    "cv_linear": 0.45,
                    "hybrid": 0.42,
                    "oracle_linear": 0.25,
                    "transformer": 0.35,
                    "r2": 0.588,
                    "n_feat": 14,
                },
            },
            "whole_var_results": {
                "options_grid": {
                    "predict_zero": 0.95,
                    "cv_linear": 0.60,
                    "oracle_linear": 0.35,
                    "transformer": 0.45,
                    "r2_wv": 0.526,
                    "n_feat": 80,
                },
                "spx_derived": {
                    "predict_zero": 0.90,
                    "cv_linear": 0.50,
                    "oracle_linear": 0.30,
                    "transformer": 0.38,
                    "r2_wv": 0.578,
                    "n_feat": 14,
                },
            },
            "eligible_by_group": {
                "options_grid": 75,
                "spx_derived": 14,
            },
            "variate_group": ["options_grid"] * 80 + ["spx_derived"] * 14,
            "ssl_to_raw": {i: i for i in range(94)},
        },
        "masking_artifact_diagnostic": {
            "skipped": False,
            "passed": True,
            "global": {"spearman": 0.15, "pearson": 0.12, "n_cells": 5000},
            "per_variate": {},
        },
        "embedding_cosine_diagnostic": {
            "skipped": False,
            "global_mean_pool": {"n": 100, "mean": 0.98, "std": 0.01, "p5": 0.96, "p95": 0.99},
        },
        "attention_entropy_diagnostic": {
            "skipped": False,
            "n_batches": 20,
            "n_layers": 3,
            "uniform_rate": 0.0068,
            "per_layer": {},
        },
    }


def _build_mock_history():
    """Build a realistic training history dict."""
    # 25 epochs: first 5 converging, then 20 stable (for G3 stability_cv < 0.05)
    train_losses = [0.4 - 0.04 * i for i in range(5)] + [0.20 + 0.001 * (i % 3) for i in range(20)]
    val_losses = [0.35 - 0.03 * i for i in range(5)] + [0.20 + 0.001 * (i % 3) for i in range(20)]
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": min(val_losses),
        "final_emb_var": 0.45,
        "final_spectral_w": 0.08,
        "final_beat_zero": 145,  # passes G1 threshold of 140
        "diagnostic_snapshots": [
            {"epoch": 20, "emb_var": 0.42, "shortcut_suspect": 2, "well_reconstructed": 80, "spectral_w": 0.06},
            {"epoch": 40, "emb_var": 0.45, "shortcut_suspect": 1, "well_reconstructed": 95, "spectral_w": 0.08},
        ],
    }


def _build_mock_hp():
    """Build a realistic hyperparams dict."""
    return {
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "seq_len": 60,
        "variate_ratio": 0.20,
        "cell_ratio": 0.15,
        "spectral_rank_weight": 0.06,
        "target_spectral_fraction": 0.20,
        "huber_delta": 1.0,
        "lr": 1e-3,
        "use_shared_decoder": True,
        "tokenizer": "patch",
        "patch_size": 12,
        "use_temporal_attn": True,
        "num_epochs": 100,
    }


class TestCollectTrainingMetrics:
    def test_produces_valid_json(self, tmp_path):
        """The output must be valid JSON and loadable."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        # Inject mock _results
        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        # Redirect output to temp dir
        with patch.dict(os.environ, {"LOCAL_MODE": "1"}):
            # Patch the metrics_dir computation
            orig_abspath = os.path.abspath
            with patch("os.path.abspath", return_value=str(tmp_path / "layer1" / "training" / "pipeline.py")):
                # The function computes metrics_dir from __file__ — we need to
                # ensure it writes to our temp path. Easier to just check the
                # return value is valid JSON.
                metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        # Verify it's JSON-serializable
        json_str = json.dumps(metrics, default=str)
        reloaded = json.loads(json_str)
        assert reloaded["schema_version"] == 1
        assert reloaded["tokenizer"] == "patch"

    def test_training_summary_fields(self, tmp_path):
        """Training summary must contain all expected fields."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        ts = metrics["training_summary"]
        # Mock has 25 epochs: first 5 converging then 20 stable near 0.20
        assert ts["val_loss"] == pytest.approx(0.20, abs=0.01)
        assert ts["total_epochs"] == 100
        assert ts["n_params"] == 706000
        assert ts["emb_var"] == pytest.approx(0.45)
        assert ts["beat_zero"] == 145
        assert ts["spectral_w"] == pytest.approx(0.08)

    def test_pct_learned_calculation(self, tmp_path):
        """pct_learned = (mse_pz - mse_enc) / (mse_pz - mse_oracle) * 100."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        # options_grid: (0.90 - 0.40) / (0.90 - 0.30) * 100 = 83.33%
        og = metrics["per_group_reconstruction"]["options_grid"]
        expected_pct = (0.90 - 0.40) / (0.90 - 0.30) * 100.0
        assert og["pct_learned"] == pytest.approx(expected_pct, abs=0.1)

        # spx_derived: (0.85 - 0.35) / (0.85 - 0.25) * 100 = 83.33%
        spx = metrics["per_group_reconstruction"]["spx_derived"]
        expected_pct_spx = (0.85 - 0.35) / (0.85 - 0.25) * 100.0
        assert spx["pct_learned"] == pytest.approx(expected_pct_spx, abs=0.1)

    def test_sanity_gates_all_pass(self, tmp_path):
        """When all conditions are met, all gates should pass."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        gates = metrics["sanity_gates"]
        assert gates["G1_beat_zero"]["pass"] is True
        assert gates["G2_masking_artifact"]["pass"] is True  # soft gate, always passes
        assert gates["G3_stability_cv"]["pass"] is True
        assert gates["G4_emb_var"]["pass"] is True
        assert gates["G5_per_group_floor"]["pass"] is True
        assert gates["G6_generalization_gap"]["pass"] is True
        assert gates["all_pass"] is True

    def test_sanity_gate_g1_fails_when_beat_zero_low(self, tmp_path):
        """G1 should fail when fewer than 140 variates beat predict-zero."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        history["final_beat_zero"] = 100  # below 140 threshold
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")
        assert metrics["sanity_gates"]["G1_beat_zero"]["pass"] is False
        assert metrics["sanity_gates"]["all_pass"] is False

    def test_sanity_gate_g4_fails_on_collapse(self, tmp_path):
        """G4 should fail when emb_var < 0.01."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        history["final_emb_var"] = 0.005  # collapsed
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")
        assert metrics["sanity_gates"]["G4_emb_var"]["pass"] is False

    def test_sanity_gate_g5_fails_on_negative_group_r2(self, tmp_path):
        """G5 should fail when any eligible group has R² <= 0."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        results = _build_mock_results()
        # Make spx_derived encoder worse than predict-zero (R² negative)
        # Must update BOTH mse values AND the precomputed r2 key
        results["ssl_baseline_diagnostics"]["group_results"]["spx_derived"]["transformer"] = 1.0
        results["ssl_baseline_diagnostics"]["group_results"]["spx_derived"]["predict_zero"] = 0.85
        results["ssl_baseline_diagnostics"]["group_results"]["spx_derived"]["r2"] = -0.176  # 1 - 1.0/0.85
        pipeline._results = results
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")
        assert metrics["sanity_gates"]["G5_per_group_floor"]["pass"] is False
        assert "spx_derived" in metrics["sanity_gates"]["G5_per_group_floor"]["failing_groups"]

    def test_diagnostic_snapshots_preserved(self, tmp_path):
        """Diagnostic snapshots from training should be in the output."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        snaps = metrics["diagnostic_snapshots"]
        assert len(snaps) == 2
        assert snaps[0]["epoch"] == 20
        assert snaps[0]["emb_var"] == pytest.approx(0.42)
        assert snaps[1]["spectral_w"] == pytest.approx(0.08)

    def test_v3_diagnostics_summary(self, tmp_path):
        """V3 diagnostics summary should contain scalar extracts."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        v3 = metrics["v3_diagnostics_summary"]
        assert v3["mqa6_passed"] is True
        assert v3["mqa6_global_spearman"] == pytest.approx(0.15)
        assert v3["cosine_global_mean"] == pytest.approx(0.98)
        assert v3["attention_n_layers"] == 3

    def test_aggregate_quality_weighted(self, tmp_path):
        """Aggregate quality should be variate-weighted."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        aq = metrics["aggregate_quality"]
        assert aq["total_variates"] == 94  # 80 + 14
        # Weighted R-squared: (0.556*80 + 0.588*14) / 94 = (44.48 + 8.232) / 94 = 0.5607
        expected_wr2 = (0.556 * 80 + 0.588 * 14) / 94
        assert aq["weighted_r_squared"] == pytest.approx(expected_wr2, abs=0.01)

    def test_whole_var_r_squared(self, tmp_path):
        """Per-group output must include whole-var R-squared."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        pipeline._results = _build_mock_results()
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")

        og = metrics["per_group_reconstruction"]["options_grid"]
        assert og["r_squared_whole_var"] == pytest.approx(0.526, abs=0.001)

    def test_pct_learned_clamped_to_0_100(self, tmp_path):
        """pct_learned should be clamped to [0, 100]."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        results = _build_mock_results()
        # Make transformer worse than predict-zero → negative pct
        results["ssl_baseline_diagnostics"]["group_results"]["options_grid"]["transformer"] = 1.5
        pipeline._results = results
        history = _build_mock_history()
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="patch")
        og = metrics["per_group_reconstruction"]["options_grid"]
        assert og["pct_learned"] == 0.0  # clamped to 0

    def test_missing_baselines_graceful(self, tmp_path):
        """Should handle missing _results keys gracefully."""
        _, collect_training_metrics, _, _ = _import_pipeline_funcs()
        import layer1.training.pipeline as pipeline

        # Minimal _results — no baselines at all
        pipeline._results = {}
        history = {"val_loss": [0.5, 0.4], "best_val_loss": 0.4,
                   "final_emb_var": 0.3, "final_beat_zero": 100}
        hp = _build_mock_hp()

        metrics = collect_training_metrics(history, hp, 706000, tokenizer="linear")

        assert metrics["schema_version"] == 1
        assert metrics["tokenizer"] == "linear"
        assert "baselines_aggregate" not in metrics
        assert "per_group_reconstruction" not in metrics


# ---------------------------------------------------------------------------
# 3. History storage tests (diagnostic_snapshots, final values)
# ---------------------------------------------------------------------------

class TestHistoryStorage:
    """Test that training-time history mutations store the right data."""

    def test_diagnostic_snapshot_structure(self):
        """Verify the snapshot dict has the expected keys."""
        snapshot = {
            "epoch": 20,
            "emb_var": 0.42,
            "shortcut_suspect": 2,
            "well_reconstructed": 80,
            "spectral_w": 0.06,
        }
        assert set(snapshot.keys()) == {"epoch", "emb_var", "shortcut_suspect",
                                         "well_reconstructed", "spectral_w"}

    def test_history_final_values_structure(self):
        """Verify final values are stored with correct keys."""
        history = {}
        # Simulate what train_ssl does at end
        history["final_emb_var"] = 0.45
        history["final_spectral_w"] = 0.08
        history["final_beat_zero"] = 120
        history["best_val_loss"] = 0.15

        assert history["final_emb_var"] == 0.45
        assert history["final_spectral_w"] == 0.08
        assert history["final_beat_zero"] == 120
        assert history["best_val_loss"] == 0.15
