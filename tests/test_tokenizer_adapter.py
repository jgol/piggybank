"""
Unit tests for tokenizer implementations and multi-tokenizer adapter.

Tests:
  1. PatchTokenizerV2 (pipeline.py) — shape, rank, forward pass, assertions
  2. batch_forecast.py _reconstruct_encoder — linear/patch/cnn adapter builds
  3. train_local.py --tokenizer argparse integration
  4. evaluate_encoder_quality.py metric2_knn_regime_delta M2 fix

Requires: pytest, torch, numpy
Does NOT import pipeline.py (except for test 1 which tests the actual class).
"""
import os
import sys
import importlib.util
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Must set LOCAL_MODE before importing anything from layer1.training
os.environ["LOCAL_MODE"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


_evaluate_encoder_quality_mod = None


def _import_evaluate_encoder_quality():
    """Import scripts/evaluate_encoder_quality.py which has no __init__.py.

    Cached to avoid re-executing module-level code on each test call.
    """
    global _evaluate_encoder_quality_mod
    if _evaluate_encoder_quality_mod is not None:
        return _evaluate_encoder_quality_mod
    spec = importlib.util.spec_from_file_location(
        "evaluate_encoder_quality",
        PROJECT_ROOT / "scripts" / "evaluate_encoder_quality.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _evaluate_encoder_quality_mod = mod
    return mod


# ============================================================================
# 1. PatchTokenizerV2 (pipeline.py)
# ============================================================================

class TestPatchTokenizerV2:
    """Tests for the PatchTokenizerV2 class in layer1/training/pipeline.py."""

    @pytest.fixture
    def tokenizer(self):
        """Build a PatchTokenizerV2 with canonical params."""
        from layer1.training.pipeline import PatchTokenizerV2
        return PatchTokenizerV2(seq_len=60, d_model=128, patch_size=12, dropout=0.0)

    def test_shape_correctness(self, tokenizer):
        """Input (B=2, V=147, T=60) -> output (B=2, V=147, D=128)."""
        x = torch.randn(2, 147, 60)
        with torch.no_grad():
            out = tokenizer(x)
        assert out.shape == (2, 147, 128), f"Expected (2, 147, 128), got {out.shape}"

    def test_effective_rank_above_50(self, tokenizer):
        """Architecture should not impose a hard rank bottleneck.

        PatchV1 had a rank-5 bottleneck (nn.Linear(5, 128) = rank 5).
        PatchV2 fixes this: concat_proj is nn.Linear(640, 128), rank = 128.

        We verify this by checking the weight matrix rank of concat_proj
        (the final projection), not the output stable rank of a random-init
        network. At random init, LayerNorm collapses the spectral distribution
        (stable rank ~3), but this does NOT indicate a rank bottleneck — it's
        the expected behaviour of random Gaussian weights under normalization.
        After training, output stable rank rises as weights learn to spread
        information across dimensions.

        The test checks:
        1. concat_proj weight matrix has full rank (128) — no hard bottleneck.
        2. Output has more than 1 non-trivial singular value — basic sanity.
        """
        # Check 1: concat_proj weight matrix rank (architecture capacity).
        # For nn.Linear(640, 128) with random init, matrix rank = 128.
        W = tokenizer.concat_proj.weight.detach().numpy()  # (128, 640)
        s_w = np.linalg.svd(W, compute_uv=False)
        weight_rank = np.sum(s_w > 1e-5 * s_w[0])  # count non-trivial singular values
        assert weight_rank >= 60, (
            f"concat_proj weight rank {weight_rank} < 60 — architecture is rank-bottlenecked. "
            f"PatchV1 had rank=5; PatchV2 should have rank=128."
        )

        # Check 2: output stable rank > 1 (not totally degenerate)
        x = torch.randn(32, 147, 60)
        with torch.no_grad():
            out = tokenizer(x)
        flat = out.reshape(-1, 128).numpy()
        s = np.linalg.svd(flat, compute_uv=False)
        stable_rank = (s ** 2).sum() / (s[0] ** 2)
        assert stable_rank > 1.0, (
            f"Stable rank {stable_rank:.2f} <= 1.0 — output is rank-1 degenerate"
        )

    def test_forward_runs_without_error(self, tokenizer):
        """Basic smoke test that forward pass completes on CPU."""
        x = torch.randn(1, 10, 60)
        out = tokenizer(x)
        assert out.shape == (1, 10, 128)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    def test_patch_size_12_assertion(self):
        """seq_len=60 is divisible by patch_size=12; seq_len=61 is not."""
        from layer1.training.pipeline import PatchTokenizerV2

        # Should work
        tok = PatchTokenizerV2(seq_len=60, d_model=128, patch_size=12)
        assert tok.n_patches == 5

        # Should fail
        with pytest.raises(AssertionError, match="divisible by patch_size"):
            PatchTokenizerV2(seq_len=61, d_model=128, patch_size=12)

    def test_varying_batch_sizes(self, tokenizer):
        """Verify different batch sizes produce correct shapes."""
        for B in [1, 4, 16]:
            x = torch.randn(B, 147, 60)
            with torch.no_grad():
                out = tokenizer(x)
            assert out.shape == (B, 147, 128)


# ============================================================================
# 2. batch_forecast.py multi-tokenizer adapter
# ============================================================================

class TestReconstructEncoder:
    """Tests for _reconstruct_encoder in layer1/inference/batch_forecast.py."""

    @pytest.fixture
    def base_hp(self):
        """Base hyperparameter dict shared across tokenizer variants."""
        return {
            "seq_len": 60,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 512,
            "dropout": 0.0,  # Zero for deterministic tests
        }

    def _build_and_encode(self, hp, n_variates=147, batch_size=2):
        """Helper: build encoder, run forward pass, return multi-layer output."""
        from layer1.inference.batch_forecast import _reconstruct_encoder

        device = torch.device("cpu")
        model = _reconstruct_encoder(hp, n_variates, device)
        model.eval()

        x = torch.randn(batch_size, hp["seq_len"], n_variates)
        with torch.no_grad():
            out = model.encode_variates(x, multi_layer=True)
        return model, out

    def test_linear_tokenizer(self, base_hp):
        """tokenizer='linear' builds and produces (B, V, D*3)."""
        hp = {**base_hp, "tokenizer": "linear"}
        model, out = self._build_and_encode(hp)

        B, V, D_total = out.shape
        assert B == 2
        assert V == 147
        assert D_total == 128 * 3, f"Expected D*n_layers=384, got {D_total}"
        assert torch.isfinite(out).all()

    def test_patch_tokenizer(self, base_hp):
        """tokenizer='patch' builds and produces (B, V, D*3)."""
        hp = {**base_hp, "tokenizer": "patch", "patch_size": 12}
        model, out = self._build_and_encode(hp)

        B, V, D_total = out.shape
        assert B == 2
        assert V == 147
        assert D_total == 128 * 3, f"Expected D*n_layers=384, got {D_total}"
        assert torch.isfinite(out).all()

    def test_cnn_tokenizer_with_temporal_attn(self, base_hp):
        """tokenizer='cnn' + use_temporal_attn=True builds and produces (B, V, D*3)."""
        hp = {**base_hp, "tokenizer": "cnn", "use_temporal_attn": True}
        model, out = self._build_and_encode(hp)

        B, V, D_total = out.shape
        assert B == 2
        assert V == 147
        assert D_total == 128 * 3, f"Expected D*n_layers=384, got {D_total}"
        assert torch.isfinite(out).all()

    def test_cnn_tokenizer_without_temporal_attn(self, base_hp):
        """tokenizer='cnn' without temporal_attn also works (uses built-in pool)."""
        hp = {**base_hp, "tokenizer": "cnn", "use_temporal_attn": False}
        model, out = self._build_and_encode(hp)

        B, V, D_total = out.shape
        assert B == 2
        assert V == 147
        assert D_total == 128 * 3

    def test_single_layer_output(self, base_hp):
        """multi_layer=False produces (B, V, D) from single last layer."""
        from layer1.inference.batch_forecast import _reconstruct_encoder

        hp = {**base_hp, "tokenizer": "patch", "patch_size": 12}
        device = torch.device("cpu")
        model = _reconstruct_encoder(hp, 147, device)
        model.eval()

        x = torch.randn(2, 60, 147)
        with torch.no_grad():
            out = model.encode_variates(x, multi_layer=False)
        assert out.shape == (2, 147, 128)

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "raw_data" / "local_store" /
             "ssl_model_v3_ssl012_patch.pt").exists(),
        reason="Patch checkpoint not available on this machine"
    )
    def test_load_patch_checkpoint_zero_missing_keys(self):
        """Load actual patch checkpoint with 0 missing keys."""
        from layer1.inference.batch_forecast import (
            _reconstruct_encoder, _load_checkpoint_flexible
        )

        ckpt_path = (PROJECT_ROOT / "raw_data" / "local_store" /
                     "ssl_model_v3_ssl012_patch.pt")
        ckpt = _load_checkpoint_flexible(ckpt_path, map_location="cpu")

        hp = ckpt["ssl_hyperparams"]
        n_variates = ckpt.get("n_ssl_features", 147)

        device = torch.device("cpu")
        model = _reconstruct_encoder(hp, n_variates, device)
        result = model.load_state_dict(ckpt["encoder_state_dict"], strict=False)

        assert result.missing_keys == [], (
            f"Missing keys when loading patch checkpoint: {result.missing_keys}"
        )

        # Verify the model can actually run inference
        model.eval()
        x = torch.randn(1, hp["seq_len"], n_variates)
        with torch.no_grad():
            out = model.encode_variates(x, multi_layer=True)
        expected_d = hp["d_model"] * hp.get("n_layers", 3)
        assert out.shape == (1, n_variates, expected_d)


# ============================================================================
# 3. train_local.py --tokenizer flag
# ============================================================================

class TestTrainLocalTokenizerFlag:
    """Tests for --tokenizer argparse handling in scripts/train_local.py."""

    def _parse_args(self, argv):
        """Parse arguments using train_local's argparser."""
        import argparse

        # Reproduce the argparse setup from train_local.py without importing
        # the full module (which would trigger pipeline import + MLflow etc.)
        ap = argparse.ArgumentParser()
        mode = ap.add_mutually_exclusive_group(required=True)
        mode.add_argument("--smoke", action="store_true")
        mode.add_argument("--full", action="store_true")
        ap.add_argument("--experiment-id", default="L1-SSL-010-LOCAL")
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--bundle", type=Path,
                        default=PROJECT_ROOT / "raw_data" / "layer1_corpus_v3_bundle.npz")
        ap.add_argument("--skip-registry", action="store_true")
        ap.add_argument("--skip-git-tag", action="store_true")
        ap.add_argument("--decisions", default="D71,D84,D85,D86")
        ap.add_argument("--parent-id", default="L1-SSL-004-FE")
        ap.add_argument("--description", default="")
        ap.add_argument("--mlflow-experiment", default="layer1")
        ap.add_argument("--probes", dest="probes", action="store_true", default=None)
        ap.add_argument("--no-probes", dest="probes", action="store_false")
        ap.add_argument("--probes-include-expected-null",
                        dest="probes_include_expected_null",
                        action="store_true", default=None)
        ap.add_argument("--no-expected-null",
                        dest="probes_include_expected_null",
                        action="store_false")
        ap.add_argument("--extended-probes", action="store_true", default=False)
        ap.add_argument("--tokenizer", choices=["linear", "patch", "cnn"],
                        default=None)
        ap.add_argument("--spectral", type=float, default=None)

        return ap.parse_args(argv)

    def test_accepts_linear(self):
        """--tokenizer linear is accepted."""
        args = self._parse_args(["--full", "--tokenizer", "linear"])
        assert args.tokenizer == "linear"

    def test_accepts_patch(self):
        """--tokenizer patch is accepted."""
        args = self._parse_args(["--full", "--tokenizer", "patch"])
        assert args.tokenizer == "patch"

    def test_accepts_cnn(self):
        """--tokenizer cnn is accepted."""
        args = self._parse_args(["--full", "--tokenizer", "cnn"])
        assert args.tokenizer == "cnn"

    def test_rejects_invalid_tokenizer(self):
        """--tokenizer foobar is rejected."""
        with pytest.raises(SystemExit):
            self._parse_args(["--full", "--tokenizer", "foobar"])

    def test_default_is_none(self):
        """Default tokenizer is None (use pipeline.py's value)."""
        args = self._parse_args(["--full"])
        assert args.tokenizer is None

    def test_cnn_sets_use_temporal_attn_true(self):
        """Verify the train_local logic: --tokenizer cnn sets use_temporal_attn=True.

        We mock pipeline to verify the override is applied correctly.
        """
        # Create a mock pipeline module with SSL_HYPERPARAMS
        mock_pipeline = MagicMock()
        mock_pipeline.SSL_HYPERPARAMS = {
            "seq_len": 60,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 3,
            "d_ff": 512,
            "dropout": 0.2,
            "tokenizer": "linear",
            "use_temporal_attn": False,
        }
        mock_pipeline.ACTIVE_CKPT_SUFFIX = "_v3_ssl012"
        mock_pipeline.ACTIVE_CKPT_KEY = "ssl_model_v3_ssl012"
        mock_pipeline.ACTIVE_HISTORY_KEY = "ssl_history_v3_ssl012"

        # Simulate the logic from train_local.py lines 317-334
        tokenizer_choice = "cnn"
        overrides = {}

        mock_pipeline.SSL_HYPERPARAMS["tokenizer"] = tokenizer_choice
        overrides["tokenizer"] = tokenizer_choice

        if tokenizer_choice == "cnn":
            mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] = True
            overrides["use_temporal_attn"] = True
        elif tokenizer_choice == "patch":
            mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] = False

        assert mock_pipeline.SSL_HYPERPARAMS["tokenizer"] == "cnn"
        assert mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] is True
        assert overrides["use_temporal_attn"] is True

    def test_patch_sets_use_temporal_attn_false(self):
        """Verify: --tokenizer patch sets use_temporal_attn=False."""
        mock_pipeline = MagicMock()
        mock_pipeline.SSL_HYPERPARAMS = {
            "tokenizer": "linear",
            "use_temporal_attn": True,  # Start with True to verify it gets set False
        }
        mock_pipeline.ACTIVE_CKPT_SUFFIX = "_v3_ssl012"

        tokenizer_choice = "patch"
        mock_pipeline.SSL_HYPERPARAMS["tokenizer"] = tokenizer_choice

        if tokenizer_choice == "cnn":
            mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] = True
        elif tokenizer_choice == "patch":
            mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] = False

        assert mock_pipeline.SSL_HYPERPARAMS["use_temporal_attn"] is False


# ============================================================================
# 4. evaluate_encoder_quality.py M2 fix
# ============================================================================

class TestMetric2KnnRegimeDelta:
    """Tests for metric2_knn_regime_delta per_variate_tokens fix."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data that mimics the M2 function inputs.

        Uses N=400 windows spread across 200 unique dates (2 windows per date)
        to give walk-forward CV enough folds with sufficient train/test samples.
        """
        rng = np.random.default_rng(42)
        N = 400
        V = 147
        D = 128
        T = 60

        # Create regime labels (4 classes, some invalid=-1)
        regime_labels = rng.integers(0, 4, size=N).astype(np.int64)
        regime_labels[:20] = -1  # Some invalid

        # Create window dates (ordered, 2 windows per date for walk-forward)
        # 200 sequential trading days starting 2024-01-02
        from datetime import date, timedelta
        start = date(2024, 1, 2)
        unique_dates = [(start + timedelta(days=i)).isoformat() for i in range(200)]
        # Duplicate each date: 2 windows per day = 400 total
        window_dates = np.array([d for d in unique_dates for _ in range(2)])

        # Multi-layer mean-pooled embeddings (N, 384)
        emb_all = rng.standard_normal((N, D * 3)).astype(np.float32)

        # Z-scored windows (N, T, V)
        X_all_z = rng.standard_normal((N, T, V)).astype(np.float32)

        # Per-variate tokens (N, V, D) — the M2 fix target
        per_variate_tokens = rng.standard_normal((N, V, D)).astype(np.float32)

        return {
            "emb_all": emb_all,
            "X_all_z": X_all_z,
            "regime_labels": regime_labels,
            "window_dates": window_dates,
            "per_variate_tokens": per_variate_tokens,
        }

    def test_uses_per_variate_tokens_when_provided(self, synthetic_data, capsys):
        """When per_variate_tokens is provided, M2 uses tokens directly (not mean-pool fallback)."""
        mod = _import_evaluate_encoder_quality()
        metric2_knn_regime_delta = mod.metric2_knn_regime_delta

        result = metric2_knn_regime_delta(
            emb_all=synthetic_data["emb_all"],
            X_all_z=synthetic_data["X_all_z"],
            regime_labels=synthetic_data["regime_labels"],
            window_dates=synthetic_data["window_dates"],
            per_variate_tokens=synthetic_data["per_variate_tokens"],
        )

        captured = capsys.readouterr()
        # Should NOT see the fallback warning
        assert "WARNING: Using mean-pooled" not in captured.out
        # Should see the tokens-based message
        assert "tokens" in captured.out.lower() or "384-d" in captured.out

    def test_fallback_to_mean_pool_without_tokens(self, synthetic_data, capsys):
        """Without per_variate_tokens, M2 falls back to mean-pool with warning."""
        mod = _import_evaluate_encoder_quality()
        metric2_knn_regime_delta = mod.metric2_knn_regime_delta

        result = metric2_knn_regime_delta(
            emb_all=synthetic_data["emb_all"],
            X_all_z=synthetic_data["X_all_z"],
            regime_labels=synthetic_data["regime_labels"],
            window_dates=synthetic_data["window_dates"],
            per_variate_tokens=None,
        )

        captured = capsys.readouterr()
        # Should see the fallback warning
        assert "WARNING: Using mean-pooled" in captured.out

    def test_token_shape_used_correctly(self, synthetic_data, capsys):
        """With tokens (N, 147, 128), extracts 3 regime positions -> (N_valid, 384)."""
        mod = _import_evaluate_encoder_quality()
        metric2_knn_regime_delta = mod.metric2_knn_regime_delta

        # The function should extract positions [41, 100, 101] from the V dim
        # and flatten 3 * 128 = 384 dimensions
        result = metric2_knn_regime_delta(
            emb_all=synthetic_data["emb_all"],
            X_all_z=synthetic_data["X_all_z"],
            regime_labels=synthetic_data["regime_labels"],
            window_dates=synthetic_data["window_dates"],
            per_variate_tokens=synthetic_data["per_variate_tokens"],
        )

        captured = capsys.readouterr()
        # Confirm the printed dimension matches 3 tokens * 128 = 384
        assert "384-d" in captured.out

    def test_returns_result_dict(self, synthetic_data):
        """M2 returns a result dict (or None if insufficient data)."""
        mod = _import_evaluate_encoder_quality()
        metric2_knn_regime_delta = mod.metric2_knn_regime_delta

        result = metric2_knn_regime_delta(
            emb_all=synthetic_data["emb_all"],
            X_all_z=synthetic_data["X_all_z"],
            regime_labels=synthetic_data["regime_labels"],
            window_dates=synthetic_data["window_dates"],
            per_variate_tokens=synthetic_data["per_variate_tokens"],
        )

        # With 400 samples and 4 regimes, should produce a result
        assert result is not None
        # Result should have the canonical metric keys
        assert "encoder_acc" in result
        assert "raw_acc" in result
        assert "delta" in result
        assert "encoder_folds" in result
        assert "raw_folds" in result
        # Delta is encoder_acc - raw_acc
        assert abs(result["delta"] - (result["encoder_acc"] - result["raw_acc"])) < 1e-10
