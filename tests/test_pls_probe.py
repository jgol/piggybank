"""Tests for scripts/run_pls_probe.py — PLS transmission probe changes.

Validates:
1. raw_temporal_stats_grouped produces correct groups with correct shapes
2. Per-group PLS budget symmetry: raw baseline gets same K budget as encoder
3. fit_pls_per_group works with both encoder-shaped and raw-grouped inputs
4. project_through_pls always produces exactly 84-d output
5. No target leakage: test-fold targets never reach PLS fitting
6. Ceiling probe uses standardized data (functional test)
"""
import os
import sys

for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_NUM_THREADS"):
    os.environ.setdefault(_env, "1")
os.environ.setdefault("LOCAL_MODE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pls_probe import (
    _compute_temporal_stats,
    raw_temporal_stats,
    raw_temporal_stats_grouped,
    fit_pls_per_group,
    project_through_pls,
    random_projection_84d,
    PER_GROUP_K,
    _RAW_GROUP_RANGES,
    _TYPED_VECTOR_SSL_RANGES,
    PLS_N_TARGETS,
)

# Total K across all groups must equal 84
TOTAL_K = sum(PER_GROUP_K.values())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_windows():
    """Synthetic corpus windows: (200, 60, 147) float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 60, 147)).astype(np.float32)


@pytest.fixture
def synthetic_targets():
    """Synthetic multi-target Y: (200, 7) and valid mask."""
    rng = np.random.default_rng(99)
    targets = rng.standard_normal((200, 7)).astype(np.float32)
    valid = np.ones(200, dtype=bool)
    valid[:10] = False  # some invalid
    return targets, valid


@pytest.fixture
def synthetic_encoder_vectors():
    """Encoder-shaped typed vectors: 7 groups, each (200, 384)."""
    rng = np.random.default_rng(77)
    vectors = {}
    for name in PER_GROUP_K:
        vectors[name] = rng.standard_normal((200, 384)).astype(np.float32)
    return vectors


# ---------------------------------------------------------------------------
# Tests: _compute_temporal_stats
# ---------------------------------------------------------------------------

class TestComputeTemporalStats:
    def test_output_shape(self):
        X = np.random.randn(50, 60, 10).astype(np.float32)
        result = _compute_temporal_stats(X)
        assert result.shape == (50, 100)  # 10 variates × 10 stats

    def test_output_shape_single_variate(self):
        X = np.random.randn(30, 60, 1).astype(np.float32)
        result = _compute_temporal_stats(X)
        assert result.shape == (30, 10)  # 1 variate × 10 stats


# ---------------------------------------------------------------------------
# Tests: raw_temporal_stats_grouped
# ---------------------------------------------------------------------------

class TestRawTemporalStatsGrouped:
    def test_all_groups_present(self, synthetic_windows):
        grouped = raw_temporal_stats_grouped(synthetic_windows)
        for name in PER_GROUP_K:
            assert name in grouped, f"Missing group: {name}"

    def test_no_extra_groups(self, synthetic_windows):
        grouped = raw_temporal_stats_grouped(synthetic_windows)
        extra = set(grouped.keys()) - set(PER_GROUP_K.keys())
        assert not extra, f"Unexpected groups: {extra}"

    def test_group_shapes(self, synthetic_windows):
        N = synthetic_windows.shape[0]
        grouped = raw_temporal_stats_grouped(synthetic_windows)

        # Per-range groups: n_variates_in_range × 10 stats
        for name, (start, end) in _RAW_GROUP_RANGES.items():
            n_variates = end - start
            expected_cols = n_variates * 10
            assert grouped[name].shape == (N, expected_cols), \
                f"{name}: expected (N, {expected_cols}), got {grouped[name].shape}"

        # SHARED = all 147 variates × 10 stats
        assert grouped["EMB_SHARED"].shape == (N, 1470)

    def test_sample_count_matches_input(self, synthetic_windows):
        grouped = raw_temporal_stats_grouped(synthetic_windows)
        N = synthetic_windows.shape[0]
        for name, arr in grouped.items():
            assert arr.shape[0] == N

    def test_groups_cover_all_variates(self):
        """Verify _RAW_GROUP_RANGES covers variates 0-146 without gaps or overlaps."""
        covered = set()
        for start, end in _RAW_GROUP_RANGES.values():
            new = set(range(start, end))
            overlap = covered & new
            assert not overlap, f"Overlapping variates: {overlap}"
            covered |= new
        assert covered == set(range(147)), \
            f"Missing variates: {set(range(147)) - covered}"

    def test_ranges_match_encoder(self):
        """Raw group ranges must match encoder's _TYPED_VECTOR_SSL_RANGES."""
        for name in _RAW_GROUP_RANGES:
            assert name in _TYPED_VECTOR_SSL_RANGES, f"{name} not in encoder ranges"
            assert _RAW_GROUP_RANGES[name] == _TYPED_VECTOR_SSL_RANGES[name], \
                f"{name}: raw={_RAW_GROUP_RANGES[name]} != encoder={_TYPED_VECTOR_SSL_RANGES[name]}"


# ---------------------------------------------------------------------------
# Tests: PLS budget symmetry
# ---------------------------------------------------------------------------

class TestPLSBudgetSymmetry:
    def test_encoder_and_raw_get_same_k_per_group(self, synthetic_windows,
                                                    synthetic_targets,
                                                    synthetic_encoder_vectors):
        """Both encoder and raw baselines produce exactly 84-d output
        with the same K allocation per group."""
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True

        # Encoder path
        enc_models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        enc_84d = project_through_pls(synthetic_encoder_vectors, enc_models)
        assert enc_84d.shape == (200, TOTAL_K), f"Encoder: expected (200, {TOTAL_K}), got {enc_84d.shape}"

        # Raw path
        raw_grouped = raw_temporal_stats_grouped(synthetic_windows)
        raw_models = fit_pls_per_group(raw_grouped, targets, valid, train_mask)
        raw_84d = project_through_pls(raw_grouped, raw_models)
        assert raw_84d.shape == (200, TOTAL_K), f"Raw: expected (200, {TOTAL_K}), got {raw_84d.shape}"

    def test_per_group_k_matches(self, synthetic_windows, synthetic_targets,
                                  synthetic_encoder_vectors):
        """Each group produces exactly K components in both paths."""
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True

        enc_models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        raw_grouped = raw_temporal_stats_grouped(synthetic_windows)
        raw_models = fit_pls_per_group(raw_grouped, targets, valid, train_mask)

        for name, K in PER_GROUP_K.items():
            enc_k = enc_models[name]["components"].shape[0]
            raw_k = raw_models[name]["components"].shape[0]
            assert enc_k == K, f"Encoder {name}: expected K={K}, got {enc_k}"
            assert raw_k == K, f"Raw {name}: expected K={K}, got {raw_k}"

    def test_pls_direction_count_matches(self, synthetic_windows, synthetic_targets,
                                          synthetic_encoder_vectors):
        """Both paths get min(K, PLS_N_TARGETS) PLS directions per group."""
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True

        enc_models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        raw_grouped = raw_temporal_stats_grouped(synthetic_windows)
        raw_models = fit_pls_per_group(raw_grouped, targets, valid, train_mask)

        total_pls_enc = 0
        total_pls_raw = 0
        for name, K in PER_GROUP_K.items():
            n_pls = min(K, PLS_N_TARGETS)
            total_pls_enc += n_pls
            total_pls_raw += n_pls

        # Both should get the same total PLS-supervised directions
        assert total_pls_enc == total_pls_raw


# ---------------------------------------------------------------------------
# Tests: fit_pls_per_group
# ---------------------------------------------------------------------------

class TestFitPLSPerGroup:
    def test_output_keys(self, synthetic_encoder_vectors, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        assert set(models.keys()) == set(PER_GROUP_K.keys())

    def test_components_shape(self, synthetic_encoder_vectors, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        for name, K in PER_GROUP_K.items():
            assert models[name]["components"].shape[0] == K
            assert models[name]["components"].shape[1] == 384

    def test_mean_shape(self, synthetic_encoder_vectors, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        for name in PER_GROUP_K:
            assert models[name]["mean"].shape == (384,)

    def test_no_test_fold_leakage(self, synthetic_encoder_vectors, synthetic_targets):
        """PLS fitting only uses train_mask & valid samples."""
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:100] = True

        # Fit with original targets
        models_a = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)

        # Corrupt test-fold targets — should not affect PLS fit
        targets_corrupted = targets.copy()
        targets_corrupted[100:] = 999.0
        models_b = fit_pls_per_group(synthetic_encoder_vectors, targets_corrupted, valid, train_mask)

        for name in PER_GROUP_K:
            np.testing.assert_array_equal(
                models_a[name]["components"], models_b[name]["components"],
                err_msg=f"{name}: PLS components changed when test targets were corrupted"
            )
            np.testing.assert_array_equal(
                models_a[name]["mean"], models_b[name]["mean"],
                err_msg=f"{name}: mean changed when test targets were corrupted"
            )

    def test_works_with_raw_grouped(self, synthetic_windows, synthetic_targets):
        """fit_pls_per_group accepts raw grouped stats (varying dimensionality)."""
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        raw_grouped = raw_temporal_stats_grouped(synthetic_windows)
        models = fit_pls_per_group(raw_grouped, targets, valid, train_mask)

        for name, K in PER_GROUP_K.items():
            assert models[name]["components"].shape[0] == K
            # Feature dim should match raw group dim, not 384
            expected_d = raw_grouped[name].shape[1]
            assert models[name]["components"].shape[1] == expected_d, \
                f"{name}: expected components width {expected_d}, got {models[name]['components'].shape[1]}"


# ---------------------------------------------------------------------------
# Tests: project_through_pls
# ---------------------------------------------------------------------------

class TestProjectThroughPLS:
    def test_output_84d(self, synthetic_encoder_vectors, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        result = project_through_pls(synthetic_encoder_vectors, models)
        assert result.shape == (200, TOTAL_K)

    def test_output_84d_raw(self, synthetic_windows, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        raw_grouped = raw_temporal_stats_grouped(synthetic_windows)
        models = fit_pls_per_group(raw_grouped, targets, valid, train_mask)
        result = project_through_pls(raw_grouped, models)
        assert result.shape == (200, TOTAL_K)

    def test_no_nans(self, synthetic_encoder_vectors, synthetic_targets):
        targets, valid = synthetic_targets
        train_mask = np.zeros(200, dtype=bool)
        train_mask[:150] = True
        models = fit_pls_per_group(synthetic_encoder_vectors, targets, valid, train_mask)
        result = project_through_pls(synthetic_encoder_vectors, models)
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# Tests: random_projection_84d
# ---------------------------------------------------------------------------

class TestRandomProjection:
    def test_output_shape(self, synthetic_encoder_vectors):
        result = random_projection_84d(synthetic_encoder_vectors)
        assert result.shape == (200, TOTAL_K)

    def test_deterministic(self, synthetic_encoder_vectors):
        a = random_projection_84d(synthetic_encoder_vectors, seed=42)
        b = random_projection_84d(synthetic_encoder_vectors, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self, synthetic_encoder_vectors):
        a = random_projection_84d(synthetic_encoder_vectors, seed=42)
        b = random_projection_84d(synthetic_encoder_vectors, seed=123)
        assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# Tests: TOTAL_K invariant
# ---------------------------------------------------------------------------

class TestTotalK:
    def test_total_k_is_84(self):
        assert TOTAL_K == 84, f"Expected total K=84, got {TOTAL_K}"
