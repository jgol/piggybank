"""Tests for fit_supervised_bases() PLS + Residual PCA projection.

Tests the core PLS logic using synthetic data: 384-d embeddings where the
first 10 dimensions correlate with a multi-target Y (7 targets matching
the production schema: spread_5, log_rv_15, log_rv_30, regime_oh×4), and
the remaining 374 dimensions are noise.

This validates:
1. Output shapes are correct
2. Combined PLS + residual PCA directions are orthogonal
3. Roundtrip: fit -> save -> load -> project produces consistent results
4. Groups with K <= 7 use all K as PLS directions (no residual PCA)
5. First PLS directions have higher target correlation than last residual PCs
"""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _make_synthetic_embeddings(n_samples=500, n_features=384, n_informative=10,
                               n_targets=7, noise_scale=0.3, seed=42):
    """Create synthetic embeddings where first n_informative dims predict targets.

    Returns:
        X: (n_samples, n_features) float64
        Y: (n_samples, n_targets) float64
        true_weights: (n_informative, n_targets) — ground-truth linear map
    """
    rng = np.random.default_rng(seed)

    # Informative signal in first n_informative dims
    X_signal = rng.standard_normal((n_samples, n_informative))

    # Random linear map from signal to targets
    true_weights = rng.standard_normal((n_informative, n_targets))
    Y = X_signal @ true_weights + noise_scale * rng.standard_normal((n_samples, n_targets))

    # Noise dims (independent of Y)
    X_noise = rng.standard_normal((n_samples, n_features - n_informative))

    X = np.hstack([X_signal, X_noise])
    return X, Y, true_weights


def _orthogonalize_and_get_combined(X, Y, n_pls, n_residual_pca):
    """Replicate the PLS + residual PCA logic from fit_supervised_bases.

    Returns:
        combined: (n_pls + n_residual_pca, 384) orthonormal-ish directions
        pls_ortho: (n_pls, 384) QR-orthogonalized PLS directions
    """
    from sklearn.decomposition import PCA

    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Fit PLS
    pls = PLSRegression(n_components=n_pls, scale=False, max_iter=500)
    pls.fit(X_centered, Y)

    # PLS x-rotations -> orthogonalize via QR
    pls_directions = pls.x_rotations_.T.copy()  # (n_pls, 384)
    Q, R = np.linalg.qr(pls_directions.T, mode="reduced")  # Q: (384, n_pls)
    pls_ortho = Q.T  # (n_pls, 384)

    if n_residual_pca > 0:
        # Project out PLS subspace
        projection = X_centered @ Q @ Q.T
        residual = X_centered - projection

        # PCA on residual
        pca = PCA(n_components=n_residual_pca, svd_solver="full", random_state=42)
        pca.fit(residual)
        resid_components = pca.components_

        combined = np.vstack([pls_ortho, resid_components])
    else:
        combined = pls_ortho

    return combined, pls_ortho


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPLSOutputShape:
    """PLS produces correct (K, 384) components per group."""

    def test_shape_k15(self):
        """K=15 > 7 PLS directions: expect 7 PLS + 8 residual PCA."""
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)
        n_pls = 7
        n_residual = 15 - n_pls  # 8

        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls, n_residual)
        assert combined.shape == (15, 384)

    def test_shape_k7(self):
        """K=7 == n_pls_directions: all 7 are PLS, no residual."""
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)
        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls=7, n_residual_pca=0)
        assert combined.shape == (7, 384)

    def test_shape_k3(self):
        """K=3 < 7: only 3 PLS directions, no residual PCA."""
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)
        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls=3, n_residual_pca=0)
        assert combined.shape == (3, 384)


class TestPLSDirectionsOrthogonal:
    """Combined PLS + residual PCA directions are mutually orthogonal."""

    def test_pls_only_orthogonal(self):
        """After QR, PLS directions should be orthonormal."""
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)
        _, pls_ortho = _orthogonalize_and_get_combined(X, Y, n_pls=7, n_residual_pca=0)

        # Inner product matrix should be identity
        gram = pls_ortho @ pls_ortho.T
        np.testing.assert_allclose(gram, np.eye(7), atol=1e-10)

    def test_combined_near_orthogonal(self):
        """PLS + residual PCA should be approximately orthogonal.

        Note: residual PCA directions are orthogonal to each other and to
        the PLS subspace by construction (we project out PLS first). However,
        PLS_ortho and resid_PCA may not be perfectly orthogonal due to finite
        sample effects. We allow atol=0.05 for cross-block terms.
        """
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)
        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls=7, n_residual_pca=8)

        gram = combined @ combined.T  # (15, 15)

        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(gram), np.ones(15), atol=1e-6)

        # Off-diagonal (cross-block PLS x Residual) should be small
        off_diag = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diag)) < 0.05, (
            f"Max off-diagonal dot product = {np.max(np.abs(off_diag)):.6f}, "
            f"expected < 0.05"
        )


class TestPLSRoundtripLoadProject:
    """fit_supervised_bases -> save_bases -> load_bases -> project works."""

    def test_roundtrip_with_tempdir(self):
        """Create fake Parquet + targets, fit, save, load, project."""
        import pandas as pd
        from layer2.inference.pca_bases import (
            fit_supervised_bases, save_bases, load_bases, project,
            EMB_COLS, _BASES_PATH, _MANIFEST_PATH
        )

        n_total = 200
        n_train = 150
        rng = np.random.default_rng(123)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Build fake Parquet with unique dates
            base = pd.Timestamp("2024-01-01")
            dates = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_total)]
            train_end = dates[n_train - 1]

            df_data = {"date": dates}
            for col in EMB_COLS:
                # Each row: list of 384 floats
                emb = rng.standard_normal((n_total, 384)).astype(np.float64)
                df_data[col] = [emb[i].tolist() for i in range(n_total)]

            df = pd.DataFrame(df_data)
            parquet_path = tmpdir / "test.parquet"
            df.to_parquet(parquet_path)

            # Build targets .npz
            targets = rng.standard_normal((n_total, 7)).astype(np.float32)
            valid = np.ones(n_total, dtype=bool)
            valid[-10:] = False  # last 10 invalid
            targets_path = tmpdir / "targets.npz"
            np.savez(targets_path, targets=targets, valid=valid)

            # Override paths
            out_npz = tmpdir / "pca_bases.npz"
            out_manifest = tmpdir / "pca_bases_manifest.json"

            # Fit with a per_group_k that is small for speed
            per_group_k = {col: 3 for col in EMB_COLS}

            bases = fit_supervised_bases(
                str(parquet_path),
                str(targets_path),
                train_end=train_end,
                require_n_train=n_train,
                per_group_k=per_group_k,
            )

            # Save
            sha, manifest_json = save_bases(bases, out_path=out_npz,
                                            manifest_path=out_manifest)
            assert out_npz.exists()
            assert out_manifest.exists()

            # Load
            loaded = load_bases(path=out_npz, manifest_path=out_manifest)

            # Project a single embedding
            test_emb = rng.standard_normal(384).astype(np.float32)
            for col in EMB_COLS:
                proj = project(test_emb, loaded[col])
                assert proj.shape == (3,), f"{col} projection shape mismatch"
                # Verify projection is finite
                assert np.all(np.isfinite(proj))


class TestPLSVixK3AllSupervised:
    """EMB_VIX with K=3 uses all 3 as PLS directions (no residual)."""

    def test_vix_k3_no_residual(self):
        """When K=3 <= n_pls_directions=7, all components are supervised."""
        X, Y, _ = _make_synthetic_embeddings(n_samples=500, n_features=384)

        # Simulate K=3 for VIX
        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls=3, n_residual_pca=0)

        assert combined.shape == (3, 384)

        # All 3 directions should have meaningful correlation with Y
        X_centered = X - X.mean(axis=0)
        projections = X_centered @ combined.T  # (500, 3)

        # Each PLS direction should correlate with at least one target
        for k in range(3):
            correlations = [
                abs(np.corrcoef(projections[:, k], Y[:, t])[0, 1])
                for t in range(Y.shape[1])
            ]
            max_corr = max(correlations)
            assert max_corr > 0.1, (
                f"PLS direction {k} has max target correlation {max_corr:.4f}, "
                f"expected > 0.1 for a supervised direction"
            )


class TestPLSDirectionsCorrelateWithTargets:
    """First PLS direction has higher correlation with targets than last residual PC."""

    def test_pls_vs_residual_correlation(self):
        """PLS directions should be more target-correlated than residual PCs."""
        X, Y, _ = _make_synthetic_embeddings(
            n_samples=500, n_features=384, n_informative=10, noise_scale=0.3
        )

        n_pls = 7
        n_residual = 8
        combined, _ = _orthogonalize_and_get_combined(X, Y, n_pls, n_residual)

        X_centered = X - X.mean(axis=0)
        projections = X_centered @ combined.T  # (500, 15)

        # Compute max |correlation| with any target for each direction
        def max_target_corr(proj_col):
            return max(
                abs(np.corrcoef(proj_col, Y[:, t])[0, 1])
                for t in range(Y.shape[1])
            )

        # First PLS direction (index 0) correlation
        pls_first_corr = max_target_corr(projections[:, 0])

        # Last residual PCA direction (index 14) correlation
        resid_last_corr = max_target_corr(projections[:, -1])

        # PLS first should dominate residual last
        assert pls_first_corr > resid_last_corr, (
            f"First PLS direction corr ({pls_first_corr:.4f}) should exceed "
            f"last residual PC corr ({resid_last_corr:.4f})"
        )

        # Stronger: mean PLS correlation should exceed mean residual correlation
        pls_mean_corr = np.mean([max_target_corr(projections[:, k]) for k in range(n_pls)])
        resid_mean_corr = np.mean([max_target_corr(projections[:, k]) for k in range(n_pls, 15)])

        assert pls_mean_corr > resid_mean_corr, (
            f"Mean PLS corr ({pls_mean_corr:.4f}) should exceed "
            f"mean residual corr ({resid_mean_corr:.4f})"
        )


class TestPLSDefensiveChecks:
    """Defensive checks in fit_supervised_bases raise on bad inputs."""

    def test_nan_embeddings_raises(self):
        """NaN in training embeddings should raise RuntimeError."""
        import pandas as pd
        from layer2.inference.pca_bases import fit_supervised_bases, EMB_COLS

        n_total = 200
        n_train = 150
        rng = np.random.default_rng(999)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Use unique dates so train/test split is unambiguous
            base = pd.Timestamp("2024-01-01")
            dates = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_total)]
            train_end = dates[n_train - 1]

            df_data = {"date": dates}
            for col in EMB_COLS:
                emb = rng.standard_normal((n_total, 384)).astype(np.float64)
                # Inject NaN into the first group's training data
                if col == "EMB_SHARED":
                    emb[10, 50] = np.nan
                df_data[col] = [emb[i].tolist() for i in range(n_total)]

            df = pd.DataFrame(df_data)
            parquet_path = tmpdir / "test.parquet"
            df.to_parquet(parquet_path)

            targets = rng.standard_normal((n_total, 7)).astype(np.float32)
            valid = np.ones(n_total, dtype=bool)
            targets_path = tmpdir / "targets.npz"
            np.savez(targets_path, targets=targets, valid=valid)

            per_group_k = {col: 3 for col in EMB_COLS}

            with pytest.raises(RuntimeError, match="NaN detected"):
                fit_supervised_bases(
                    str(parquet_path),
                    str(targets_path),
                    train_end=train_end,
                    require_n_train=n_train,
                    per_group_k=per_group_k,
                )

    def test_row_count_mismatch_raises(self):
        """Targets with wrong row count should raise RuntimeError."""
        import pandas as pd
        from layer2.inference.pca_bases import fit_supervised_bases, EMB_COLS

        n_total = 100
        n_train = 80
        rng = np.random.default_rng(888)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            base = pd.Timestamp("2024-01-01")
            dates = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_total)]
            train_end = dates[n_train - 1]

            df_data = {"date": dates}
            for col in EMB_COLS:
                emb = rng.standard_normal((n_total, 384)).astype(np.float64)
                df_data[col] = [emb[i].tolist() for i in range(n_total)]

            df = pd.DataFrame(df_data)
            parquet_path = tmpdir / "test.parquet"
            df.to_parquet(parquet_path)

            # Wrong number of rows in targets
            targets = rng.standard_normal((n_total + 5, 7)).astype(np.float32)
            valid = np.ones(n_total + 5, dtype=bool)
            targets_path = tmpdir / "targets.npz"
            np.savez(targets_path, targets=targets, valid=valid)

            per_group_k = {col: 3 for col in EMB_COLS}

            with pytest.raises(RuntimeError, match="rows but Parquet has"):
                fit_supervised_bases(
                    str(parquet_path),
                    str(targets_path),
                    train_end=train_end,
                    require_n_train=n_train,
                    per_group_k=per_group_k,
                )
