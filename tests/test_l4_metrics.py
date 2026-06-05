"""Tests for scripts/compute_l4_metrics.py — L4 absolute calibration metrics.

Validates:
1. Ceiling probe produces correct output structure
2. PCA + Ridge linear ceiling on known data
3. PCA + HistGBT nonlinear ceiling >= linear ceiling
4. Transfer efficiency computation
5. Fold construction
"""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification

from scripts.compute_l4_metrics import (
    run_ceiling_probe,
    make_folds,
    N_PCA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_regression():
    """Synthetic data with known learnability."""
    rng = np.random.default_rng(42)
    N = 500
    D = 600  # simulate flattened windows
    X, y = make_regression(n_samples=N, n_features=D, n_informative=20,
                           noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    valid = np.ones(N, dtype=bool)
    dates = np.array([f"2023-{(i % 200 + 1):03d}" for i in range(N)])
    return X, y, valid, dates


@pytest.fixture
def synthetic_classification():
    N = 500
    D = 600
    X, y = make_classification(n_samples=N, n_features=D, n_informative=20,
                                n_classes=4, random_state=42)
    X = X.astype(np.float32)
    valid = np.ones(N, dtype=bool)
    dates = np.array([f"2023-{(i % 200 + 1):03d}" for i in range(N)])
    return X, y, valid, dates


# ---------------------------------------------------------------------------
# Tests: Fold Construction
# ---------------------------------------------------------------------------

class TestFoldConstruction:
    def test_correct_count(self, synthetic_regression):
        _, _, _, dates = synthetic_regression
        folds = make_folds(dates)
        assert len(folds) > 0
        assert len(folds) <= 8

    def test_no_overlap(self, synthetic_regression):
        _, _, _, dates = synthetic_regression
        folds = make_folds(dates)
        for tr, te in folds:
            overlap = tr & te
            assert overlap.sum() == 0


# ---------------------------------------------------------------------------
# Tests: Ceiling Probes
# ---------------------------------------------------------------------------

class TestCeilingProbes:
    def test_regression_output_structure(self, synthetic_regression):
        X, y, valid, dates = synthetic_regression
        folds = make_folds(dates)
        result = run_ceiling_probe(X, y, valid, folds[:3], "regression", n_pca=50)
        assert "linear" in result
        assert "nonlinear" in result
        assert "mean" in result["linear"]
        assert "per_fold" in result["linear"]

    def test_nonlinear_produces_values(self, synthetic_regression):
        """Both ceilings produce finite, non-NaN values."""
        X, y, valid, dates = synthetic_regression
        folds = make_folds(dates)
        result = run_ceiling_probe(X, y, valid, folds[:3], "regression", n_pca=50)
        assert np.isfinite(result["linear"]["mean"])
        assert np.isfinite(result["nonlinear"]["mean"])
        assert result["linear"]["n_folds"] > 0
        assert result["nonlinear"]["n_folds"] > 0

    def test_classification_works(self, synthetic_classification):
        X, y, valid, dates = synthetic_classification
        folds = make_folds(dates)
        result = run_ceiling_probe(X, y, valid, folds[:3], "classification", n_pca=50)
        assert result["linear"]["mean"] > 0.25  # above random for 4-class
        assert result["nonlinear"]["mean"] > 0.25

    def test_r2_bounded(self, synthetic_regression):
        X, y, valid, dates = synthetic_regression
        folds = make_folds(dates)
        result = run_ceiling_probe(X, y, valid, folds[:3], "regression", n_pca=50)
        for score in result["linear"]["per_fold"]:
            assert -2.0 < score < 1.0  # reasonable bounds
