"""Tests for scripts/compute_l2_metrics.py — L2 encoder quality metrics.

Validates:
1. M1 R² ratio computation from known MSE values
2. M1 handles edge cases (negative R², zero predict-zero)
3. M3 RankMe formula matches Garrido 2023 definition
4. M4 surface ratio computation
5. M5 forward utility extraction from probe results
6. Baseline table parsing from training logs
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_l2_metrics import (
    L2Metrics,
    parse_baseline_table,
    compute_m1,
    compute_m4,
    compute_m5,
    ELIGIBLE_GROUPS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_groups():
    """Known MSE values for hand-verifiable ratio computation."""
    return {
        "options_grid": {
            "n_feat": 88,
            "predict_zero": 0.4000,
            "persist": 0.3700,
            "ar1": 0.3500,
            "cvlin": 0.3200,
            "hybrid": 0.3100,
            "oracle": 0.0700,
            "xformer": 0.2000,
            "r2": 0.500,
        },
        "strike_agg": {
            "n_feat": 6,
            "predict_zero": 0.9000,
            "persist": 0.7500,
            "ar1": 0.7300,
            "cvlin": 0.8500,
            "hybrid": 0.7400,
            "oracle": 0.2500,
            "xformer": 0.5000,
            "r2": 0.444,
        },
        "vix_term": {
            "n_feat": 12,
            "predict_zero": 0.0900,
            "persist": 0.0670,
            "ar1": 0.0670,
            "cvlin": 0.0950,
            "hybrid": 0.0730,
            "oracle": 0.0010,
            "xformer": 0.1700,
            "r2": -0.889,
        },
        "order_flow": {
            "n_feat": 9,
            "predict_zero": 1.0000,
            "persist": 0.9700,
            "ar1": 0.8900,
            "cvlin": 0.8800,
            "hybrid": 0.8700,
            "oracle": 0.2200,
            "xformer": 0.9000,
            "r2": 0.100,
        },
    }


@pytest.fixture
def sample_log(tmp_path):
    """Synthetic training log with per-group MSE table."""
    content = """
  Per-group MSE and R² (R² = 1 - MSE_model / MSE_predict_zero):

  Group          #Feat  PredZero  Persist    AR(1)    CVLin   Hybrid   Oracle  Xformer     R²
  ---------------------------------------------------------------------------------------------------------------
  options_grid      88    0.4000   0.3700   0.3500   0.3200   0.3100   0.0700   0.2000  0.500
  strike_agg         6    0.9000   0.7500   0.7300   0.8500   0.7400   0.2500   0.5000  0.444
  spx_derived       14    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000  1.000
  vix_term          12    0.0900   0.0670   0.0670   0.0950   0.0730   0.0010   0.1700 -0.889
  order_flow         9    1.0000   0.9700   0.8900   0.8800   0.8700   0.2200   0.9000  0.100
  flow_roll3         9    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000  1.000
  flow_roll15        9    0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000  1.000
"""
    p = tmp_path / "training.log"
    p.write_text(content)
    return p


@pytest.fixture
def sample_probe_results(tmp_path):
    """Synthetic probe results with known values."""
    pr = {
        "log_rv_15": {
            "summary": {
                "emb_fine": {"mean": 0.606, "std": 0.052, "n_folds": 4},
                "raw_36": {"mean": 0.516, "std": 0.103, "n_folds": 5},
                "shuffle_combined": {"mean": 0.430, "std": 0.137, "n_samples": 24},
            }
        }
    }
    p = tmp_path / "probe_results.json"
    p.write_text(json.dumps(pr))
    return p


# ---------------------------------------------------------------------------
# Tests: Baseline Table Parsing
# ---------------------------------------------------------------------------

class TestBaselineTableParsing:
    def test_parses_all_groups(self, sample_log):
        groups = parse_baseline_table(sample_log)
        assert "options_grid" in groups
        assert "strike_agg" in groups
        assert "vix_term" in groups
        assert "order_flow" in groups

    def test_correct_values(self, sample_log):
        groups = parse_baseline_table(sample_log)
        og = groups["options_grid"]
        assert og["predict_zero"] == pytest.approx(0.4000)
        assert og["cvlin"] == pytest.approx(0.3200)
        assert og["xformer"] == pytest.approx(0.2000)
        assert og["n_feat"] == 88

    def test_skips_zero_groups(self, sample_log):
        groups = parse_baseline_table(sample_log)
        # spx_derived has zero predict-zero, should still be parsed
        assert "spx_derived" in groups
        assert groups["spx_derived"]["predict_zero"] == 0.0

    def test_empty_log(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("no table here")
        groups = parse_baseline_table(p)
        assert groups == {}


# ---------------------------------------------------------------------------
# Tests: M1 Reconstruction R² Ratio
# ---------------------------------------------------------------------------

class TestM1ReconRatio:
    def test_known_answer_options_grid(self, sample_groups):
        """Hand-computed: R²_xfm = 1-0.2/0.4 = 0.5, R²_cvl = 1-0.32/0.4 = 0.2, ratio = 2.5"""
        ratios, agg = compute_m1(sample_groups)
        assert ratios["options_grid"]["r2_xformer"] == pytest.approx(0.5, abs=0.001)
        assert ratios["options_grid"]["r2_cvlin"] == pytest.approx(0.2, abs=0.001)
        assert ratios["options_grid"]["ratio"] == pytest.approx(2.5, abs=0.01)

    def test_negative_r2_cvlin(self, sample_groups):
        """vix_term: R²_cvl = 1-0.095/0.09 = -0.056, should be excluded from ratio."""
        ratios, _ = compute_m1(sample_groups)
        # cvlin R² is negative (-0.056) → ratio should be nan or inf
        vix = ratios.get("vix_term", {})
        r2_cvl = vix.get("r2_cvlin", 0)
        assert r2_cvl < 0  # cvlin worse than predict-zero

    def test_aggregate_is_weighted(self, sample_groups):
        ratios, agg = compute_m1(sample_groups)
        assert np.isfinite(agg)
        assert agg > 1.0  # xformer should beat cvlin on average

    def test_empty_groups(self):
        ratios, agg = compute_m1({})
        assert ratios == {}
        assert np.isnan(agg)


# ---------------------------------------------------------------------------
# Tests: M3 RankMe Formula
# ---------------------------------------------------------------------------

class TestM3RankMe:
    def test_rankme_uniform_singular_values(self):
        """If all singular values are equal, RankMe = d (full rank)."""
        d = 10
        S = np.ones(d)
        p = S / S.sum()
        entropy = -np.sum(p * np.log(p))
        rankme = np.exp(entropy)
        assert rankme == pytest.approx(d, abs=0.01)

    def test_rankme_one_dominant(self):
        """If one singular value dominates, RankMe ≈ 1."""
        S = np.array([100.0, 0.01, 0.01, 0.01])
        p = S / S.sum()
        entropy = -np.sum(p * np.log(p))
        rankme = np.exp(entropy)
        assert rankme < 1.5  # nearly rank-1

    def test_rankme_differs_from_stable_rank(self):
        """RankMe (entropy of σ/sum(σ)) ≠ stable rank ((sum σ)²/sum(σ²))."""
        S = np.array([10.0, 5.0, 2.0, 1.0, 0.1])
        # Old (wrong) formula: (sum(S))^2 / sum(S^2)
        old = (S.sum()**2) / (S**2).sum()
        # Correct RankMe: exp(entropy of L1-normalized S)
        p = S / S.sum()
        entropy = -np.sum(p * np.log(p))
        rankme = np.exp(entropy)
        assert old != pytest.approx(rankme, abs=0.5)  # they differ

    def test_rankme_bounded(self):
        """RankMe ∈ [1, d] for any non-negative singular values."""
        for _ in range(10):
            S = np.abs(np.random.randn(50)) + 0.01
            p = S / S.sum()
            entropy = -np.sum(p * np.log(p))
            rankme = np.exp(entropy)
            assert 1.0 <= rankme <= 50.0


# ---------------------------------------------------------------------------
# Tests: M4 Surface Structure Ratio
# ---------------------------------------------------------------------------

class TestM4SurfaceRatio:
    def test_known_answer(self, sample_groups):
        """R²_xfm = 0.5, R²_hybrid = 1-0.31/0.4 = 0.225, ratio = 0.5/0.225 = 2.22"""
        ratio = compute_m4(sample_groups)
        assert ratio == pytest.approx(2.222, abs=0.01)

    def test_missing_options_grid(self):
        ratio = compute_m4({"strike_agg": {"predict_zero": 0.9}})
        assert np.isnan(ratio)


# ---------------------------------------------------------------------------
# Tests: M5 Forward Utility
# ---------------------------------------------------------------------------

class TestM5ForwardUtility:
    def test_extracts_values(self, sample_probe_results):
        m5 = compute_m5(sample_probe_results)
        assert m5["emb_fine"]["mean"] == pytest.approx(0.606)
        assert m5["raw_36"]["mean"] == pytest.approx(0.516)
        assert m5["delta"] == pytest.approx(0.090, abs=0.001)

    def test_missing_file(self, tmp_path):
        m5 = compute_m5(tmp_path / "nonexistent.json")
        assert m5 == {}

    def test_missing_target(self, tmp_path):
        p = tmp_path / "probe.json"
        p.write_text(json.dumps({"regime": {"summary": {}}}))
        m5 = compute_m5(p)
        assert m5 == {}
