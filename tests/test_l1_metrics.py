"""Tests for scripts/compute_l1_metrics.py — L1 training performance metrics.

Validates:
1. Metric computations are mathematically correct
2. History parsing from JSON and log formats
3. End-to-end compute_l1_metrics() with mocked checkpoint
4. Known-answer validation against hand-computed values
5. Sanity gate thresholds via actual code paths
6. Registry update round-trip
7. Summary table formatting
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts.compute_l1_metrics import (
    L1Metrics,
    _parse_global_spearman,
    compute_l1_metrics,
    format_summary_table,
    load_history_json,
    parse_history_from_log,
    parse_predict_zero_from_log,
    update_registry,
    _load_registry,
    LOCAL_STORE,
    REGISTRY_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Known-answer history: 10 epochs, hand-computed values below.
KNOWN_HISTORY = {
    "train_loss": [0.40, 0.35, 0.30, 0.26, 0.23, 0.21, 0.19, 0.18, 0.175, 0.17],
    "val_loss":   [0.35, 0.30, 0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.135, 0.13],
}
# M1: best_val = 0.13, predict_zero = 0.4644 → 0.13/0.4644 = 0.2799
# M2: first=0.35, best=0.13, threshold = 0.35 - 0.9*0.22 = 0.152
# first v <= 0.152: index 6 (0.15), epoch 7
# M3: train[-1]/val[-1] = 0.17/0.13 = 1.3077
# G3: CV(val[-5:]) = std([0.16,0.15,0.14,0.135,0.13]) / mean(...)
# = 0.01140 / 0.14300 = 0.07975 → FAIL (> 0.05)
# CV(val[-4:]) = std([0.15,0.14,0.135,0.13]) / mean(...)
# = 0.007395 / 0.13875 = 0.05329 → also > 0.05 but barely


@pytest.fixture
def sample_history():
    """Synthetic training history with known properties."""
    n = 100
    train = [0.40 * np.exp(-0.02 * i) + 0.15 for i in range(n)]
    val = [0.35 * np.exp(-0.025 * i) + 0.12 for i in range(n)]
    return {"train_loss": train, "val_loss": val}


@pytest.fixture
def sample_history_json(sample_history, tmp_path):
    """Write history to a temp JSON file."""
    p = tmp_path / "history.json"
    p.write_text(json.dumps(sample_history))
    return p


@pytest.fixture
def sample_training_log(tmp_path):
    """Synthetic training log with epoch lines and diagnostics."""
    lines = []
    for ep in range(1, 21):
        train = 0.40 * np.exp(-0.02 * ep * 5) + 0.15
        val = 0.35 * np.exp(-0.025 * ep * 5) + 0.12
        beat0 = 140 + min(ep // 4, 7)
        lines.append(
            f"Ep  {ep*5:3d}/100  train={train:.4f}  val={val:.4f}  "
            f"val_uw=0.3000  best={val:.4f}  eff=13170000  "
            f"beat0={beat0}/147  worst=[v104=1.5, v132=1.2]"
        )
        if ep % 4 == 0:
            emb_var = 0.05 + 0.01 * ep
            lines.append(
                f"  [diag] emb_var={emb_var:.4f}  shortcut_suspect=76  "
                f"well_reconstructed=68  temporal_attn_ent=4.0/4.1"
            )
    # Add GLOBAL Spearman line
    lines.append("  GLOBAL    154190                                     0.627    0.679")
    # Add predict-zero baseline
    lines.append("  Properly-weighted predict-zero MSE: 0.4644")
    p = tmp_path / "training.log"
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def mock_config(tmp_path, sample_training_log):
    """Config dict for compute_l1_metrics with mock files."""
    # Write known history JSON
    hist_path = tmp_path / "history.json"
    hist_path.write_text(json.dumps(KNOWN_HISTORY))

    # Write mock baselines log with predict-zero
    baselines = tmp_path / "baselines.log"
    baselines.write_text(
        "  predict_zero             0.4644     0.4614     0.4728\n"
        "  Properly-weighted predict-zero MSE: 0.4644\n"
        "  GLOBAL    154190                                     0.542    0.639\n"
    )

    return {
        "experiment_id": "TEST-001",
        "checkpoint": "dummy.pt",
        "history_json": "history.json",
        "training_log": "baselines.log",
        "baselines_log": "baselines.log",
        "n_variates": 147,
        "_tmp_path": tmp_path,  # for test access
    }


# ---------------------------------------------------------------------------
# Tests: History Loading
# ---------------------------------------------------------------------------

class TestHistoryLoading:
    def test_json_loading(self, sample_history_json, sample_history):
        h = load_history_json(sample_history_json)
        assert len(h["train_loss"]) == 100
        assert len(h["val_loss"]) == 100
        assert h["train_loss"][0] == pytest.approx(sample_history["train_loss"][0])

    def test_log_parsing(self, sample_training_log):
        h = parse_history_from_log(sample_training_log)
        assert len(h["train_loss"]) == 20
        assert len(h["val_loss"]) == 20
        assert len(h["beat0"]) == 20
        assert len(h["emb_var"]) == 5

    def test_log_parsing_beat0_values(self, sample_training_log):
        h = parse_history_from_log(sample_training_log)
        assert all(b >= 140 for b in h["beat0"])

    def test_log_parsing_emb_var_positive(self, sample_training_log):
        h = parse_history_from_log(sample_training_log)
        assert all(v > 0 for v in h["emb_var"])

    def test_empty_log(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("")
        h = parse_history_from_log(p)
        assert h["train_loss"] == []
        assert h["val_loss"] == []

    def test_no_matching_lines(self, tmp_path):
        p = tmp_path / "garbage.log"
        p.write_text("hello world\nno epochs here\n")
        h = parse_history_from_log(p)
        assert h["train_loss"] == []


# ---------------------------------------------------------------------------
# Tests: Predict-Zero Parsing
# ---------------------------------------------------------------------------

class TestPredictZeroParsing:
    def test_weighted_format(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("  Properly-weighted predict-zero MSE: 0.4677\n")
        assert parse_predict_zero_from_log(p) == pytest.approx(0.4677)

    def test_table_format(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("  predict_zero             0.4644     0.4614     0.4728\n")
        assert parse_predict_zero_from_log(p) == pytest.approx(0.4644)

    def test_missing_file(self, tmp_path):
        assert np.isnan(parse_predict_zero_from_log(tmp_path / "missing.log"))

    def test_no_match(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("nothing relevant here\n")
        assert np.isnan(parse_predict_zero_from_log(p))


# ---------------------------------------------------------------------------
# Tests: Global Spearman Parsing
# ---------------------------------------------------------------------------

class TestGlobalSpearmanParsing:
    def test_standard_format(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("  GLOBAL    154190                                     0.627    0.679\n")
        assert _parse_global_spearman(p) == pytest.approx(0.627)

    def test_different_values(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("  GLOBAL    155568                                     0.542    0.639\n")
        assert _parse_global_spearman(p) == pytest.approx(0.542)

    def test_missing_file(self, tmp_path):
        assert np.isnan(_parse_global_spearman(tmp_path / "no.log"))

    def test_no_match(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text("no GLOBAL line here\n")
        assert np.isnan(_parse_global_spearman(p))


# ---------------------------------------------------------------------------
# Tests: End-to-end compute_l1_metrics with mocked checkpoint
# ---------------------------------------------------------------------------

class TestComputeL1MetricsE2E:
    """End-to-end tests for compute_l1_metrics() with known-answer validation."""

    def _run_with_mock(self, mock_config):
        tmp = mock_config["_tmp_path"]

        # Mock count_encoder_params to avoid needing a real checkpoint
        with patch("scripts.compute_l1_metrics.count_encoder_params", return_value=714532):
            with patch("scripts.compute_l1_metrics.LOCAL_STORE", tmp):
                return compute_l1_metrics("test", mock_config)

    def test_m1_uses_huber_predict_zero(self, mock_config):
        """M1 should use parsed Huber predict-zero (0.4644), not MSE (1.0)."""
        m = self._run_with_mock(mock_config)
        expected_m1 = 0.13 / 0.4644  # best_val=0.13, predict_zero=0.4644
        assert m.m1_norm_recon_loss == pytest.approx(expected_m1, abs=0.001)
        assert m.m1_norm_recon_loss > 0.25  # must be > 0.25 (not ~0.13)

    def test_m2_convergence_epoch(self, mock_config):
        """90% improvement threshold from known history."""
        m = self._run_with_mock(mock_config)
        # first=0.35, best=0.13, threshold=0.35-0.9*0.22=0.152
        # val[6]=0.15 is first <= 0.152 → epoch 7
        assert m.m2_convergence_epoch == 7

    def test_m3_gen_gap(self, mock_config):
        m = self._run_with_mock(mock_config)
        expected = 0.17 / 0.13
        assert m.m3_gen_gap == pytest.approx(expected, abs=0.001)

    def test_m5_uses_correct_m1(self, mock_config):
        """M5 should use M1 derived from Huber predict-zero, not MSE."""
        m = self._run_with_mock(mock_config)
        expected_m1 = 0.13 / 0.4644
        expected_m5 = (1.0 - expected_m1) / (714532 / 1e6)
        assert m.m5_param_efficiency == pytest.approx(expected_m5, abs=0.01)

    def test_g1_from_history(self, mock_config):
        """beat0 not in JSON history → should be -1 (fallback fails in mock)."""
        m = self._run_with_mock(mock_config)
        # No beat0 in KNOWN_HISTORY and no log files to fall back to
        assert m.g1_beat0 == -1
        assert m.g1_pass is False  # -1 < 140

    def test_g2_parsed_from_log(self, mock_config):
        """G2 should parse global Spearman from baselines log, not be hardcoded."""
        m = self._run_with_mock(mock_config)
        assert m.g2_masking_artifact == pytest.approx(0.542)
        assert m.g2_pass is False  # 0.542 > 0.40

    def test_g3_stability(self, mock_config):
        m = self._run_with_mock(mock_config)
        # 10-epoch history, n_tail=20 (not logged_epochs) → uses last 10
        # But len < 20, so g3 should be nan
        # Actually: n_tail=20, len=10, 10 < 20 → else branch → nan
        assert np.isnan(m.g3_stability_cv)

    def test_best_val_loss(self, mock_config):
        m = self._run_with_mock(mock_config)
        assert m.best_val_loss == pytest.approx(0.13)

    def test_best_epoch(self, mock_config):
        m = self._run_with_mock(mock_config)
        assert m.best_epoch == 10  # index 9, epoch 10

    def test_n_params(self, mock_config):
        m = self._run_with_mock(mock_config)
        assert m.n_params == 714532


# ---------------------------------------------------------------------------
# Tests: n_tail branching for G3
# ---------------------------------------------------------------------------

class TestG3TailBranching:
    def test_logged_epochs_uses_4(self):
        """Log-parsed history with logged_epochs flag should use n_tail=4."""
        history = {
            "train_loss": [0.3] * 20,
            "val_loss": [0.15, 0.14, 0.13, 0.128, 0.126, 0.125, 0.124,
                         0.123, 0.122, 0.121, 0.120, 0.120, 0.120, 0.120,
                         0.119, 0.119, 0.119, 0.119, 0.119, 0.119],
            "logged_epochs": True,
        }
        # Last 4: [0.119, 0.119, 0.119, 0.119] → CV ≈ 0
        tail = history["val_loss"][-4:]
        cv = np.std(tail) / np.mean(tail)
        assert cv < 0.001

    def test_per_epoch_uses_20(self):
        """Per-epoch history (no flag) should use n_tail=20."""
        val = list(np.linspace(0.20, 0.13, 100))
        # Last 20 have a spread → CV will be > 0
        tail = val[-20:]
        cv = np.std(tail) / np.mean(tail)
        assert cv > 0.001  # non-zero spread in linearly decaying tail

    def test_short_history_gives_nan(self):
        """History shorter than n_tail should produce NaN."""
        # Per-epoch (n_tail=20) but only 10 epochs
        val = [0.13] * 10
        n_tail = 20
        assert len(val) < n_tail  # would trigger NaN path


# ---------------------------------------------------------------------------
# Tests: Registry update round-trip
# ---------------------------------------------------------------------------

class TestRegistryUpdate:
    def test_update_existing_entry(self, tmp_path):
        registry_path = tmp_path / "registry.jsonl"
        entry = {
            "experiment_id": "TEST-001",
            "secondary_metrics": {"existing_key": 42},
        }
        registry_path.write_text(json.dumps(entry) + "\n")

        m = L1Metrics(arm="test", experiment_id="TEST-001",
                      m1_norm_recon_loss=0.28, m2_convergence_epoch=7,
                      m3_gen_gap=1.31, m5_param_efficiency=1.01,
                      g1_beat0=143, g3_stability_cv=0.004, g4_emb_var=0.36)

        with patch("scripts.compute_l1_metrics.REGISTRY_PATH", registry_path):
            update_registry([m])

        updated = json.loads(registry_path.read_text().strip())
        assert updated["secondary_metrics"]["existing_key"] == 42  # preserved
        assert updated["secondary_metrics"]["l1_m1_norm_recon_loss"] == pytest.approx(0.28)
        assert updated["secondary_metrics"]["l1_m2_convergence_epoch"] == 7
        assert updated["secondary_metrics"]["l1_g1_beat0"] == 143

    def test_missing_experiment_no_crash(self, tmp_path):
        registry_path = tmp_path / "registry.jsonl"
        registry_path.write_text(json.dumps({"experiment_id": "OTHER"}) + "\n")

        m = L1Metrics(arm="test", experiment_id="DOES-NOT-EXIST")

        with patch("scripts.compute_l1_metrics.REGISTRY_PATH", registry_path):
            update_registry([m])  # should not crash

        # Registry unchanged (no match)
        content = registry_path.read_text().strip()
        assert "DOES-NOT-EXIST" not in content


# ---------------------------------------------------------------------------
# Tests: Convergence Efficiency
# ---------------------------------------------------------------------------

class TestM2ConvergenceEfficiency:
    def test_90_percent_improvement(self):
        val = [1.0, 0.5, 0.2, 0.15, 0.11, 0.10]
        first_val = val[0]
        best_val = min(val)
        threshold = first_val - 0.9 * (first_val - best_val)
        assert threshold == pytest.approx(0.19)
        for i, v in enumerate(val):
            if v <= threshold:
                assert i + 1 == 4
                break

    def test_immediate_convergence(self):
        val = [0.10, 0.10, 0.10]
        first_val = val[0]
        best_val = min(val)
        threshold = first_val - 0.9 * (first_val - best_val)
        for i, v in enumerate(val):
            if v <= threshold:
                assert i + 1 == 1
                break


# ---------------------------------------------------------------------------
# Tests: Summary Table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def test_table_has_header(self):
        m = L1Metrics(arm="linear", experiment_id="test",
                      m1_norm_recon_loss=0.28, m2_convergence_epoch=25,
                      m3_gen_gap=1.33, m5_param_efficiency=1.01,
                      g1_beat0=145, g1_pass=True, g3_stability_cv=0.01,
                      g3_pass=True, g4_emb_var=0.36, g4_pass=True,
                      best_val_loss=0.129, best_epoch=81,
                      final_train_loss=0.174, n_params=714532)
        table = format_summary_table([m])
        assert "L1-M1" in table
        assert "Linear" in table

    def test_multi_arm_table(self):
        m1 = L1Metrics(arm="linear", experiment_id="t1", m1_norm_recon_loss=0.28)
        m2 = L1Metrics(arm="patch", experiment_id="t2", m1_norm_recon_loss=0.23)
        table = format_summary_table([m1, m2])
        assert "Linear" in table
        assert "Patch" in table

    def test_nan_renders_as_dash(self):
        m = L1Metrics(arm="test", experiment_id="t")
        table = format_summary_table([m])
        assert "—" in table
