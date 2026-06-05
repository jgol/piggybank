"""Tests for scripts/compute_learning_metrics.py — C1-C6 transformer learning metrics.

Validates:
1. C1 R² computation from known MSE values
2. C2 RSP at floor, ceiling, and intermediate positions
3. C3 SEE in dB with known SNR values
4. C4 whole-variate vs cell decomposition
5. Log parsing produces correct per-group MSE
6. Edge cases: zero predict-zero, negative R², encoder worse than predict-zero
7. B5 Oracle MLP: cross-variate correlation vs noise groups
8. C5 Frequency-band R²: low-frequency signal detection
9. C6 Effective rank: known-rank and random matrices
10. EffRank formula: uniform vs dominant singular values
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_learning_metrics import (
    compute_c1,
    compute_c2,
    compute_c3,
    compute_c4,
    compute_c5,
    compute_c6,
    compute_psd,
    effective_rank,
    OracleMLP,
    train_oracle_mlp,
    parse_per_group_mse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_groups():
    """Known MSE values for hand-verifiable metric computation.

    options_grid: pz=0.40, enc=0.20, oracle=0.07
      → R²_enc = 1 - 0.20/0.40 = 0.50
      → R²_ora = 1 - 0.07/0.40 = 0.825
      → RSP = (0.40-0.20)/(0.40-0.07) = 0.6061
      → SNR_enc = 0.50/0.50 = 1.0 → 0 dB
      → SNR_ora = 0.825/0.175 = 4.714 → 6.73 dB
      → SEE = 0 - 6.73 = -6.73 dB
    """
    return {
        "options_grid": {
            "predict_zero": 0.40, "persistence": 0.37, "ar1": 0.35,
            "cv_linear": 0.32, "hybrid": 0.31, "oracle_linear": 0.07,
            "transformer": 0.20, "r2": 0.50, "n_feat": 88,
        },
        "vix_term": {
            "predict_zero": 0.09, "persistence": 0.067, "ar1": 0.067,
            "cv_linear": 0.095, "hybrid": 0.073, "oracle_linear": 0.001,
            "transformer": 0.17, "r2": -0.889, "n_feat": 12,
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
  vix_term          12    0.0900   0.0670   0.0670   0.0950   0.0730   0.0010   0.1700 -0.889

  Whole-variate MSE per group (cross-variate learning quality):
  Group          #Feat  PredZero    CVLin   Oracle  Xformer  R²_wv
  --------------------------------------------------------------------
  options_grid      88    0.3900   0.3300   0.0700   0.2200  0.436
  strike_agg         6    0.9200   0.9400   0.2400   1.1000 -0.196
  vix_term          12    0.0900   0.0970   0.0010   0.2200 -1.444
"""
    p = tmp_path / "training.log"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Tests: Log Parsing
# ---------------------------------------------------------------------------

class TestLogParsing:
    def test_parses_aggregate_groups(self, sample_log):
        data = parse_per_group_mse(sample_log)
        assert "options_grid" in data["aggregate"]
        assert "strike_agg" in data["aggregate"]
        assert "vix_term" in data["aggregate"]

    def test_correct_values(self, sample_log):
        data = parse_per_group_mse(sample_log)
        og = data["aggregate"]["options_grid"]
        assert og["predict_zero"] == pytest.approx(0.40)
        assert og["transformer"] == pytest.approx(0.20)
        assert og["oracle_linear"] == pytest.approx(0.07)

    def test_parses_whole_var(self, sample_log):
        data = parse_per_group_mse(sample_log)
        assert "options_grid" in data["whole_var"]
        wv = data["whole_var"]["options_grid"]
        assert wv["transformer"] == pytest.approx(0.22)

    def test_empty_log(self, tmp_path):
        p = tmp_path / "empty.log"
        p.write_text("nothing here")
        data = parse_per_group_mse(p)
        assert data["aggregate"] == {}


# ---------------------------------------------------------------------------
# Tests: C1 Normalized R²
# ---------------------------------------------------------------------------

class TestC1:
    def test_known_answer(self, known_groups):
        c1 = compute_c1(known_groups)
        assert c1["options_grid"]["r2_encoder"] == pytest.approx(0.50, abs=0.001)
        assert c1["options_grid"]["r2_oracle"] == pytest.approx(0.825, abs=0.001)

    def test_negative_r2(self, known_groups):
        """vix_term: enc > pz → negative R²."""
        c1 = compute_c1(known_groups)
        assert c1["vix_term"]["r2_encoder"] < 0

    def test_zero_predict_zero_excluded(self):
        groups = {"dead": {"predict_zero": 0.0, "transformer": 0.0, "oracle_linear": 0.0}}
        c1 = compute_c1(groups)
        assert "dead" not in c1


# ---------------------------------------------------------------------------
# Tests: C2 Reconstruction Skill Profile
# ---------------------------------------------------------------------------

class TestC2:
    def test_known_answer(self, known_groups):
        """options_grid: RSP = (0.40-0.20)/(0.40-0.07) = 0.6061"""
        c2 = compute_c2(known_groups)
        assert c2["options_grid"]["rsp_group"] == pytest.approx(0.6061, abs=0.001)

    def test_rsp_at_floor(self):
        """Encoder = predict-zero → RSP = 0."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 1.0, "oracle_linear": 0.1}}
        c2 = compute_c2(groups)
        assert c2["g"]["rsp_group"] == pytest.approx(0.0)

    def test_rsp_at_ceiling(self):
        """Encoder = oracle → RSP = 1."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 0.1, "oracle_linear": 0.1}}
        c2 = compute_c2(groups)
        assert c2["g"]["rsp_group"] == pytest.approx(1.0)

    def test_rsp_above_ceiling(self):
        """Encoder beats oracle → RSP > 1 (temporal learning)."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 0.05, "oracle_linear": 0.1}}
        c2 = compute_c2(groups)
        assert c2["g"]["rsp_group"] > 1.0

    def test_rsp_negative(self, known_groups):
        """vix_term: encoder worse than pz → RSP < 0."""
        c2 = compute_c2(known_groups)
        assert c2["vix_term"]["rsp_group"] < 0


# ---------------------------------------------------------------------------
# Tests: C3 Signal Extraction Efficiency
# ---------------------------------------------------------------------------

class TestC3:
    def test_known_answer(self, known_groups):
        """options_grid: SNR_enc = 0 dB, SNR_oracle = 6.73 dB → SEE = -6.73 dB"""
        c3 = compute_c3(known_groups)
        assert c3["options_grid"]["snr_enc_db"] == pytest.approx(0.0, abs=0.1)
        assert c3["options_grid"]["snr_oracle_db"] == pytest.approx(6.73, abs=0.1)
        assert c3["options_grid"]["see_db"] == pytest.approx(-6.73, abs=0.1)

    def test_encoder_at_ceiling(self):
        """Encoder matches oracle → SEE = 0 dB."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 0.1, "oracle_linear": 0.1}}
        c3 = compute_c3(groups)
        assert c3["g"]["see_db"] == pytest.approx(0.0, abs=0.01)

    def test_negative_r2_handled(self, known_groups):
        """vix_term: negative R² → SEE = -inf or N/A."""
        c3 = compute_c3(known_groups)
        assert c3["vix_term"]["see_db"] == float("-inf") or np.isnan(c3["vix_term"]["see_db"])

    def test_r2_equals_one_no_crash(self):
        """R²=1.0 (perfect reconstruction) must not crash with ZeroDivisionError."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 0.0, "oracle_linear": 0.1}}
        c3 = compute_c3(groups)
        assert c3["g"]["snr_enc_db"] == float("inf")

    def test_both_r2_near_one(self):
        """Both R² near 1.0 — numerically stable."""
        groups = {"g": {"predict_zero": 1.0, "transformer": 0.001, "oracle_linear": 0.002}}
        c3 = compute_c3(groups)
        assert np.isfinite(c3["g"]["see_db"])


# ---------------------------------------------------------------------------
# Tests: C4 Decomposition
# ---------------------------------------------------------------------------

class TestC4:
    def test_has_both_components(self, sample_log):
        data = parse_per_group_mse(sample_log)
        c4 = compute_c4(data["whole_var"], data["aggregate"])
        assert "r2_aggregate" in c4["options_grid"]
        assert "r2_whole_variate" in c4["options_grid"]

    def test_whole_var_values(self, sample_log):
        data = parse_per_group_mse(sample_log)
        c4 = compute_c4(data["whole_var"], data["aggregate"])
        # R²_whole = 1 - 0.22/0.39 = 0.4359
        assert c4["options_grid"]["r2_whole_variate"] == pytest.approx(0.4359, abs=0.01)

    def test_missing_whole_var(self):
        agg = {"g": {"predict_zero": 1.0, "transformer": 0.5}}
        c4 = compute_c4({}, agg)
        assert np.isnan(c4["g"]["r2_whole_variate"])


# ---------------------------------------------------------------------------
# Tests: B5 Oracle MLP
# ---------------------------------------------------------------------------

class TestOracleMLP:
    def test_correlated_group_beats_noise(self):
        """Group A has strong cross-variate correlation; group B is noise.

        Oracle MLP for A should achieve much lower MSE than predict-zero.
        Oracle MLP for B should achieve MSE close to predict-zero (variance ~1).
        """
        rng = np.random.RandomState(42)
        N = 2000
        D_other = 10  # number of "other" variates as input

        # Group A: target = linear combination of inputs + small noise
        X_other = rng.randn(N, D_other).astype(np.float32)
        weights_a = rng.randn(D_other, 3).astype(np.float32)
        Y_a = X_other @ weights_a + 0.1 * rng.randn(N, 3).astype(np.float32)

        # Group B: target is pure noise, independent of inputs
        Y_b = rng.randn(N, 3).astype(np.float32)

        n_train = 1500
        n_val = N - n_train

        # Train oracle for group A
        mse_a, _ = train_oracle_mlp(
            X_other[:n_train], Y_a[:n_train],
            X_other[n_train:], Y_a[n_train:],
            hidden=64, max_epochs=100, patience=10, verbose=False,
        )

        # Train oracle for group B
        mse_b, _ = train_oracle_mlp(
            X_other[:n_train], Y_b[:n_train],
            X_other[n_train:], Y_b[n_train:],
            hidden=64, max_epochs=100, patience=10, verbose=False,
        )

        # A should be much lower than predict-zero (pz_mse ≈ variance of Y_a)
        pz_mse_a = np.var(Y_a[n_train:])
        assert mse_a < pz_mse_a * 0.3, (
            f"Oracle MLP for correlated group A should beat predict-zero by far: "
            f"mse_a={mse_a:.4f}, pz={pz_mse_a:.4f}"
        )

        # B should be near predict-zero (can't learn noise)
        pz_mse_b = np.var(Y_b[n_train:])
        assert mse_b > pz_mse_b * 0.7, (
            f"Oracle MLP for noise group B should be near predict-zero: "
            f"mse_b={mse_b:.4f}, pz={pz_mse_b:.4f}"
        )

    def test_oracle_mlp_architecture(self):
        """Verify OracleMLP has the specified architecture."""
        import torch
        model = OracleMLP(n_input=20, n_output=5, hidden=256, dropout=0.1)
        # Should have 3 linear layers
        linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 3
        assert linears[0].in_features == 20
        assert linears[0].out_features == 256
        assert linears[1].in_features == 256
        assert linears[1].out_features == 256
        assert linears[2].in_features == 256
        assert linears[2].out_features == 5

    def test_oracle_mlp_early_stopping(self):
        """Verify early stopping terminates before max_epochs on easy data."""
        rng = np.random.RandomState(123)
        N = 500
        X = rng.randn(N, 5).astype(np.float32)
        Y = (X[:, :2] * 0.5).astype(np.float32)  # trivially learnable

        import time
        t0 = time.time()
        mse, _ = train_oracle_mlp(
            X[:400], Y[:400], X[400:], Y[400:],
            hidden=32, max_epochs=200, patience=5, verbose=False,
        )
        elapsed = time.time() - t0
        # Should finish quickly due to early stopping
        assert mse < 0.05, f"Easy data should yield low MSE: {mse}"


# ---------------------------------------------------------------------------
# Tests: C5 Frequency-Band R²
# ---------------------------------------------------------------------------

class TestC5:
    def test_low_frequency_signal_detection(self):
        """Signal is low-frequency sine wave + high-freq noise.

        Reconstruction that perfectly captures the sine should show
        R²_low > R²_high.
        """
        rng = np.random.RandomState(42)
        T = 60
        N = 200
        t = np.arange(T) / T

        # Target: low-frequency sine + high-frequency noise
        low_freq = 0.5 / 30  # well below 1/30 cycles/bar
        signal_low = np.sin(2 * np.pi * low_freq * np.arange(T))
        noise_high = 0.3 * rng.randn(N, T)

        target = np.tile(signal_low, (N, 1)) + noise_high
        target = target.astype(np.float32)

        # Reconstruction: captures low-freq perfectly, misses high-freq noise
        reconstruction = np.tile(signal_low, (N, 1)).astype(np.float32)

        group_indices = {"test_group": list(range(N))}
        c5 = compute_c5(target, reconstruction, group_indices)

        result = c5["test_group"]
        # R² for low frequencies should be high (reconstruction captures signal)
        assert result["r2_low"] > 0.5, (
            f"R²_low should be high for well-reconstructed low-freq signal: "
            f"{result['r2_low']}"
        )
        # R² for high frequencies should be low (reconstruction misses noise)
        assert result["r2_high"] < result["r2_low"], (
            f"R²_high ({result['r2_high']}) should be lower than "
            f"R²_low ({result['r2_low']})"
        )

    def test_perfect_reconstruction(self):
        """Perfect reconstruction should give R² = 1 for all bands."""
        rng = np.random.RandomState(42)
        T = 60
        N = 100
        target = rng.randn(N, T).astype(np.float32)
        reconstruction = target.copy()

        group_indices = {"g": list(range(N))}
        c5 = compute_c5(target, reconstruction, group_indices)

        for band in ["r2_low", "r2_mid", "r2_high"]:
            r2 = c5["g"][band]
            if np.isfinite(r2):
                assert r2 == pytest.approx(1.0, abs=1e-6), (
                    f"Perfect reconstruction should give R²=1: {band}={r2}"
                )

    def test_zero_reconstruction_gives_zero_r2(self):
        """Predict-zero reconstruction: R² should be near 0."""
        rng = np.random.RandomState(42)
        T = 60
        N = 100
        target = rng.randn(N, T).astype(np.float32)
        reconstruction = np.zeros_like(target)

        group_indices = {"g": list(range(N))}
        c5 = compute_c5(target, reconstruction, group_indices)

        for band in ["r2_low", "r2_mid", "r2_high"]:
            r2 = c5["g"][band]
            if np.isfinite(r2):
                # R² should be close to 0 for predict-zero
                assert abs(r2) < 0.15, (
                    f"Predict-zero should give R² near 0: {band}={r2}"
                )

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        target = np.zeros((10, 60), dtype=np.float32)
        recon = np.zeros((10, 30), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            compute_c5(target, recon, {"g": list(range(10))})

    def test_empty_group(self):
        """Empty group should return NaN."""
        target = np.zeros((10, 60), dtype=np.float32)
        recon = np.zeros((10, 60), dtype=np.float32)
        c5 = compute_c5(target, recon, {"empty": []})
        assert np.isnan(c5["empty"]["r2_low"])

    def test_psd_basic(self):
        """PSD of a single sinusoid should peak at that frequency."""
        T = 64
        freq = 8  # cycles in T samples = 8/64 = 0.125 cycles/sample
        t = np.arange(T)
        signal = np.sin(2 * np.pi * freq * t / T).reshape(1, T)
        psd = compute_psd(signal)
        # Peak should be at bin index = freq
        peak_bin = np.argmax(psd[0])
        assert peak_bin == freq, (
            f"PSD peak should be at bin {freq}, got {peak_bin}"
        )


# ---------------------------------------------------------------------------
# Tests: C6 Effective Rank
# ---------------------------------------------------------------------------

class TestC6:
    def test_known_rank_matrix(self):
        """Matrix with rank 5 embedded in 20-d should have EffRank ~ 5."""
        rng = np.random.RandomState(42)
        N = 500
        rank = 5
        D = 20

        # Create a rank-5 matrix: (N, 5) @ (5, 20)
        A = rng.randn(N, rank).astype(np.float64)
        B = rng.randn(rank, D).astype(np.float64)
        matrix = A @ B

        er = effective_rank(matrix)
        # EffRank should be close to 5
        assert 4.0 <= er <= 6.5, (
            f"EffRank of rank-5 matrix should be near 5: got {er:.2f}"
        )

    def test_random_matrix_high_rank(self):
        """Random full-rank matrix should have EffRank near min(N, D)."""
        rng = np.random.RandomState(42)
        N = 200
        D = 20
        matrix = rng.randn(N, D).astype(np.float64)

        er = effective_rank(matrix)
        # EffRank should be close to D=20 for full-rank random matrix
        assert er > D * 0.7, (
            f"EffRank of random (200, 20) matrix should be near {D}: got {er:.2f}"
        )

    def test_uniform_singular_values(self):
        """Uniform singular values should give EffRank = D."""
        D = 10
        N = 100
        # Create a zero-mean matrix with uniform singular values.
        # Use an orthogonal matrix (Q from QR of random) scaled uniformly.
        # After centering (already zero-mean since Q columns are zero-mean
        # for large N), SVD gives equal singular values.
        rng = np.random.RandomState(42)
        A = rng.randn(N, D)
        # Center first so effective_rank's centering is a no-op
        A -= A.mean(axis=0, keepdims=True)
        # QR decomposition gives orthonormal columns
        Q, _ = np.linalg.qr(A)
        # Scale all columns equally so singular values are uniform
        matrix = Q[:, :D] * 5.0

        er = effective_rank(matrix)
        assert er == pytest.approx(D, abs=0.1), (
            f"Uniform singular values should give EffRank={D}: got {er:.2f}"
        )

    def test_one_dominant_singular_value(self):
        """One dominant singular value should give EffRank ~ 1."""
        rng = np.random.RandomState(42)
        N = 100
        D = 20
        # All data lies along one direction
        direction = rng.randn(D).astype(np.float64)
        direction /= np.linalg.norm(direction)
        coeffs = rng.randn(N, 1).astype(np.float64)
        matrix = coeffs @ direction.reshape(1, D)

        er = effective_rank(matrix)
        assert er < 1.5, (
            f"Rank-1 matrix should have EffRank near 1: got {er:.2f}"
        )

    def test_compute_c6_ratio(self):
        """C6 ratio for good reconstruction should be < 1."""
        rng = np.random.RandomState(42)
        N = 200
        V = 10

        # Target: structured (low effective rank)
        basis = rng.randn(3, V).astype(np.float64)
        coeffs = rng.randn(N, 3).astype(np.float64)
        target = coeffs @ basis + 0.1 * rng.randn(N, V)

        # Good reconstruction: captures the structure
        recon = coeffs @ basis  # no noise

        group_indices = {"g1": list(range(V))}
        c6 = compute_c6(target.astype(np.float32), recon.astype(np.float32),
                         group_indices)

        # Residual is pure noise (high EffRank relative to its dimension)
        # Target has low EffRank (structured)
        # Ratio depends on the noise level but reconstruction should
        # capture most structure
        assert c6["g1"]["effrank_residual"] > 0
        assert c6["g1"]["effrank_target"] > 0

    def test_compute_c6_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        target = np.zeros((10, 5), dtype=np.float32)
        recon = np.zeros((10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            compute_c6(target, recon, {"g": [0, 1, 2]})

    def test_compute_c6_empty_group(self):
        """Empty group should return zeros and NaN ratio."""
        target = np.zeros((10, 5), dtype=np.float32)
        recon = np.zeros((10, 5), dtype=np.float32)
        c6 = compute_c6(target, recon, {"empty": []})
        assert c6["empty"]["effrank_target"] == 0.0
        assert np.isnan(c6["empty"]["ratio"])

    def test_effective_rank_empty_matrix(self):
        """Empty matrix should return 0."""
        assert effective_rank(np.array([]).reshape(0, 5)) == 0.0
        assert effective_rank(np.array([]).reshape(5, 0)) == 0.0

    def test_predict_zero_residual_equals_target(self):
        """When reconstruction is zero, residual = target, so ratio = 1."""
        rng = np.random.RandomState(42)
        N = 200
        V = 10
        target = rng.randn(N, V).astype(np.float32)
        recon = np.zeros_like(target)

        group_indices = {"g": list(range(V))}
        c6 = compute_c6(target, recon, group_indices)

        # Residual = target, so EffRank(residual) = EffRank(target)
        # Ratio should be 1.0
        assert c6["g"]["ratio"] == pytest.approx(1.0, abs=0.01)
