"""Unit tests for layer1.data.backfill_flow_aggregates.

These tests cover the pure computation layer (`compute_aggregates`,
`detect_dead_day`, `_signed_log1p`, `_signed_expm1`, `_sha256_of_ndarray`)
which is deployable-to-QC-Research-notebook code but has no QuantBook
dependency. I/O helpers that touch `qb.ObjectStore` are not exercised here;
they are verified on the first dry-run against live ObjectStore.

Run locally:
    PYTHONPATH=. python3 -m pytest tests/test_backfill_flow_aggregates.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow the test file to import the backfill module without installing the
# package. Matches the convention in tests/test_grammar.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util

import numpy as np
import pytest

# The backfill module imports QC-only APIs lazily via `get_qb()` — importing
# the module itself is safe outside QC Research because the `get_qb()`
# function is never invoked in these tests.
_spec = importlib.util.spec_from_file_location(
    "backfill_flow_aggregates",
    Path(__file__).resolve().parents[1] / "layer1" / "data" / "backfill_flow_aggregates.py",
)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BARS = backfill.BARS_PER_DAY_NOMINAL  # 405 — v2 corpus default; tests may use others
SOURCE_V = backfill.SOURCE_V
TARGET_V = backfill.TARGET_V
FLOW_START = backfill.FLOW_START
FLOW_END = backfill.FLOW_END


def _synthetic_v2(seed: int = 42, include_log1p: bool = True) -> np.ndarray:
    """Return a synthetic (390, 141) float32 tensor that passes shape/dtype guards.

    - Flow block v131-v139 is filled with reproducible random values in [-1, 1]
      for non-log1p indices and with pre-log1p-compressed synthetic values
      (in range ~ ±10) for indices 1 and 6.
    - Remaining variates (non-flow) are filled with random floats in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-1.0, 1.0, size=(BARS, SOURCE_V)).astype(np.float32)
    # Fill flow block with moderate values
    flow = rng.uniform(-1.0, 1.0, size=(BARS, 9)).astype(np.float32)
    if include_log1p:
        # Simulate stored signed_log1p on indices 1 (v132) and 6 (v137)
        # Raw values of a few thousand → log1p(|x|) is ~7-8
        raw_132 = rng.uniform(-1000.0, 1000.0, size=(BARS,))
        raw_137 = rng.uniform(-5000.0, 5000.0, size=(BARS,))
        flow[:, 1] = (np.sign(raw_132) * np.log1p(np.abs(raw_132))).astype(np.float32)
        flow[:, 6] = (np.sign(raw_137) * np.log1p(np.abs(raw_137))).astype(np.float32)
    arr[:, FLOW_START:FLOW_END] = flow
    return arr


# ---------------------------------------------------------------------------
# _signed_log1p / _signed_expm1 roundtrip
# ---------------------------------------------------------------------------

class TestSignedLog1pRoundtrip:
    def test_expm1_inverts_log1p_on_positives(self):
        x = np.array([0.0, 1.0, 100.0, 1e4, 1e5], dtype=np.float64)
        compressed = backfill._signed_log1p(x)
        recovered = backfill._signed_expm1(compressed)
        np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-6)

    def test_expm1_inverts_log1p_on_negatives(self):
        x = np.array([0.0, -1.0, -100.0, -1e4, -1e5], dtype=np.float64)
        compressed = backfill._signed_log1p(x)
        recovered = backfill._signed_expm1(compressed)
        np.testing.assert_allclose(recovered, x, rtol=1e-10, atol=1e-6)

    def test_sign_zero_stable(self):
        # sign(0) = 0; log1p(0) = 0; composition is 0
        assert backfill._signed_log1p(np.array([0.0])) == 0.0
        assert backfill._signed_expm1(np.array([0.0])) == 0.0


# ---------------------------------------------------------------------------
# SHA-256 determinism
# ---------------------------------------------------------------------------

class TestSha256:
    def test_deterministic_same_array(self):
        arr = _synthetic_v2(seed=0)
        h1 = backfill._sha256_of_ndarray(arr)
        h2 = backfill._sha256_of_ndarray(arr)
        assert h1 == h2

    def test_different_arrays_differ(self):
        a = _synthetic_v2(seed=0)
        b = _synthetic_v2(seed=1)
        assert backfill._sha256_of_ndarray(a) != backfill._sha256_of_ndarray(b)

    def test_stable_across_views_and_copies(self):
        arr = _synthetic_v2(seed=0)
        view = arr[:]  # view, not copy
        copy = arr.copy()
        assert backfill._sha256_of_ndarray(arr) == backfill._sha256_of_ndarray(view)
        assert backfill._sha256_of_ndarray(arr) == backfill._sha256_of_ndarray(copy)


# ---------------------------------------------------------------------------
# Dead-day detection
# ---------------------------------------------------------------------------

class TestDeadDayDetection:
    def test_all_zero_is_dead(self):
        flow = np.zeros((BARS, 9), dtype=np.float64)
        assert backfill.detect_dead_day(flow) is True

    def test_any_nan_is_dead(self):
        flow = np.ones((BARS, 9), dtype=np.float64)
        flow[100, 3] = np.nan
        assert backfill.detect_dead_day(flow) is True

    def test_empty_is_dead(self):
        flow = np.zeros((0, 0), dtype=np.float64)
        assert backfill.detect_dead_day(flow) is True

    def test_normal_is_not_dead(self):
        rng = np.random.default_rng(0)
        flow = rng.uniform(-1.0, 1.0, size=(BARS, 9))
        assert backfill.detect_dead_day(flow) is False

    def test_99pct_zero_triggers_dead(self):
        rng = np.random.default_rng(0)
        flow = rng.uniform(-1.0, 1.0, size=(BARS, 9))
        # Zero out 99.5% of cells → should trigger
        mask = rng.uniform(0, 1, size=flow.shape) < 0.995
        flow[mask] = 0.0
        assert backfill.detect_dead_day(flow) is True

    def test_95pct_zero_not_dead(self):
        rng = np.random.default_rng(0)
        flow = rng.uniform(-1.0, 1.0, size=(BARS, 9))
        # Zero out only 95% — below threshold
        mask = rng.uniform(0, 1, size=flow.shape) < 0.95
        flow[mask] = 0.0
        # Note: some seeds will still push above 99% due to the random +/-
        # uniform also including near-zero values. Regenerate with wider range.
        flow = rng.uniform(0.1, 1.0, size=(BARS, 9))
        flow[rng.uniform(0, 1, size=flow.shape) < 0.95] = 0.0
        assert backfill.detect_dead_day(flow) is False


# ---------------------------------------------------------------------------
# compute_aggregates — shape, dtype, byte-invariance
# ---------------------------------------------------------------------------

class TestComputeAggregatesShape:
    def test_output_shape(self):
        v2 = _synthetic_v2()
        v3, _diag = backfill.compute_aggregates(v2)
        assert v3.shape == (BARS, TARGET_V)

    def test_output_dtype_float32(self):
        v2 = _synthetic_v2()
        v3, _diag = backfill.compute_aggregates(v2)
        assert v3.dtype == np.float32

    def test_byte_invariance_first_141_columns(self):
        v2 = _synthetic_v2()
        v3, _diag = backfill.compute_aggregates(v2)
        assert np.array_equal(v3[:, :SOURCE_V], v2), (
            "byte-invariance violated: v3[:, :141] != v2"
        )

    def test_rejects_wrong_dtype(self):
        bad = np.zeros((BARS, SOURCE_V), dtype=np.float64)
        with pytest.raises(ValueError, match="expected source dtype float32"):
            backfill.compute_aggregates(bad)

    def test_rejects_wrong_variate_count(self):
        bad = np.zeros((BARS, 100), dtype=np.float32)
        with pytest.raises(ValueError, match=r"expected source shape \(N, 141\)"):
            backfill.compute_aggregates(bad)

    def test_rejects_empty_tensor(self):
        bad = np.zeros((0, SOURCE_V), dtype=np.float32)
        with pytest.raises(ValueError):
            backfill.compute_aggregates(bad)

    def test_accepts_variable_bar_counts(self):
        """v2 corpus has variable N per day (405 regular / fewer on half-days).
        The aggregate formula is N-agnostic."""
        for n_bars in (100, 390, 405, 210):  # 210 ≈ typical half-day
            v2 = _synthetic_v2(seed=n_bars)[:n_bars] if n_bars <= BARS else None
            if v2 is None:
                continue
            v3, _ = backfill.compute_aggregates(v2)
            assert v3.shape == (n_bars, TARGET_V), f"failed at N={n_bars}"
            assert v3.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_aggregates — formula correctness
# ---------------------------------------------------------------------------

class TestComputeAggregatesFormula:
    def test_rolling_mean_on_non_log1p_variate(self):
        """v133 (index 2 in flow block, non-log1p) should be a pure rolling mean.

        Aggregate layout: v141..v149 = roll3_v{131..139}, so roll3_v133 is at
        column SOURCE_V + 2 (v131 is offset 0, v132 is offset 1, v133 is offset 2).
        """
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        ramp = np.arange(BARS, dtype=np.float32)
        v2[:, 133] = ramp
        v3, _diag = backfill.compute_aggregates(v2)
        col_roll3_v133 = SOURCE_V + 2  # v143 = flow_roll3_v133
        # Warmup: bar 0 uses 1 value [0]; bar 1 uses [0,1]; bar 2 uses [0,1,2]; bar 3 uses [1,2,3]
        assert v3[0, col_roll3_v133] == pytest.approx(0.0)
        assert v3[1, col_roll3_v133] == pytest.approx(0.5)
        assert v3[2, col_roll3_v133] == pytest.approx(1.0)
        assert v3[3, col_roll3_v133] == pytest.approx(2.0)
        assert v3[10, col_roll3_v133] == pytest.approx(9.0)  # mean of 8,9,10

    def test_rolling_mean_15_vs_3(self):
        """Spike at bar 50 should propagate through the 3-bar and 15-bar windows
        for different durations."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        # Populate flow block with constant baseline so dead-day detection
        # (≥99% zero) doesn't fire.
        v2[:, FLOW_START:FLOW_END] = 1.0
        v2[50, 133] = 11.0  # spike on top of baseline of 1.0
        v3, _diag = backfill.compute_aggregates(v2)
        col_roll3 = SOURCE_V + 2        # v143 = flow_roll3_v133
        col_roll15 = SOURCE_V + 9 + 2   # v152 = flow_roll15_v133
        # 3-bar window sees spike for bars 50, 51, 52 (above baseline=1.0)
        assert v3[50, col_roll3] > 1.0
        assert v3[52, col_roll3] > 1.0
        assert v3[53, col_roll3] == pytest.approx(1.0, rel=1e-5)
        # 15-bar window sees spike for bars 50..64
        assert v3[50, col_roll15] > 1.0
        assert v3[64, col_roll15] > 1.0
        assert v3[65, col_roll15] == pytest.approx(1.0, rel=1e-5)

    def test_log1p_inversion_on_v132(self):
        """v132 (index 1 in flow block) is stored as signed_log1p. Aggregate should
        invert, take rolling mean in raw units, re-compress."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        # Set v132 such that raw value is 1000 at every bar
        raw = 1000.0
        v2[:, 132] = np.sign(raw) * np.log1p(abs(raw))
        v3, _diag = backfill.compute_aggregates(v2)
        col_roll3_v132 = SOURCE_V + 1  # v142
        # Rolling mean of constant raw=1000 is still 1000; re-compressed is log1p(1000)
        expected_storage = np.log1p(1000.0)
        # Allow for fp32 precision tolerance
        np.testing.assert_allclose(
            v3[10, col_roll3_v132], expected_storage, rtol=1e-4, atol=1e-4
        )

    def test_log1p_inversion_on_v137(self):
        """Same as v132 but for index 6 (v137)."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        raw = -5000.0
        v2[:, 137] = np.sign(raw) * np.log1p(abs(raw))
        v3, _diag = backfill.compute_aggregates(v2)
        col_roll15_v137 = SOURCE_V + 9 + 6  # v156
        expected_storage = np.sign(raw) * np.log1p(abs(raw))
        np.testing.assert_allclose(
            v3[20, col_roll15_v137], expected_storage, rtol=1e-4, atol=1e-4
        )

    def test_non_log1p_aggregate_does_not_invert(self):
        """v131 (put/call ratio, stored raw) must NOT be inverted. Rolling mean of
        constant c is c, not log1p(c)."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        v2[:, 131] = 2.5  # constant put/call ratio
        v3, _diag = backfill.compute_aggregates(v2)
        col_roll3_v131 = SOURCE_V + 0
        np.testing.assert_allclose(
            v3[10, col_roll3_v131], 2.5, rtol=1e-6, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Dead-day propagation
# ---------------------------------------------------------------------------

class TestDeadDayPropagation:
    def test_all_zero_flow_yields_nan_aggregates(self):
        # Zero out the flow block; everything else is unchanged
        v2 = _synthetic_v2()
        v2[:, FLOW_START:FLOW_END] = 0.0
        v3, diag = backfill.compute_aggregates(v2)
        assert diag["dead_day"] is True
        # All aggregate columns must be NaN
        assert np.all(np.isnan(v3[:, SOURCE_V:])), "dead day should force NaN in aggregates"
        # First 141 columns still byte-equal v2
        assert np.array_equal(v3[:, :SOURCE_V], v2)

    def test_nan_in_flow_triggers_dead_day(self):
        v2 = _synthetic_v2()
        v2[10, 133] = np.nan  # single NaN anywhere in flow block
        v3, diag = backfill.compute_aggregates(v2)
        assert diag["dead_day"] is True
        assert np.all(np.isnan(v3[:, SOURCE_V:]))

    def test_normal_day_produces_finite_aggregates(self):
        v2 = _synthetic_v2(seed=123)
        v3, diag = backfill.compute_aggregates(v2)
        assert diag["dead_day"] is False
        assert np.all(np.isfinite(v3[:, SOURCE_V:]))


# ---------------------------------------------------------------------------
# Raw-scale info flags (not fatal; informational only)
# ---------------------------------------------------------------------------

class TestRawScaleInfoFlags:
    def test_normal_range_has_no_flags(self):
        v2 = _synthetic_v2()
        _v3, diag = backfill.compute_aggregates(v2)
        assert diag["info_flags"] == []
        # max_abs_raw_scale dict is populated for each log1p variate
        assert "132" in diag["max_abs_raw_scale"]
        assert "137" in diag["max_abs_raw_scale"]

    def test_moderately_high_raw_values_no_flag(self):
        """2022-11-23 real data: raw ~1.3M on v132. Not fatal, no flag unless > 1e10."""
        v2 = _synthetic_v2()
        moderate = 1.3e6
        v2[0, 132] = np.sign(moderate) * np.log1p(abs(moderate))
        _v3, diag = backfill.compute_aggregates(v2)
        # Should NOT raise, should NOT flag (1.3M << 1e10 info threshold)
        assert diag["info_flags"] == []
        assert diag["max_abs_raw_scale"]["132"] >= 1e6

    def test_extreme_raw_values_flagged_but_not_fatal(self):
        """Stored value > log1p(1e10) ≈ 23 trips the info flag but does NOT abort."""
        v2 = _synthetic_v2()
        extreme = 1e11  # raw 100B, unambiguously weird
        v2[0, 132] = np.sign(extreme) * np.log1p(abs(extreme))
        # compute_aggregates must succeed; info_flags records the anomaly
        v3, diag = backfill.compute_aggregates(v2)
        assert v3.shape[1] == TARGET_V
        assert len(diag["info_flags"]) >= 1
        flag = diag["info_flags"][0]
        assert flag["variate"] == 132
        assert flag["max_abs_raw_scale"] >= 1e10


# ---------------------------------------------------------------------------
# NaN counts in diagnostics
# ---------------------------------------------------------------------------

class TestNaNCounts:
    def test_nan_counts_sum_zero_on_normal_day(self):
        v2 = _synthetic_v2(seed=7)
        _v3, diag = backfill.compute_aggregates(v2)
        total = sum(diag["nan_counts"].values())
        assert total == 0

    def test_nan_counts_on_dead_day(self):
        v2 = _synthetic_v2()
        v2[:, FLOW_START:FLOW_END] = 0.0
        _v3, diag = backfill.compute_aggregates(v2)
        total = sum(diag["nan_counts"].values())
        # All 18 aggregate columns × 390 bars = 7020 NaNs on a dead day
        assert total == backfill.AGGREGATE_COUNT * BARS


# ---------------------------------------------------------------------------
# _load_day_tensor silent-truncation guard (Code Reviewer FIX 1)
# ---------------------------------------------------------------------------

class TestLoadCorruptionGuard:
    """Verify the load helper raises a routable RuntimeError on corrupted payloads.

    Uses monkeypatching of get_qb so we can simulate ObjectStore returning
    invalid JSON / truncated content without needing a real QuantBook.
    """

    def _make_fake_qb(self, returns_value, contains_key=True):
        class FakeObjectStore:
            def ContainsKey(self, key):
                return contains_key
            def Read(self, key):
                return returns_value
            def Save(self, key, val):
                pass

        class FakeQB:
            ObjectStore = FakeObjectStore()

        return FakeQB()

    def test_corrupt_json_raises_sev1(self, monkeypatch):
        fake = self._make_fake_qb("#")  # silent-truncation pattern from deploy_diagnostic
        monkeypatch.setattr(backfill, "get_qb", lambda: fake)
        with pytest.raises(RuntimeError, match="Sev-1 abort"):
            backfill._load_day_tensor("layer1_feat_v2/2023-01-03.npz")

    def test_missing_data_key_raises_sev1(self, monkeypatch):
        fake = self._make_fake_qb('{"meta": {}}')  # valid JSON, missing data field
        monkeypatch.setattr(backfill, "get_qb", lambda: fake)
        with pytest.raises(RuntimeError, match="Sev-1 abort"):
            backfill._load_day_tensor("layer1_feat_v2/2023-01-03.npz")

    def test_bad_base64_raises_sev1(self, monkeypatch):
        fake = self._make_fake_qb('{"meta": {}, "data": "!!!not-base64!!!"}')
        monkeypatch.setattr(backfill, "get_qb", lambda: fake)
        with pytest.raises(RuntimeError, match="Sev-1 abort"):
            backfill._load_day_tensor("layer1_feat_v2/2023-01-03.npz")

    def test_missing_key_returns_none(self, monkeypatch):
        # Key absent should return (None, None), NOT raise
        fake = self._make_fake_qb("", contains_key=False)
        monkeypatch.setattr(backfill, "get_qb", lambda: fake)
        arr, meta = backfill._load_day_tensor("layer1_feat_v2/2023-01-03.npz")
        assert arr is None and meta is None


# ---------------------------------------------------------------------------
# Aggregate layout correctness — does v141..v158 match the documented order?
# ---------------------------------------------------------------------------

class TestAggregateLayout:
    def test_v141_is_roll3_of_v131(self):
        """Sanity-check the documented layout: aggregates v141..v149 are
        flow_roll3_v{131..139} in order."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        # Put a unique marker in v131 (put/call ratio, non-log1p)
        v2[:, 131] = 3.3
        v3, _diag = backfill.compute_aggregates(v2)
        # v141 (SOURCE_V + 0) = flow_roll3_v131
        np.testing.assert_allclose(v3[100, SOURCE_V + 0], 3.3, rtol=1e-6)

    def test_v150_is_roll15_of_v131(self):
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        v2[:, 131] = 3.3
        v3, _diag = backfill.compute_aggregates(v2)
        # v150 (SOURCE_V + 9) = flow_roll15_v131
        np.testing.assert_allclose(v3[100, SOURCE_V + 9], 3.3, rtol=1e-6)

    def test_aggregate_block_ordering(self):
        """v141..v158 must be [roll3_v131..roll3_v139, roll15_v131..roll15_v139]."""
        v2 = np.zeros((BARS, SOURCE_V), dtype=np.float32)
        # Different constant in each flow variate (for non-log1p ones)
        for offset in range(9):
            if offset in backfill.SIGNED_LOG1P_INDICES_IN_FLOW:
                continue  # skip log1p indices (they have nonlinear transform)
            v2[:, FLOW_START + offset] = 10.0 + offset
        v3, _diag = backfill.compute_aggregates(v2)
        # After warmup, every bar should see the constant
        for offset in range(9):
            if offset in backfill.SIGNED_LOG1P_INDICES_IN_FLOW:
                continue
            expected = 10.0 + offset
            np.testing.assert_allclose(
                v3[100, SOURCE_V + offset], expected, rtol=1e-6,
                err_msg=f"roll3 aggregate misordered at flow offset {offset}",
            )
            np.testing.assert_allclose(
                v3[100, SOURCE_V + 9 + offset], expected, rtol=1e-6,
                err_msg=f"roll15 aggregate misordered at flow offset {offset}",
            )
