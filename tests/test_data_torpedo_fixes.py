"""DATA-layer torpedo fixes (audit 2026-06-02). Three audit-confirmed defects in
the minute-Parquet data path, each pinned so it cannot silently regress:

  H3 (HIGH) — the corpus stores MinutesToClose=0 at every day's bar 0 (open),
    colliding open with close. That makes ThetaUrgency=1/sqrt(max(0,1))=1.0 (the
    +4.5 normalized clamp-region MAX) at every open; Lag(ThetaUrgency,1)/Delta at
    bar 1 then read that garbage and the GP hunts spurious "trade-at-open" rules.
    Fixed at the source (generate_minute_parquet.py): bar-0 MTC is repaired to the
    true open value (~404 = max within-day MTC + 1).

  M5 (LOW-MED) — ATM_IV saturated the +5 normalization clamp on ~2.6% of bars
    (upper-tail discrimination loss). Fixed by winsorizing the raw ATM_IV source
    at 0.80 and recalibrating the terminal_stats scale (cap→+4.5) so raw IV~0.5
    maps inside ±5 and <0.5% of bars clamp.

  M1 (MEDIUM) — experiment.py computed per-fold norm stats from the raw Parquet
    columns, so the 8+ SYNTHESIZED terminals (RV5d, VIXChange, ThetaUrgency,
    SPXReturn3d, VIXMean5d, IVRVGap5d, ATM_IV_5m, RealizedVol30m_5m) fell back to
    the frozen date<=2024-09-30 constants — a per-fold leak. Fixed by fitting
    per-fold stats over the SYNTHESIZED (un-normalized) train arrays via
    compute_norm_stats_from_arrays.

These tests load the regenerated parquet where present; the M1 train-only/leak
assertions are pure-function and run without it.
"""
import os

import numpy as np
import pandas as pd
import pytest

from layer2.terminal_stats import (
    TERMINAL_NORM_STATS,
    compute_norm_stats_from_arrays,
    compute_norm_stats_from_data,
    normalize,
)

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
_TRAIN_END = "2024-09-27"  # same train-split convention as the 2026-06 minute stats
_CLAMP = 5.0               # evaluator OOD clamp is [-5, +5]
_has_parquet = os.path.exists(_PARQUET)
_skip_no_parquet = pytest.mark.skipif(not _has_parquet, reason="minute parquet absent")


# --- shared helpers ---------------------------------------------------------

def _load(cols):
    return pd.read_parquet(_PARQUET, columns=cols)


def _ff_atm_iv_within_day(df):
    """Replicate prepare_terminal_data's ATM_IV=0 within-day forward-fill so the
    clamp rate measured here matches what the evaluator actually normalizes."""
    a = df["ATM_IV"].values.astype(np.float64).copy()
    dts = df["date"].values
    for j in range(len(a)):
        if a[j] == 0.0 and j > 0 and str(dts[j]) == str(dts[j - 1]):
            a[j] = a[j - 1]
    return a


# ===========================================================================
# H3 — bar-0 MinutesToClose / ThetaUrgency open artifact
# ===========================================================================

@_skip_no_parquet
def test_bar0_minutes_to_close_is_session_length_not_zero():
    """Every day's bar 0 (open) carries the true session length (~404), never 0.
    The post-open ramp (bars 1, 2 = 403, 402) must remain untouched."""
    df = _load(["date", "bar_position", "MinutesToClose"])
    b0 = df[df.bar_position == 0]["MinutesToClose"]
    assert (b0 != 0).all(), "some bar-0 MinutesToClose is still 0 (open==close collision)"
    # corpus session is 09:30→16:15 = 405 min; the open bar is max-within-day + 1.
    assert (b0 == 404.0).all(), f"bar-0 MTC should be 404, got values {b0.unique()[:5]}"
    # later bars are the genuine descending ramp — only bar 0 was repaired.
    assert df[df.bar_position == 1]["MinutesToClose"].median() == 403.0
    assert df[df.bar_position == 2]["MinutesToClose"].median() == 402.0


@_skip_no_parquet
def test_theta_urgency_at_open_not_max():
    """ThetaUrgency at the open is now the MINIMUM urgency (raw 1/sqrt(404)≈0.0498,
    z≈-0.24), NOT the spurious +4.5 clamp-region max it was when bar-0 MTC=0."""
    df = _load(["date", "bar_position", "MinutesToClose"])
    mtc0 = df[df.bar_position == 0]["MinutesToClose"].values.astype(np.float64)
    tu0_raw = 1.0 / np.sqrt(np.maximum(mtc0, 1.0))
    assert np.all(tu0_raw < 0.06), f"bar-0 ThetaUrgency raw still high: max={tu0_raw.max()}"
    z0 = np.array([normalize("ThetaUrgency", v) for v in tu0_raw])
    assert np.all(np.abs(z0) < 1.0), f"bar-0 ThetaUrgency normalized not near 0: {z0[:3]}"
    assert np.all(z0 < 4.0), "bar-0 ThetaUrgency must be far below the +4.5 max"
    # The genuine urgency MAX (raw 1.0, z≈+4.5) is preserved at the close (MTC→1).
    mtc_last = df[df.bar_position == 404]["MinutesToClose"].values.astype(np.float64)
    tu_last = 1.0 / np.sqrt(np.maximum(mtc_last, 1.0))
    assert normalize("ThetaUrgency", tu_last[0]) > 4.0, "close-region urgency max lost"


@_skip_no_parquet
def test_lag_theta_urgency_at_bar1_no_longer_sees_plus_4_5():
    """`Lag(ThetaUrgency, 1)` at within-day bar 1 reads bar 0's ThetaUrgency. With
    the bar-0 MTC=0 artifact that was the +4.5 max (the "trade-at-open" garbage the
    GP hunted); after the repair it is the open's true ~-0.24."""
    df = _load(["date", "bar_position", "MinutesToClose"]).sort_values(
        ["date", "bar_position"]).reset_index(drop=True)
    mtc = df["MinutesToClose"].values.astype(np.float64)
    tu_raw = 1.0 / np.sqrt(np.maximum(mtc, 1.0))
    tu_z = np.array([normalize("ThetaUrgency", v) for v in tu_raw])
    dates = df["date"].values
    bp = df["bar_position"].values
    # Lag-1 of ThetaUrgency at every within-day bar 1 == bar 0's normalized value.
    bar1_idx = np.where((bp == 1) & (np.array([str(dates[i]) == str(dates[i - 1])
                                               for i in range(len(dates))])))[0]
    assert len(bar1_idx) > 800, "expected a bar-1 row on (almost) every day"
    lagged = tu_z[bar1_idx - 1]  # Lag(ThetaUrgency, 1) seen at bar 1
    # NONE may be at/above the +4.5 clamp-region max (the pre-fix artifact).
    assert lagged.max() < 4.0, (
        f"Lag(ThetaUrgency,1) at bar 1 still sees the +4.5 open artifact "
        f"(max lagged z={lagged.max():.3f})")
    # And they all sit near the open minimum.
    assert np.all(np.abs(lagged) < 1.0), "bar-1 lagged ThetaUrgency not at the open minimum"


# ===========================================================================
# M5 — ATM_IV clamp saturation
# ===========================================================================

@_skip_no_parquet
def test_atm_iv_clamp_saturation_below_half_percent_train():
    """REGRESSION PIN: < 0.5% of TRAIN bars saturate the [-5,+5] OOD clamp.
    Pre-fix this was ~2.5% (frozen scale 0.0486 mapped IV~0.5 to z=+8.2)."""
    df = _load(["date", "bar_position", "ATM_IV"])
    train = df[df.date <= _TRAIN_END]
    a = _ff_atm_iv_within_day(train)
    z = np.array([normalize("ATM_IV", v) for v in a])
    frac = float(np.mean(np.abs(z) >= _CLAMP))
    assert frac < 0.005, f"ATM_IV clamp saturation {frac:.4%} exceeds 0.5% (train)"
    # IV~0.5 must map INSIDE the clamp (audit requirement).
    assert abs(normalize("ATM_IV", 0.5)) < _CLAMP, "raw IV=0.5 still clamps"


@_skip_no_parquet
def test_atm_iv_clamp_saturation_below_half_percent_all():
    """Same < 0.5% clamp bound over the FULL parquet (not just train)."""
    df = _load(["date", "bar_position", "ATM_IV"])
    a = _ff_atm_iv_within_day(df)
    z = np.array([normalize("ATM_IV", v) for v in a])
    assert float(np.mean(np.abs(z) >= _CLAMP)) < 0.005, "ATM_IV clamp >0.5% over all bars"


@_skip_no_parquet
def test_atm_iv_source_winsorized_at_0_80():
    """The raw ATM_IV source is winsorized at 0.80 — no bar exceeds it, and the
    grid IV surface (IV_ATM etc.) is intentionally NOT winsorized (pricing keeps
    its full range)."""
    df = _load(["ATM_IV", "IV_ATM"])
    assert df["ATM_IV"].max() <= 0.80 + 1e-9, "ATM_IV not winsorized at 0.80"
    # IV_ATM is the same underlying ATM-call IV but is the SURFACE column; it must
    # keep its raw (un-winsorized) tail so pricing/skew are unaffected.
    assert df["IV_ATM"].max() > 0.80, "grid IV surface was winsorized (should not be)"


def test_atm_iv_norm_constants_recalibrated():
    """Frozen ATM_IV / ATM_IV_5m scales widened from ~0.048 (cap→+4.5 on winsorized
    data) so the upper tail no longer saturates."""
    assert TERMINAL_NORM_STATS["ATM_IV"][1] > 0.10, "ATM_IV scale not widened"
    assert TERMINAL_NORM_STATS["ATM_IV_5m"][1] > 0.10, "ATM_IV_5m scale not widened"
    # IV=0.5 lands well inside ±5 under the new affine map.
    c, s, _ = TERMINAL_NORM_STATS["ATM_IV"]
    assert abs((0.5 - c) / s) < 4.0


# ===========================================================================
# M1 — per-fold norm stats from SYNTHESIZED terminal arrays
# ===========================================================================

# Synthesized terminals that are NOT raw Parquet columns, so the old
# compute_norm_stats_from_data path could only return their frozen constants.
_SYNTH_TERMS = ["RV5d", "VIXChange", "SPXReturn3d", "VIXMean5d",
                "IVRVGap5d", "ATM_IV_5m", "RealizedVol30m_5m"]


def _synthetic_minute_df(day_closes, atm_iv_by_day=None, vix_by_day=None):
    """Build a minimal multi-day minute DataFrame (30 bars/day, MTC 44..15) that
    exercises the synthesized-terminal code in prepare_terminal_data."""
    rows = []
    for d, close in enumerate(day_closes):
        date = f"2024-0{1 + d // 28}-{1 + d % 28:02d}"
        atm = 0.12 if atm_iv_by_day is None else atm_iv_by_day[d]
        vix = 16.0 if vix_by_day is None else vix_by_day[d]
        for k, mtc in enumerate(range(44, 14, -1)):  # 30 bars
            frac = k / 29.0
            px = close - (1.0 - frac) * 2.0
            rows.append({"date": date, "bar_position": k, "MinutesToClose": float(mtc),
                         "SPXClose": px, "ATM_IV": atm, "RawSpread": 0.015,
                         "VIXSpot": vix})
    return pd.DataFrame(rows)


@_skip_no_parquet
def test_m1_synthesized_terminals_get_per_fold_stats_not_frozen():
    """Over a real fold-1 train window, every synthesized terminal gets a per-fold
    stat that DIFFERS from its frozen constant — whereas the OLD column-only path
    (compute_norm_stats_from_data) returns the frozen constant for all of them.
    This is the exact M1 leak the fix closes."""
    from layer2.evaluator_vectorized import prepare_terminal_data
    df = _load(None if False else  # full columns needed for synthesis
               ["date", "bar_position", "ATM_IV", "RawSpread", "DeltaSpread1",
                "DeltaSpread5", "MinutesToClose", "VIXSpot", "VIXTermSlope",
                "RealizedVol30m", "SPXClose"]).sort_values(
        ["date", "bar_position"]).reset_index(drop=True)
    fold1_train = df[df.date <= "2024-03-29"]
    assert fold1_train.date.nunique() > 100, "fold-1 train window too small for the test"

    synth_raw = prepare_terminal_data(fold1_train, normalize_terminals=False)
    new_stats = compute_norm_stats_from_arrays(synth_raw)
    old_stats = compute_norm_stats_from_data(fold1_train)

    for t in _SYNTH_TERMS:
        frozen = TERMINAL_NORM_STATS[t]
        # OLD path could not see the synthesized terminal -> frozen fallback.
        assert old_stats[t] == frozen, (
            f"{t}: old column path unexpectedly produced a non-frozen stat")
        # NEW path fits it from the synthesized train array -> differs from frozen.
        assert new_stats[t] != frozen, (
            f"{t}: per-fold synthesized stat did not differ from frozen {frozen}")


def test_m1_per_fold_stats_are_train_only():
    """The fitted stat for a data-dependent synthesized terminal must be computed
    from ONLY the rows passed in: appending POST-train days changes it (proving a
    fold-1 stat is not influenced by post-train data — the look-ahead guard)."""
    # 18 days of rising VIX so VIXMean5d/VIXChange are clearly data-dependent.
    vix = [14, 15, 16, 15, 17, 18, 16, 19, 20, 18, 21, 22, 20, 23, 24, 22, 25, 26]
    closes = [5800 + 5 * i for i in range(len(vix))]
    full = _synthetic_minute_df(closes, vix_by_day=vix)

    from layer2.evaluator_vectorized import prepare_terminal_data
    dates_sorted = sorted(full["date"].unique())
    train_dates = set(dates_sorted[:10])          # "fold-1" train = first 10 days
    train = full[full["date"].isin(train_dates)]

    train_stats = compute_norm_stats_from_arrays(
        prepare_terminal_data(train, normalize_terminals=False))
    full_stats = compute_norm_stats_from_arrays(
        prepare_terminal_data(full, normalize_terminals=False))

    # VIXMean5d is built from prior daily VIX -> a train-only fit MUST differ from
    # the fit that also saw the rising post-train days.
    assert train_stats["VIXMean5d"] != full_stats["VIXMean5d"], (
        "VIXMean5d stat unchanged when post-train days were appended -> the fit is "
        "NOT train-only (look-ahead leak)")

    # And re-fitting on the SAME train rows is deterministic / unaffected by data
    # outside the slice (idempotent on identical input).
    train_stats2 = compute_norm_stats_from_arrays(
        prepare_terminal_data(train, normalize_terminals=False))
    assert train_stats["VIXMean5d"] == train_stats2["VIXMean5d"]


def test_m1_compute_norm_stats_from_arrays_skips_vectors_and_keeps_frozen():
    """compute_norm_stats_from_arrays must: (a) fit 1-D arrays it is given, (b) keep
    the frozen constant for terminals it is NOT given, (c) keep regime indicators
    frozen (fixed {0.5,0.5} affine maps), (d) skip 2-D typed-vector arrays."""
    arrays = {
        "VIXSpot": np.linspace(12.0, 30.0, 200),           # present, robust -> fit
        "RealizedVol30m": np.abs(np.random.RandomState(0).randn(200)) * 1e-3,  # standard -> fit
        "RegimeIsHigh": np.array([0.0, 1.0] * 100),         # must stay frozen
        "EMB_fine": np.zeros((50, 8)),                      # 2-D -> skipped (keep frozen)
    }
    out = compute_norm_stats_from_arrays(arrays)
    assert out["VIXSpot"] != TERMINAL_NORM_STATS["VIXSpot"], "VIXSpot should be re-fit"
    assert out["VIXSpot"][2] == "robust", "method must match the frozen method"
    assert out["RealizedVol30m"][2] == "standard"
    assert out["RegimeIsHigh"] == TERMINAL_NORM_STATS["RegimeIsHigh"], (
        "regime indicators must keep their fixed {0.5,0.5} affine map")
    assert "EMB_fine" not in out, "2-D typed vectors must not get a scalar stat"
    # A terminal entirely absent from `arrays` keeps its frozen constant.
    assert out["ATM_IV"] == TERMINAL_NORM_STATS["ATM_IV"]


def test_m1_too_few_samples_falls_back_to_frozen():
    """<10 finite samples -> keep the frozen constant (don't fit a degenerate stat)."""
    out = compute_norm_stats_from_arrays({"VIXSpot": np.array([14.0, 15.0, 16.0])})
    assert out["VIXSpot"] == TERMINAL_NORM_STATS["VIXSpot"]


# ===========================================================================
# Integrity — regen sanity (row count, no NaN/inf)
# ===========================================================================

@_skip_no_parquet
def test_regen_row_count_and_no_nan_inf():
    df = pd.read_parquet(_PARQUET)
    assert 350_000 < len(df) < 365_000, f"unexpected row count {len(df)}"
    assert df.groupby("date").size().nunique() == 1, "bars-per-day not constant"
    num = df.select_dtypes(include=[np.number])
    assert int(num.isna().sum().sum()) == 0, "NaN present in regenerated parquet"
    assert int(np.isinf(num.values).sum()) == 0, "inf present in regenerated parquet"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
