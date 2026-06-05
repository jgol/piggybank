"""ITEM #4 — robust (median / IQR-1.349) per-fold normalization.

Locks: (1) median/IQR computed correctly vs a known array; (2) robust scale <
mean/std scale on a heavy-tailed sample (the outlier-damping that recovers
within-regime IV discrimination); (3) the per-fold pipeline path produces finite,
non-degenerate stats on REAL data; (4) backward-compat — the DEFAULT (robust=False)
path is byte-identical to the historical per-terminal-method behaviour; (5) codegen
parity — QC consumes the SAME robust per-fold stats (positional center/scale).
"""
import os
import inspect

import numpy as np
import pytest

from layer2.terminal_stats import (
    ROBUST_IQR_TO_SIGMA,
    TERMINAL_NORM_STATS,
    _REGIME_SKIP,
    _fit_one_stat,
    compute_norm_stats_from_arrays,
    compute_norm_stats_from_data,
)

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
_skip_no_parquet = pytest.mark.skipif(
    not os.path.exists(_PARQUET), reason="production minute parquet not present")


# ---------------------------------------------------------------------------
# (1) median / IQR correctness vs a known array
# ---------------------------------------------------------------------------

def test_robust_iqr_to_sigma_is_normal_consistent():
    """The divisor is the normal-consistent IQR->sigma factor 2*Phi^-1(0.75)."""
    # 2 * 0.6744897501960817 = 1.3489795..., rounded in the constant to 1.349.
    assert abs(ROBUST_IQR_TO_SIGMA - 1.349) < 1e-9


def test_fit_one_stat_robust_median_and_iqr_known_array():
    """Robust fit returns (median, IQR/1.349) exactly on a known array."""
    # 0..100 inclusive: median=50, q75=75, q25=25, IQR=50.
    arr = np.arange(0, 101, dtype=np.float64)
    center, scale, method = _fit_one_stat(arr, method="standard", robust=True)
    assert method == "robust"
    assert center == pytest.approx(50.0)
    assert scale == pytest.approx(50.0 / 1.349, rel=1e-6)


def test_compute_from_arrays_robust_matches_numpy_on_iv():
    """compute_norm_stats_from_arrays(robust=True) reproduces numpy median/IQR."""
    rng = np.random.RandomState(1)
    iv = np.abs(rng.standard_t(3, size=5000)) * 0.05 + 0.10  # heavy-tailed, IV-like
    out = compute_norm_stats_from_arrays({"ATM_IV": iv}, robust=True)
    c, s, m = out["ATM_IV"]
    q75, q25 = np.percentile(iv, 75), np.percentile(iv, 25)
    assert m == "robust"
    assert c == pytest.approx(round(float(np.median(iv)), 6), abs=1e-6)
    assert s == pytest.approx(round(float((q75 - q25) / 1.349), 6), abs=1e-6)


# ---------------------------------------------------------------------------
# (2) robust dampens outliers: robust scale < mean/std scale on heavy tails
# ---------------------------------------------------------------------------

def test_robust_scale_smaller_than_meanstd_on_heavy_tail():
    """On a sample with a 2022-bear-like heavy upper tail, the robust scale is
    materially smaller than the mean/std scale — the mechanism that recovers
    within-regime discrimination (the 2x STD inflation the plan measured)."""
    rng = np.random.RandomState(7)
    calm = rng.normal(0.12, 0.02, 9000)        # calm regime
    bear = rng.normal(0.45, 0.18, 1000)        # 10% heavy upper tail (2022-like)
    iv = np.concatenate([calm, bear])
    robust = compute_norm_stats_from_arrays({"ATM_IV": iv}, robust=True)["ATM_IV"]
    # mean/std reference (force standard via _fit_one_stat on the same data)
    _, std_scale, _ = _fit_one_stat(iv[np.isfinite(iv)], "standard", robust=False)
    assert robust[1] < std_scale, (
        f"robust scale {robust[1]} not < mean/std scale {std_scale} — "
        f"no outlier damping")
    # The damping should be substantial (>25%) for this fat-tailed mix.
    assert robust[1] < 0.75 * std_scale


def test_robust_recovers_within_regime_iv_discrimination_synthetic():
    """A calm-quantile point sits FARTHER from center (more sigma) under robust
    than under mean/std on a heavy-tailed mix — i.e. robust recovers the
    discrimination the inflated STD compresses."""
    rng = np.random.RandomState(11)
    iv = np.concatenate([rng.normal(0.12, 0.02, 9000), rng.normal(0.50, 0.20, 1000)])
    p25 = float(np.percentile(iv, 25))
    rob_c, rob_s, _ = _fit_one_stat(iv, "standard", robust=True)
    std_c, std_s, _ = _fit_one_stat(iv, "standard", robust=False)
    z_robust = abs((p25 - rob_c) / rob_s)
    z_meanstd = abs((p25 - std_c) / std_s)
    assert z_robust > z_meanstd, (
        f"robust did not increase calm-point separation: "
        f"robust |z|={z_robust:.3f} vs mean/std |z|={z_meanstd:.3f}")


# ---------------------------------------------------------------------------
# (3) per-fold pipeline path: finite, non-degenerate stats on REAL data
# ---------------------------------------------------------------------------

@_skip_no_parquet
def test_per_fold_robust_stats_finite_nondegenerate_real_data():
    """The exact per-fold pipeline call (prepare_terminal_data(raw) ->
    compute_norm_stats_from_arrays(robust=True)) on fold-1 train yields finite,
    positive scales for every continuous terminal it fits."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    df = pd.read_parquet(_PARQUET)
    df["date"] = df["date"].astype(str)
    train = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                          embargo_days=0)["train"]
    raw = prepare_terminal_data(train, normalize_terminals=False)
    stats = compute_norm_stats_from_arrays(raw, robust=True)
    assert stats, "no stats produced"
    for name, (c, s, m) in stats.items():
        assert np.isfinite(c), f"{name} center not finite"
        assert np.isfinite(s) and s > 0, f"{name} scale not finite/positive: {s}"
        # Continuous terminals actually present in the synth dict are forced robust;
        # regime indicators stay standard {0.5,0.5}.
        if name in _REGIME_SKIP:
            assert (c, s, m) == TERMINAL_NORM_STATS[name]
        elif isinstance(raw.get(name), np.ndarray) and raw[name].ndim == 1:
            # Continuous terminals are forced robust; boundary-degenerate-IQR ones
            # (e.g. GridReliability) use the documented robust-center/std-scale
            # fallback so a real-spread terminal is not zeroed.
            assert m in ("robust", "robust_std_fallback"), (
                f"{name} continuous terminal not robust-fit (got {m})")


@_skip_no_parquet
def test_per_fold_robust_atm_iv_scale_below_meanstd_real_data():
    """On REAL fold-1 train, the robust ATM_IV scale is ~half the mean/std scale —
    the measured 2x STD inflation the 2022 bear causes (plan §1.3)."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    df = pd.read_parquet(_PARQUET)
    df["date"] = df["date"].astype(str)
    train = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                          embargo_days=0)["train"]
    raw = prepare_terminal_data(train, normalize_terminals=False)
    iv = np.asarray(raw["ATM_IV"], dtype=np.float64)
    iv = iv[np.isfinite(iv)]
    robust_scale = compute_norm_stats_from_arrays(raw, robust=True)["ATM_IV"][1]
    meanstd_scale = float(iv.std())
    ratio = robust_scale / meanstd_scale
    # Measured ~0.506 on this parquet; allow a band for parquet revisions.
    assert 0.35 < ratio < 0.70, (
        f"robust/meanstd ATM_IV scale ratio {ratio:.3f} outside expected ~0.5 band")


@_skip_no_parquet
def test_per_fold_robust_stats_are_train_only_no_lookahead():
    """The robust per-fold stat for a data-dependent terminal is computed from ONLY
    the rows passed in — appending post-train days changes it (look-ahead guard)."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    df = pd.read_parquet(_PARQUET)
    df["date"] = df["date"].astype(str)
    train = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                          embargo_days=0)["train"]
    train_more = split_by_date(df, train_end="2024-09-27", val_end="2025-01-31",
                               embargo_days=0)["train"]
    s1 = compute_norm_stats_from_arrays(
        prepare_terminal_data(train, normalize_terminals=False), robust=True)
    s2 = compute_norm_stats_from_arrays(
        prepare_terminal_data(train_more, normalize_terminals=False), robust=True)
    assert s1["ATM_IV"] != s2["ATM_IV"], (
        "robust ATM_IV stat unchanged when later days were appended -> not train-only")


# ---------------------------------------------------------------------------
# (4) backward-compat: default (robust=False) unchanged
# ---------------------------------------------------------------------------

def test_default_path_preserves_per_terminal_method():
    """robust=False keeps each terminal's FROZEN method (the historical behaviour
    existing callers/tests rely on)."""
    arrays = {
        "VIXSpot": np.linspace(12.0, 30.0, 500),        # frozen method robust
        "RealizedVol30m": np.abs(np.random.RandomState(0).randn(500)) * 1e-3,  # standard
    }
    out = compute_norm_stats_from_arrays(arrays)  # default robust=False
    assert out["VIXSpot"][2] == "robust"          # matches frozen
    assert out["RealizedVol30m"][2] == "standard"  # matches frozen


def test_default_uses_135_divisor_robust_uses_1349():
    """Back-compat robust branch keeps the historical 1.35 rounding; the FORCE-
    robust path uses the normal-consistent 1.349. (Distinct, by design.)"""
    arr = np.arange(0, 101, dtype=np.float64)  # IQR = 50
    # default path, a frozen-robust terminal -> 1.35 divisor
    c_def, s_def, _ = _fit_one_stat(arr, "robust", robust=False)
    # force-robust path -> 1.349 divisor
    c_rob, s_rob, _ = _fit_one_stat(arr, "standard", robust=True)
    assert s_def == pytest.approx(50.0 / 1.35, rel=1e-6)
    assert s_rob == pytest.approx(50.0 / 1.349, rel=1e-6)


def test_regime_indicators_never_refit_in_either_mode():
    """Regime indicators stay {0.5, 0.5} in BOTH modes (fixed affine maps)."""
    arrays = {"RegimeIsHigh": np.array([0.0, 1.0] * 200, dtype=np.float64)}
    for robust in (False, True):
        out = compute_norm_stats_from_arrays(arrays, robust=robust)
        assert out["RegimeIsHigh"] == TERMINAL_NORM_STATS["RegimeIsHigh"]


def test_from_data_robust_param_forces_robust():
    """compute_norm_stats_from_data(robust=True) forces robust for a STANDARD-method
    column; default keeps it standard."""
    import pandas as pd
    rng = np.random.RandomState(3)
    # RealizedVol30m is a frozen-STANDARD column.
    df = pd.DataFrame({"RealizedVol30m": np.abs(rng.randn(500)) * 1e-3})
    default = compute_norm_stats_from_data(df)
    forced = compute_norm_stats_from_data(df, robust=True)
    assert default["RealizedVol30m"][2] == "standard"
    assert forced["RealizedVol30m"][2] == "robust"


def test_too_few_samples_falls_back_to_frozen_in_robust_mode():
    """<10 finite samples -> frozen constant, even in robust mode."""
    out = compute_norm_stats_from_arrays({"VIXSpot": np.array([14.0, 15.0, 16.0])},
                                         robust=True)
    assert out["VIXSpot"] == TERMINAL_NORM_STATS["VIXSpot"]


def test_degenerate_iqr_uses_std_fallback_not_zero_scale():
    """A boundary-degenerate terminal (>50% mass on one value -> IQR=0) must NOT get
    a 0 scale (which zeroes the terminal). It keeps the robust median center and
    falls back to std as the scale. Mirrors GridReliability (1.0 on ~94% of bars)."""
    # 94% ones, 6% spread below — median=p25=p75=1.0 so IQR=0, but std>0.
    arr = np.concatenate([np.ones(940), np.linspace(0.0, 0.9, 60)]).astype(np.float64)
    assert np.percentile(arr, 75) == np.percentile(arr, 25)  # IQR is exactly 0
    c, s, m = _fit_one_stat(arr, "standard", robust=True)
    assert c == pytest.approx(1.0)               # robust median center preserved
    assert s == pytest.approx(round(float(arr.std()), 6), abs=1e-6)  # std-scale fallback
    assert s > 0.05, "degenerate fallback produced a near-zero scale"
    assert m == "robust_std_fallback"
    # And normalize() with these stats is non-degenerate (a low value maps off-zero).
    z_low = (0.0 - c) / s
    assert abs(z_low) > 1.0


def test_min_scale_floor_survives_rounding():
    """The min-scale floor must survive round(.,6) — a bare 1e-12 floor rounds to
    0.0 and trips normalize()'s guard. A truly-constant terminal gets _MIN_SCALE."""
    const = np.full(500, 0.5, dtype=np.float64)  # constant -> IQR=0 AND std=0
    c, s, m = _fit_one_stat(const, "standard", robust=True)
    assert s >= 1e-6 and round(s, 6) > 0.0, f"floor rounded to {s}"
    assert m == "robust"  # std also ~0 -> stays robust-tagged (genuinely constant)


# ---------------------------------------------------------------------------
# (5) codegen parity — QC uses the SAME robust per-fold stats
# ---------------------------------------------------------------------------

def test_codegen_consumes_norm_stats_positionally():
    """L3 codegen reads the per-fold stats positionally (center, scale) and ignores
    the method tag — so the robust per-fold stats flow into QC UNCHANGED (P0-6
    single-source). This is the parity guarantee for ITEM #4: no codegen change
    needed because robust only changes center/scale values codegen already passes
    through."""
    from layer3 import codegen
    src = inspect.getsource(codegen)
    # The override loop builds {name: (vals[0], vals[1])} from the passed norm_stats.
    assert "vals[0], vals[1]" in src
    # And the frozen base is also taken positionally.
    assert "(v[0], v[1])" in src
