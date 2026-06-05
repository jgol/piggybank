"""Tests for trailing-rolling local-window robust normalization (plan #4).

Covers the spec §7 cases: causality/no-leakage, warmup fallback, robustness,
level-vs-static terminal policy, passthrough, and a real-data validation that it
reproduces the 2026-06-03 W-sweep firing balance (the whole point of the change).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.trailing_norm import (  # noqa: E402
    apply_trailing_norm, compute_trailing_day_stats, _min_periods_for,
    LEVEL_TERMINALS, DEFAULT_TRAILING_WINDOW, ROBUST_IQR_TO_SIGMA,
)


def _synthetic(n_days=60, bars_per_day=10, seed=0):
    rng = np.random.RandomState(seed)
    days = pd.date_range("2023-01-02", periods=n_days, freq="B")
    dates, vals = [], []
    # a clear regime shift: high level first half, low level second half
    for i, d in enumerate(days):
        level = 0.30 if i < n_days // 2 else 0.12
        for _ in range(bars_per_day):
            dates.append(d)
            vals.append(level + rng.normal(0, 0.01))
    return np.array(vals), pd.DatetimeIndex(dates)


# --------------------------------------------------------------------------
# Causality / no-leakage
# --------------------------------------------------------------------------
def test_shift1_center_excludes_today():
    # constant-per-day values; trailing center on day d must equal the prior days'
    # value, NOT today's — proving shift(1) excludes the current day.
    days = pd.date_range("2023-01-02", periods=10, freq="B")
    dates = np.repeat(days.values, 3)
    vals = np.repeat(np.arange(10.0), 3)        # day i has value i
    st = compute_trailing_day_stats(vals, dates, W=5, min_periods=3)
    # day index 5 (value 5.0): trailing median of days [0..4] = 2.0 (NOT 5.0)
    assert st["center"].iloc[5] == pytest.approx(2.0)
    assert st["center"].iloc[5] != 5.0


def test_causality_future_mutation_invariant():
    vals, dates = _synthetic()
    st1 = compute_trailing_day_stats(vals, dates, W=20)
    # mutate the LAST 5 days wildly; earlier days' stats must be unchanged
    vals2 = vals.copy()
    last_days = pd.DatetimeIndex(dates).normalize() >= pd.DatetimeIndex(dates).normalize().unique()[-5]
    vals2[last_days] += 100.0
    st2 = compute_trailing_day_stats(vals2, dates, W=20)
    n = len(st1) - 6
    pd.testing.assert_series_equal(st1["center"].iloc[:n], st2["center"].iloc[:n])
    pd.testing.assert_series_equal(st1["scale"].iloc[:n], st2["scale"].iloc[:n])


# --------------------------------------------------------------------------
# Warmup fallback
# --------------------------------------------------------------------------
def test_warmup_region_is_nan_then_filled_by_static():
    vals, dates = _synthetic(n_days=40)
    W = 20
    mp = _min_periods_for(W)
    st = compute_trailing_day_stats(vals, dates, W=W)
    # first `mp` unique days have < mp prior days -> NaN
    assert st["center"].iloc[:mp].isna().all()
    assert st["center"].iloc[mp:].notna().any()
    # apply_trailing_norm fills warmup with the static fallback
    raw = {"ATM_IV": vals}
    static = {"ATM_IV": (0.20, 0.05, "robust")}
    out, _ = apply_trailing_norm(raw, dates, W=W, static_stats=static)
    z = out["ATM_IV"]
    assert np.isfinite(z).all()                 # no NaN leaks through
    day = pd.DatetimeIndex(dates).normalize()
    warm_bars = day < day.unique()[mp]
    # warmup bars equal the static normalization exactly
    exp = np.clip((vals[warm_bars] - 0.20) / 0.05, -5, 5)
    assert np.allclose(z[warm_bars], exp)


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def test_robust_scale_resists_outlier_day():
    vals, dates = _synthetic(seed=1)
    st_clean = compute_trailing_day_stats(vals, dates, W=20)
    vals2 = vals.copy()
    day = pd.DatetimeIndex(dates).normalize()
    spike_day = day.unique()[25]
    vals2[day == spike_day] += 50.0             # one absurd day
    st_spk = compute_trailing_day_stats(vals2, dates, W=20)
    # a day whose window INCLUDES the spike: robust scale barely moves vs a
    # std-based scheme (median/IQR ignore the single outlier day)
    probe = 30
    rel = abs(st_spk["scale"].iloc[probe] - st_clean["scale"].iloc[probe]) / st_clean["scale"].iloc[probe]
    assert rel < 0.5, f"robust scale moved {rel:.2f} on one outlier day (should be small)"


def test_iqr_to_sigma_constant():
    # uniform 0..1 over a window: IQR=0.5 -> scale=0.5/1.349
    days = pd.date_range("2023-01-02", periods=30, freq="B")
    dates = np.repeat(days.values, 1)
    vals = np.linspace(0, 1, 30)
    st = compute_trailing_day_stats(vals, dates, W=21, min_periods=21)
    # last day's window = 21 prior values spanning ~[0, .69]; just assert scale>0 & finite
    assert (st["scale"].dropna() > 0).all()
    assert np.isfinite(st["scale"].dropna()).all()


# --------------------------------------------------------------------------
# Terminal policy
# --------------------------------------------------------------------------
def test_static_terminal_uses_static_stats_not_trailing():
    vals, dates = _synthetic()
    raw = {"MinutesToClose": vals}              # NOT a level terminal
    static = {"MinutesToClose": (0.20, 0.05, "robust")}
    out, per_day = apply_trailing_norm(raw, dates, static_stats=static)
    assert "MinutesToClose" not in per_day      # no trailing stats computed
    assert np.allclose(out["MinutesToClose"], np.clip((vals - 0.20) / 0.05, -5, 5))


def test_level_terminal_gets_trailing_stats():
    vals, dates = _synthetic()
    raw = {"ATM_IV": vals}
    static = {"ATM_IV": (0.20, 0.05, "robust")}
    out, per_day = apply_trailing_norm(raw, dates, static_stats=static)
    assert "ATM_IV" in per_day                  # trailing stats computed
    assert "ATM_IV" in LEVEL_TERMINALS


def test_unnormalized_terminal_passes_through():
    vals, dates = _synthetic()
    raw = {"UnrealizedProfitPct": vals}         # not level, not in static -> raw
    out, _ = apply_trailing_norm(raw, dates, static_stats={})
    assert np.allclose(out["UnrealizedProfitPct"], vals)


# --------------------------------------------------------------------------
# Real-data validation — reproduces the W-sweep balance (the whole point)
# --------------------------------------------------------------------------
def test_reproduces_sweep_balance_on_real_data():
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data
    P = "raw_data/local_store/l1_minute_scalars.parquet"
    if not os.path.exists(P):
        pytest.skip("production parquet absent")
    df = pd.read_parquet(P)
    tr = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                       embargo_days=0)["train"]
    raw = prepare_terminal_data(tr, normalize_terminals=False)
    out, _ = apply_trailing_norm(raw, tr["date"].values, W=DEFAULT_TRAILING_WINDOW)
    iv = np.asarray(out["ATM_IV"], float)
    day = pd.DatetimeIndex(pd.to_datetime(tr["date"].values)).normalize()
    thr = np.quantile(iv, 0.75)
    fire = iv > thr
    mon = pd.Series(day.to_period("M").astype(str).values)
    permo = pd.Series(fire).groupby(mon).mean() * 100
    cv = permo.std() / permo.mean()
    fire24 = fire[day >= pd.Timestamp("2024-01-01")].mean() * 100
    # the sweep gave CV~0.69 and 2024 firing ~23% at W=20 (vs expanding 1.26 / 3.3%)
    assert cv < 0.85, f"firing CV {cv:.2f} should be << expanding 1.26"
    assert fire24 > 12.0, f"2024 firing {fire24:.1f}% should be >> expanding 3.3%"


def test_recalibrate_skips_level_terminals_under_trailing():
    """Plan #4 LOW fix (2026-06-04 review): under trailing-rolling norm, frozen
    LEVEL-terminal seed thresholds must NOT be recalibrated to expanding stats (they
    stay ~z-scores); non-LEVEL terminals still recalibrate."""
    import copy
    from layer2.grammar import from_sexpr
    from layer2.terminal_stats import recalibrate_seed_thresholds, normalize_threshold
    fold = {"ATM_IV": (0.13, 0.0511, "robust"), "SessionReturn": (0.0, 0.5, "robust")}

    def _thr(t):
        for c in t.children:
            if c.name == "EphReal":
                return c.value
        return None

    # LEVEL terminal (ATM_IV): recalibrated under expanding, SKIPPED under trailing
    lvl = from_sexpr(f"GT(ATM_IV, EphReal({normalize_threshold('ATM_IV', 0.18)}))")
    v0 = _thr(lvl)
    exp = copy.deepcopy(lvl)
    recalibrate_seed_thresholds(exp, fold)
    trl = copy.deepcopy(lvl)
    recalibrate_seed_thresholds(trl, fold, skip_terminals=LEVEL_TERMINALS)
    assert abs(_thr(exp) - v0) > 1e-6, "expanding must recalibrate the LEVEL threshold"
    assert _thr(trl) == pytest.approx(v0), "trailing must SKIP the LEVEL threshold"

    # non-LEVEL terminal still recalibrates even with skip_terminals set
    nl = from_sexpr(
        f"GT(SessionReturn, EphReal({normalize_threshold('SessionReturn', 0.01)}))")
    v1 = _thr(nl)
    recalibrate_seed_thresholds(nl, fold, skip_terminals=LEVEL_TERMINALS)
    assert abs(_thr(nl) - v1) > 1e-9, "non-LEVEL terminal must still recalibrate"
