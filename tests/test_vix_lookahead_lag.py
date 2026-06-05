"""VIX 1-day lookahead remediation (2026-05-31).

The L1 collector stored day-D's VIX *close* (research_collector.py:666), and the
daily VIX terminals are daily-constant. Using day-D's close for an intraday day-D
entry decision is a 1-trading-day LOOKAHEAD: a live QC backtest only has the prior
session's close intraday. `prepare_terminal_data(lag_daily_vix=True)` lags the
daily VIX-family terminals one session, removing the lookahead AND creating
proxy<->QC parity (QC's VIXChange[D] == the proxy's old VIXChange[D-1]).

Measured impact: bcc_f2 Sharpe +2.65 -> -0.73 once the peek is removed — the
entire apparent edge was lookahead — and the proxy's entry days shift from
Jan 2/7/10 to Jan 3/8/13, matching the real QC backtest.
"""
import numpy as np
import pandas as pd
import pytest

from layer2.evaluator_vectorized import _session_lag_daily, prepare_terminal_data


def _frame(daily_vix, daily_slope=None, bars_per_day=4):
    """Synthetic minute frame: each day is daily-constant in the VIX family."""
    rows = []
    for d, v in enumerate(daily_vix):
        date = f"2025-02-{d+3:02d}"
        for b in range(bars_per_day):
            rows.append({
                "date": date,
                "VIXSpot": float(v),
                "VIXTermSlope": float(daily_slope[d]) if daily_slope else 0.0,
                "SPXClose": 5000.0 + d * 10 + b,
                "ATM_IV": 0.15,
                "MinutesToClose": float(390 - b),
                "RawSpread": 0.5,
                "RealizedVol30m": 0.1,
            })
    return pd.DataFrame(rows)


def test_session_lag_daily_shifts_one_session():
    """Each day's value becomes the PRIOR day's; the first day keeps its own."""
    dates = np.array(["d1", "d1", "d2", "d2", "d2", "d3"])
    vals = np.array([10., 10., 18., 18., 18., 16.])
    out = _session_lag_daily(vals, dates)
    assert list(out) == [10., 10., 10., 10., 10., 18.], (
        "day1 keeps 10; day2 -> 10 (day1); day3 -> 18 (day2)"
    )


def test_vixspot_lagged_one_session():
    df = _frame([10., 18., 16., 14.])
    td = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=True)
    # per-day VIXSpot (first bar of each 4-bar day)
    per_day = td["VIXSpot"][::4]
    assert list(per_day) == [10., 10., 18., 16.], "VIXSpot must be lagged one session"


def test_vixchange_lag_equals_unlagged_prior_day():
    """The parity property: lagged VIXChange[D] == unlagged VIXChange[D-1].
    This is exactly why the lagged proxy matches the realistic QC value."""
    df = _frame([10., 18., 16., 14.])
    lagged = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=True)["VIXChange"][::4]
    unlag = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=False)["VIXChange"][::4]
    # day D (lagged) == day D-1 (unlagged), for D >= 1
    for d in range(1, 4):
        assert lagged[d] == pytest.approx(unlag[d-1]), (
            f"lagged VIXChange[{d}] ({lagged[d]}) must equal unlagged[{d-1}] ({unlag[d-1]})"
        )
    # and the first lagged day is 0 (no prior move visible)
    assert lagged[0] == pytest.approx(0.0)


def test_vixtermslope_lagged():
    df = _frame([10., 18., 16., 14.], daily_slope=[0.5, -0.3, 0.2, 0.1])
    td = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=True)
    per_day = td["VIXTermSlope"][::4]
    assert per_day[0] == pytest.approx(0.5)   # first day unchanged
    assert per_day[1] == pytest.approx(0.5)   # day2 -> day1
    assert per_day[2] == pytest.approx(-0.3)  # day3 -> day2


def test_vixmean5d_not_double_lagged():
    """VIXMean5d is already lookahead-free (prior 5 days). It must read the
    ORIGINAL closes, NOT the lagged column — otherwise it would be one day
    staler than necessary."""
    vix = [10., 11., 12., 13., 14., 15., 99.]  # day index 0..6
    df = _frame(vix)
    td = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=True)
    per_day = td["VIXMean5d"][::4]
    # day 6 uses ORIGINAL days 1..5 = mean(11,12,13,14,15) = 13.0
    # (double-lagged would use days 0..4 = mean(10..14) = 12.0)
    assert per_day[6] == pytest.approx(13.0), (
        f"VIXMean5d[day6] must use ORIGINAL closes days1-5 (13.0), got {per_day[6]}"
    )


def test_lag_off_preserves_legacy_behavior():
    """lag_daily_vix=False must reproduce the pre-fix (lookahead) values, so the
    flag cleanly isolates the change."""
    df = _frame([10., 18., 16.])
    td = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=False)
    per_day = td["VIXSpot"][::4]
    assert list(per_day) == [10., 18., 16.], "lag=False keeps today's (lookahead) VIX"


# --- residual first-slice-day lookahead fix (code review 2026-05-31) -----------

def test_session_lag_prior_value_used_for_first_day():
    """With prior_value given, the FIRST day uses it (the embargo-gap prior close)
    instead of keeping its own same-day value."""
    dates = np.array(["d1", "d1", "d2", "d2"])
    vals = np.array([18., 18., 16., 16.])
    out = _session_lag_daily(vals, dates, prior_value=15.0)
    assert list(out) == [15., 15., 18., 18.], "day1 must use the prior session (15)"


def test_vix_prior_fixes_first_slice_day():
    """prepare_terminal_data(vix_prior=...) must make the FIRST slice day use the
    true prior-session close, not its own (lookahead) close."""
    df = _frame([18., 16., 14.])  # a slice whose true prior session closed at 15
    td = prepare_terminal_data(df, normalize_terminals=False, lag_daily_vix=True,
                               vix_prior={"VIXSpot": 15.0})
    per_day = td["VIXSpot"][::4]
    assert per_day[0] == 15.0, "first slice day must use the supplied prior (15), not 18"
    assert per_day[1] == 18.0 and per_day[2] == 16.0


def test_split_by_date_returns_vix_prior():
    """split_by_date must expose the prior-session VIX for each slice (from the
    embargo-gap session in the full corpus), so callers can close the first-day
    lookahead without recomputing it."""
    from layer2.io import split_by_date
    dates = [f"2024-01-{d:02d}" for d in range(1, 26)]  # 25 trading days
    rows = []
    for i, d in enumerate(dates):
        for b in range(3):
            rows.append({"date": d, "VIXSpot": 10.0 + i, "VIXTermSlope": 0.0,
                         "SPXClose": 5000.0, "ATM_IV": 0.15})
    df = pd.DataFrame(rows)
    # train ends day 10, embargo 5 -> val starts day 16 (index 15); prior = day 15
    splits = split_by_date(df, train_end="2024-01-10", val_end="2024-01-20",
                           embargo_days=5)
    assert "vix_prior" in splits
    val_first = splits["val"]["date"].iloc[0]
    vi = dates.index(val_first)
    # prior-session VIXSpot == the value of the immediately-preceding corpus day
    assert splits["vix_prior"]["val"]["VIXSpot"] == pytest.approx(10.0 + (vi - 1))
