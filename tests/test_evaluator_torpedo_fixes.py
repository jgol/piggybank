"""Regression tests for 6 audit-confirmed torpedoes in evaluator_vectorized.py.

Each torpedo had a verified /tmp/gate_hunt*.py repro; these tests pin the fix as a
persistent guard. Run with:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_NUM_THREADS=1 /opt/anaconda3/bin/python -m pytest \
        tests/test_evaluator_torpedo_fixes.py -v

The six fixes, all confined to layer2/evaluator_vectorized.py:

  #1 B3      churning gate (>80% early SIGNAL exits)      -> profit-conditional
  #2 B3-twin early-stop churn gate (>80% early stop exits)-> profit-conditional
  #3 B4      clock-exit gate (signal-exit std < 3 bars)   -> profit-conditional
  #4 M2      _net_pos_value structural-width clamp for defined-risk verticals
  #5 M3      Sortino independent of Sharpe when < _MIN_DOWNSIDE_OBS losing days
  #6 L3      avg_position_size reads the SIGNAL bar (entry_bar-1), not the fill bar

Design principle for the three gate tests (#1/#2/#3): the gate sentinel is set on
`raw.sharpe` inside `vectorized_backtest`. So the precise, mechanism-level assertion
is: under a strategy whose trades DO trigger the gate's frac/std condition,
  - a PROFITABLE one is NOT sentineled (raw.sharpe finite and > -1e5), and
  - a LOSING one with the SAME shape IS still sentineled (raw.sharpe <= -1e5).
We additionally assert the gate's triggering condition actually fires, so the test
proves the *guard*, not merely that some strategy passes.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_NUM_THREADS", "1")

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import layer2.evaluator_vectorized as ev
from layer2.evaluator_vectorized import (
    prepare_terminal_data,
    vectorized_backtest,
    _net_pos_value,
    _estimate_surface_params,
    GRID_IV_COLUMNS,
    ENTRY_CUTOFF_MTC,
    _MIN_DOWNSIDE_OBS,
    _MAX_SORTINO,
)
from layer2.grammar import from_sexpr
from layer2.terminal_stats import compute_norm_stats_from_data
from layer2.templates import (
    Leg,
    bull_put_credit_standard,
    ratio_put_backspread_base,
)
from layer2.io import PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS

from tests.test_gp_positive_control import build_planted_edge_fixture

SENTINEL_RAW = -1e5  # raw.sharpe <= this == a post-proc gate sentinel (-1e6) fired
MIN_BARS = 15        # vectorized_backtest default min_bars_in_trade


# ---------------------------------------------------------------------------
# Shared fixture builders (adapted from the /tmp/gate_hunt*.py repros)
# ---------------------------------------------------------------------------
def _common_row(date, w, mtc_top, spot, vix, rv, rng, extra=None):
    row = {
        "date": date, "window_idx": w, "bar_position": w,
        PRICE_COLUMN: float(spot), "ATM_IV": 0.20,
        "MinutesToClose": float(mtc_top - w), "VIXSpot": float(vix),
        "VIXTermSlope": -0.5, "RawSpread": 0.20,
        "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
        "RealizedVol30m": float(rv), "PredRV15": 0.30, "PredRV30": 0.30,
        "PredRegime": 0, "PredSpread": 0.0,
    }
    if extra:
        row.update(extra)
    for col in TYPED_VECTOR_COLUMNS:
        row[col] = rng.standard_normal(384).astype(np.float32).tolist()
    for col in REGIME_PROB_COLUMNS:
        row[col] = float(rng.uniform())
    return row


def _build_fast_move_fixture(n_days=140, bars=120, seed=11, direction=-1.0):
    """Half the days: a FAST early move (±4% over first ~12 bars) then flat; other
    half: flat (debit decay). A ratio put backspread entered at open on a DOWN move
    profits; an UP move makes it lose. `direction=-1` = profitable for the backspread;
    `direction=+1` = the same shape but a LOSER (move goes the wrong way)."""
    rng = np.random.default_rng(seed)
    day_rng = np.random.default_rng(seed + 1)
    is_move = day_rng.random(n_days) < 0.5
    start = pd.Timestamp("2024-01-02")
    mtc_top = (ENTRY_CUTOFF_MTC + 1.0) + (bars - 1)
    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        move = bool(is_move[d])
        spot = 5000.0
        open_spot = spot
        path_rng = np.random.default_rng(seed * 100003 + d)
        vix = 30.0 if move else 12.0
        rv = 8.0e-4 if move else 2.0e-4
        for w in range(bars):
            if move and w < 12:
                step = (direction * 0.04 * 5000.0 / 12.0) + path_rng.normal(0.0, 0.3)
            else:
                step = path_rng.normal(0.0, 0.05)
            spot += step
            sess = (spot - open_spot) / open_spot
            rows.append(_common_row(date, w, mtc_top, spot, vix, rv, rng,
                                    extra={"SessionReturn": float(sess)}))
    return pd.DataFrame(rows), is_move


def _build_move_fixture_lowmtc(n_days=140, bars=120, seed=11, direction=-1.0):
    """Move days: a move over the WHOLE day so a near-close exit still captures the
    gain; mtc descends low (bottom=2) so a consistent-time SIGNAL exit at mtc just
    above EOD_FORCE_CLOSE_MTC lands near close. `direction=-1` profitable for a put
    backspread, `+1` a same-shaped loser."""
    rng = np.random.default_rng(seed)
    day_rng = np.random.default_rng(seed + 1)
    is_move = day_rng.random(n_days) < 0.5
    start = pd.Timestamp("2024-01-02")
    mtc_top = 2.0 + (bars - 1)
    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        move = bool(is_move[d])
        spot = 5000.0
        open_spot = spot
        path_rng = np.random.default_rng(seed * 100003 + d)
        vix = 30.0 if move else 12.0
        rv = 8.0e-4 if move else 2.0e-4
        drift = (direction * 0.04 * 5000.0) / bars if move else 0.0
        for w in range(bars):
            step = drift + path_rng.normal(0.0, 0.3 if move else 0.05)
            spot += step
            rows.append(_common_row(date, w, mtc_top, spot, vix, rv, rng,
                                    extra={"SessionReturn": float((spot - open_spot) / open_spot)}))
    return pd.DataFrame(rows), is_move


def _within_day_pos(df):
    dates = df["date"].values
    wdp = np.zeros(len(df), dtype=np.int64)
    pos = 0
    prev = None
    for i, d in enumerate(dates):
        if prev is not None and d != prev:
            pos = 0
        wdp[i] = pos
        pos += 1
        prev = d
    return wdp


def _run_raw(df, td, tmpl, e, x, s="EphReal(0.5)", dt=None, warmup=15):
    entry, exit_t, size_t = from_sexpr(e), from_sexpr(x), from_sexpr(s)
    delta_t = from_sexpr(dt) if dt else None
    return vectorized_backtest(entry, exit_t, size_t, df, tmpl, delta_tree=delta_t,
                               terminal_data=td, warmup_bars=warmup)


def _early_signal_frac(raw):
    if not raw.trades:
        return 0.0
    n = sum(1 for t in raw.trades
            if t.exit_bar - t.entry_bar <= MIN_BARS + 1 and t.exit_reason == "signal")
    return n / len(raw.trades)


def _early_stop_frac(raw):
    if not raw.trades:
        return 0.0
    n = sum(1 for t in raw.trades
            if t.exit_bar - t.entry_bar <= MIN_BARS + 1 and t.exit_reason == "stop_loss")
    return n / len(raw.trades)


def _signal_exit_std(raw, wdp, n_bars):
    sig = [t for t in raw.trades if t.exit_reason == "signal"]
    if len(sig) < 5:
        return None
    return float(np.std([wdp[min(t.exit_bar, n_bars - 1)] for t in sig]))


# ===========================================================================
# FIX #1 — B3 churning gate (>80% early SIGNAL exits) is now profit-conditional
# ===========================================================================
def test_fix1_b3_churn_gate_spares_profitable_but_kills_loser():
    """A debit put backspread that catches a FAST down-move and locks the gain via a
    reversal/profit SIGNAL exit within min_bars+1 bars triggers the >80% early-signal
    churn gate. PROFITABLE => must NOT be sentineled. The SAME shape on UP-moves
    (a net loser) => must STILL be sentineled.

    Before: `if min_bar_frac > 0.80: sharpe = -1e6` (profitability-blind).
    After:  `if min_bar_frac > 0.80 and sharpe <= 0.0: sharpe = -1e6`.
    """
    tmpl = ratio_put_backspread_base()
    # Profit-take exit AT the move's extreme so the exit fires fast (< 16 bars).
    from layer2.terminal_stats import normalize_threshold
    thr = normalize_threshold("SessionReturn", -0.030)
    entry = "GT(RealizedVol30m, EphReal(0.0))"          # enter on move (high-vol) days
    exit_ = f"GT(SessionReturn, EphReal({thr:.4f}))"     # exit when down-move deep -> fast

    # --- profitable variant (down moves) ---
    dfw, _ = _build_fast_move_fixture(direction=-1.0)
    norm = compute_norm_stats_from_data(dfw)
    tdw = prepare_terminal_data(dfw, norm_stats_override=norm, lag_daily_vix=False)
    win = _run_raw(dfw, tdw, tmpl, entry, exit_, dt="EphReal(0.5)")

    assert len(win.trades) >= 5, "need >=5 trades for the churn gate to be active"
    frac_w = _early_signal_frac(win)
    assert frac_w > 0.80, (
        f"profitable variant must TRIGGER the churn gate (>80% early signal exits); "
        f"got frac={frac_w:.3f}, reasons={dict(Counter(t.exit_reason for t in win.trades))}")
    total_pnl_w = sum(t.pnl for t in win.trades)
    assert total_pnl_w > 0.0 and win.sharpe > 0.0, (
        f"profitable variant must be net positive; pnl={total_pnl_w:.1f} sharpe={win.sharpe:.3f}")
    assert win.sharpe > SENTINEL_RAW and math.isfinite(win.sharpe), (
        f"FIX #1 FAILED: profitable fast-signal-exit backspread was sentineled "
        f"(sharpe={win.sharpe}) despite triggering the churn gate while net profitable")

    # --- losing variant (same shape, up moves) ---
    dfl, _ = _build_fast_move_fixture(direction=+1.0)
    norml = compute_norm_stats_from_data(dfl)
    tdl = prepare_terminal_data(dfl, norm_stats_override=norml, lag_daily_vix=False)
    lose = _run_raw(dfl, tdl, tmpl, entry, exit_, dt="EphReal(0.5)")
    if _early_signal_frac(lose) > 0.80 and sum(t.pnl for t in lose.trades) < 0:
        assert lose.sharpe <= SENTINEL_RAW, (
            f"churn gate must STILL sentinel an UNPROFITABLE early-signal churner; "
            f"sharpe={lose.sharpe} pnl={sum(t.pnl for t in lose.trades):.1f}")


def test_fix1_b3_churn_gate_unit_loser_sentineled():
    """Direct check that an UNPROFITABLE early-signal churner is STILL hard-gated:
    the SAME backspread fast-signal-exit shape as the headline test, but on UP-moves
    so the put backspread LOSES. It still trips the >80% early-signal churn gate and,
    being a net loser, must be sentineled (the `and sharpe <= 0.0` guard does NOT
    spare it)."""
    from layer2.terminal_stats import normalize_threshold
    thr = normalize_threshold("SessionReturn", -0.030)
    entry = "GT(RealizedVol30m, EphReal(0.0))"
    exit_ = f"GT(SessionReturn, EphReal({thr:.4f}))"
    dfl, _ = _build_fast_move_fixture(direction=+1.0)  # up moves -> backspread loses
    norm = compute_norm_stats_from_data(dfl)
    td = prepare_terminal_data(dfl, norm_stats_override=norm, lag_daily_vix=False)
    raw = _run_raw(dfl, td, ratio_put_backspread_base(), entry, exit_, dt="EphReal(0.5)")
    assert _early_signal_frac(raw) > 0.80, (
        f"loser must TRIGGER the churn gate; frac={_early_signal_frac(raw):.3f}")
    assert sum(t.pnl for t in raw.trades) < 0.0, "loser must be net negative"
    assert raw.sharpe <= SENTINEL_RAW, (
        f"FIX #1 regression: an UNPROFITABLE early-signal churner must STILL be "
        f"sentineled; sharpe={raw.sharpe}")


# ===========================================================================
# FIX #2 — B3-twin early-stop-loss churn gate is now profit-conditional
# ===========================================================================
def _stop_churn_df(n_days=60, bars=24, seed=21, dayslope=-0.06):
    """A whole-day steep monotonic DROP so a short-put-credit position goes deep ITM
    and stop-losses EARLY (< 16 bars) on essentially every day -> >80% early stop
    exits, net loss. (A steady all-day drop trips the stop within a few bars of every
    entry; verified early_stop_frac == 1.0.)"""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-03-01")
    mtc_top = (ENTRY_CUTOFF_MTC + 1.0) + (bars - 1)
    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        spot = 5000.0
        path = np.random.default_rng(seed * 7 + d)
        for w in range(bars):
            spot += (dayslope * 5000.0 / bars) + path.normal(0.0, 0.15)
            rows.append(_common_row(date, w, mtc_top, spot, 22.0, 6.0e-4, rng))
    return pd.DataFrame(rows)


def test_fix2_b3twin_early_stop_gate_unit_loser_sentineled():
    """A degenerate entry that systematically stop-losses early (>80% early stop
    exits) and is NET NEGATIVE must STILL be sentineled. (A pure stop exit cannot also
    be net profitable on that trade, so the gate's economic intent is to spare a
    strategy whose net Sharpe is positive — proven at the source by the companion test;
    here we pin that the LOSER is still hard-gated and the predicate carries the
    `and sharpe <= 0.0` guard.)"""
    df = _stop_churn_df()  # steep all-day drop -> early stop on ~every day, net loss
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    tmpl = bull_put_credit_standard()  # short put spread; the down move trips the stop
    raw = _run_raw(df, td, tmpl,
                   "GT(EphReal(1.0), EphReal(0.0))",        # always enter
                   "GT(EphReal(-10.0), EphReal(0.0))")      # never signal-exit -> stop is exit
    frac = _early_stop_frac(raw)
    assert frac > 0.80, (
        f"loser must TRIGGER the early-stop churn gate (>80% early stop exits); "
        f"got frac={frac:.3f}, reasons={dict(Counter(t.exit_reason for t in raw.trades))}")
    assert sum(t.pnl for t in raw.trades) < 0.0, "loser must be net negative"
    assert raw.sharpe <= SENTINEL_RAW, (
        f"FIX #2 regression: an UNPROFITABLE early-stop churner must STILL be "
        f"sentineled; early_stop_frac={frac:.3f} sharpe={raw.sharpe}")


def test_fix2_b3twin_guard_is_profit_conditional_by_source():
    """Source-level guarantee that the early-stop gate carries the profit guard so a
    PROFITABLE strategy cannot be killed by it (the runtime loser case is covered
    above). Asserts the exact `and sharpe <= 0.0` predicate is present on the
    early_stop_frac gate."""
    import inspect
    src = inspect.getsource(ev.vectorized_backtest)
    # locate the early_stop_frac gate and confirm its guard. The profit signal is
    # `_raw_sharpe` (the pre-gate-cascade Sharpe) so the max-hold subtractive
    # penalty cannot bleed a profitable strategy under the gate's profit test.
    assert "early_stop_frac > 0.80 and _raw_sharpe <= 0.0" in src, (
        "FIX #2 missing: early-stop churn gate must be profit-conditional "
        "(`early_stop_frac > 0.80 and _raw_sharpe <= 0.0`)")
    # All three profit-conditional gates must use the mutation-independent
    # _raw_sharpe, NOT the running `sharpe` (which the cascade itself degrades).
    assert "if min_bar_frac > 0.80 and _raw_sharpe <= 0.0:" in src, (
        "FIX #1 must gate on _raw_sharpe (pre-cascade profitability)")
    assert "if _exit_std < 3.0 and _raw_sharpe <= 0.0:" in src, (
        "FIX #3 must gate on _raw_sharpe (pre-cascade profitability)")


# ===========================================================================
# FIX #3 — B4 clock-exit gate (signal-exit std < 3) is now profit-conditional
# ===========================================================================
def test_fix3_b4_clock_exit_gate_spares_profitable_but_kills_loser():
    """A consistent-time ("always close near 15:45") SIGNAL exit has std≈0 and trips
    the clock-exit gate. On a tape where the held position is PROFITABLE by that exit
    time, it must NOT be sentineled; the SAME consistent-time exit on a tape where it
    LOSES must STILL be sentineled."""
    tmpl = ratio_put_backspread_base()
    entry = "GT(RealizedVol30m, EphReal(0.0))"          # enter on move days
    exit_ = "LT(MinutesToClose, EphReal(-0.9))"          # fire at a consistent near-close time

    # --- profitable: down move accrues all day, captured at the consistent exit ---
    dfw, _ = _build_move_fixture_lowmtc(direction=-1.0)
    normw = compute_norm_stats_from_data(dfw)
    tdw = prepare_terminal_data(dfw, norm_stats_override=normw, lag_daily_vix=False)
    win = _run_raw(dfw, tdw, tmpl, entry, exit_, dt="EphReal(0.5)")
    wdp_w = _within_day_pos(dfw)
    std_w = _signal_exit_std(win, wdp_w, len(dfw))

    assert std_w is not None and std_w < 3.0, (
        f"profitable variant must TRIGGER the clock-exit gate (signal-exit std<3); "
        f"got std={std_w}, reasons={dict(Counter(t.exit_reason for t in win.trades))}")
    assert win.sharpe > 0.0 and sum(t.pnl for t in win.trades) > 0.0, (
        f"profitable variant must be net positive; sharpe={win.sharpe:.3f}")
    assert win.sharpe > SENTINEL_RAW and math.isfinite(win.sharpe), (
        f"FIX #3 FAILED: a profitable consistent-time (clock) exit was sentineled "
        f"(sharpe={win.sharpe}) despite being net profitable")

    # --- losing: same consistent-time exit, up move -> backspread loses ---
    dfl, _ = _build_move_fixture_lowmtc(direction=+1.0)
    norml = compute_norm_stats_from_data(dfl)
    tdl = prepare_terminal_data(dfl, norm_stats_override=norml, lag_daily_vix=False)
    lose = _run_raw(dfl, tdl, tmpl, entry, exit_, dt="EphReal(0.5)")
    wdp_l = _within_day_pos(dfl)
    std_l = _signal_exit_std(lose, wdp_l, len(dfl))
    if std_l is not None and std_l < 3.0 and sum(t.pnl for t in lose.trades) < 0:
        assert lose.sharpe <= SENTINEL_RAW, (
            f"clock-exit gate must STILL sentinel an UNPROFITABLE clock exit; "
            f"std={std_l:.3f} sharpe={lose.sharpe} pnl={sum(t.pnl for t in lose.trades):.1f}")


def test_fix3_b4_clock_exit_unit_loser_sentineled():
    """Unit: a money-LOSING strategy that exits at a single consistent time-of-day
    (std≈0 via a SIGNAL exit) must STILL be hard-gated. Uses the same low-mtc geometry
    as the headline test but with UP moves (`direction=+1`) so the put backspread
    loses while exiting at the same within-day bar every day (std==0)."""
    tmpl = ratio_put_backspread_base()
    dfl, _ = _build_move_fixture_lowmtc(direction=+1.0)
    norm = compute_norm_stats_from_data(dfl)
    td = prepare_terminal_data(dfl, norm_stats_override=norm, lag_daily_vix=False)
    raw = _run_raw(dfl, td, tmpl,
                   "GT(RealizedVol30m, EphReal(0.0))",
                   "LT(MinutesToClose, EphReal(-0.9))",   # consistent near-close SIGNAL exit
                   dt="EphReal(0.5)")
    wdp = _within_day_pos(dfl)
    std = _signal_exit_std(raw, wdp, len(dfl))
    assert std is not None and std < 3.0, (
        f"loser must TRIGGER the clock-exit gate (signal-exit std<3); got std={std}, "
        f"reasons={dict(Counter(t.exit_reason for t in raw.trades))}")
    assert sum(t.pnl for t in raw.trades) < 0.0, "loser must be net negative"
    assert raw.sharpe <= SENTINEL_RAW, (
        f"FIX #3 regression: an UNPROFITABLE clock exit must STILL be sentineled; "
        f"std={std:.3f} sharpe={raw.sharpe}")


# ===========================================================================
# FIX #4 — M2 structural-width clamp for defined-risk verticals
# ===========================================================================
def _nonmonotone_grid(atm_iv=0.20, spike_idx=2, spike=2.5):
    """A NON-MONOTONE 11-point grid IV array (one strike's IV is corrupted high),
    which is exactly the surface pathology that mismarks a vertical beyond its width.
    Returns (grid_array, rho_t, beta_t)."""
    g = np.full(len(GRID_IV_COLUMNS), atm_iv, dtype=np.float64)
    g[spike_idx] = spike  # e.g. corrupt the 25-delta put IV to 250%
    rho_t, beta_t = _estimate_surface_params(g, atm_iv)
    return g, rho_t, beta_t


def test_fix4_m2_credit_vertical_marks_within_width():
    """A defined-risk PUT CREDIT vertical (short -0.25Δ put, long -0.10Δ put) priced on
    a corrupted non-monotone grid must net within [-width, 0] (credit, bounded by the
    strike width) — NEVER beyond its structural width, and NEVER flipped to a debit."""
    legs = bull_put_credit_standard().legs  # short put @-0.25, long put @-0.10
    spot, mtc = 5000.0, 180.0
    strikes = [4900.0, 4850.0]  # short higher (closer), long lower (wing): width=50
    width = abs(strikes[0] - strikes[1])
    # Sweep a range of corrupting spikes at several grid points.
    worst = 0.0
    for spike_idx in range(len(GRID_IV_COLUMNS)):
        for spike in (1.5, 2.5, 5.0):
            g, rho, beta = _nonmonotone_grid(spike_idx=spike_idx, spike=spike)
            v = _net_pos_value(legs, strikes, spot, mtc, 0.20,
                               grid_ivs=g, rho_t=rho, beta_t=beta)
            assert abs(v) <= width + 1e-6, (
                f"FIX #4 FAILED: credit vertical marked beyond width |{v:.4f}| > {width} "
                f"(spike_idx={spike_idx}, spike={spike})")
            assert v <= 1e-6, (
                f"FIX #4 FAILED: credit vertical flipped to a DEBIT (v={v:.4f} > 0) "
                f"on a corrupt grid (spike_idx={spike_idx}, spike={spike})")
            worst = max(worst, abs(v) / width)
    assert worst <= 1.0 + 1e-6


def test_fix4_m2_debit_vertical_marks_within_width():
    """A defined-risk DEBIT call vertical (long lower strike, short higher strike) must
    net within [0, width] on a corrupt grid — never negative (a debit cannot become a
    credit) and never beyond width."""
    legs = (Leg("call", +0.50, qty_sign=+1, ratio=1),   # long lower strike
            Leg("call", +0.20, qty_sign=-1, ratio=1))   # short higher strike (cap)
    spot, mtc = 5000.0, 180.0
    strikes = [5000.0, 5040.0]  # long lower, short higher: width=40 -> debit vertical
    width = abs(strikes[0] - strikes[1])
    for spike_idx in range(len(GRID_IV_COLUMNS)):
        for spike in (1.5, 2.5, 5.0):
            g, rho, beta = _nonmonotone_grid(spike_idx=spike_idx, spike=spike)
            v = _net_pos_value(legs, strikes, spot, mtc, 0.20,
                               grid_ivs=g, rho_t=rho, beta_t=beta)
            assert -1e-6 <= v <= width + 1e-6, (
                f"FIX #4 FAILED: debit vertical out of [0,{width}] band: v={v:.4f} "
                f"(spike_idx={spike_idx}, spike={spike})")


def test_fix4_m2_iron_condor_within_combined_width():
    """A 4-leg iron condor (two verticals) on a corrupt grid: each vertical is clamped
    to its own width, so the net cannot exceed the sum of the two widths (and stays a
    credit, <= 0)."""
    from layer2.templates import iron_condor_standard
    legs = iron_condor_standard().legs
    spot, mtc = 5000.0, 180.0
    # call side: short 5060 / long 5110 (w=50); put side: short 4940 / long 4890 (w=50)
    strikes = [5060.0, 5110.0, 4940.0, 4890.0]
    wcall = abs(strikes[0] - strikes[1])
    wput = abs(strikes[2] - strikes[3])
    for spike_idx in range(len(GRID_IV_COLUMNS)):
        g, rho, beta = _nonmonotone_grid(spike_idx=spike_idx, spike=3.0)
        v = _net_pos_value(legs, strikes, spot, mtc, 0.20,
                           grid_ivs=g, rho_t=rho, beta_t=beta)
        assert abs(v) <= wcall + wput + 1e-6, (
            f"FIX #4 FAILED: IC marked beyond combined width |{v:.4f}| > "
            f"{wcall + wput} (spike_idx={spike_idx})")
        assert v <= 1e-6, f"FIX #4: IC flipped to a debit (v={v:.4f})"


def test_fix4_m2_clamp_is_noop_when_grid_well_behaved():
    """The clamp must NOT change marks on a MONOTONE (well-behaved) grid — it only
    binds on the pathological non-monotone surface. We compare against re-pricing with
    a smooth grid and assert the value is well inside the band (clamp inactive)."""
    legs = bull_put_credit_standard().legs
    spot, mtc = 5000.0, 180.0
    strikes = [4900.0, 4850.0]
    width = abs(strikes[0] - strikes[1])
    # Smooth, monotone put-skew grid (puts higher IV than calls), no spike.
    smooth = np.array([0.30, 0.28, 0.25, 0.23, 0.22, 0.21, 0.205,
                       0.20, 0.198, 0.196, 0.195], dtype=np.float64)
    rho, beta = _estimate_surface_params(smooth, 0.20)
    v = _net_pos_value(legs, strikes, spot, mtc, 0.20,
                       grid_ivs=smooth, rho_t=rho, beta_t=beta)
    assert -width < v < 0.0, (
        f"on a well-behaved grid the BPC must mark as a normal credit strictly inside "
        f"its width; got v={v:.4f} (width={width})")
    # And the clamp did not bind: value strictly above -width (not pinned to the floor).
    assert v > -width + 1e-3


# ===========================================================================
# FIX #5 — M3 Sortino is independent of Sharpe when < _MIN_DOWNSIDE_OBS losers
# ===========================================================================
def _skewed_downside_tape(n_days=60, bars=40, seed=4, nbig=3, downmove=-0.012):
    """A left-skewed tape: many mild up days + a FEW (< _MIN_DOWNSIDE_OBS) down days,
    so the BPC posts a few losing DAYS but not enough to estimate downside dev
    empirically — exactly the M3 branch. Left-skewed daily returns (skew << 0)."""
    rng = np.random.default_rng(seed)
    day_rng = np.random.default_rng(seed + 1)
    big_down = set(day_rng.choice(n_days, size=nbig, replace=False).tolist())
    start = pd.Timestamp("2024-05-01")
    mtc_top = (ENTRY_CUTOFF_MTC + 1.0) + (bars - 1)
    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
        spot = 5000.0
        path = np.random.default_rng(seed * 13 + d)
        if d in big_down:
            drift = (downmove * 5000.0) / bars   # down day -> BPC loss
            vix, rv = 24.0, 7.0e-4
        else:
            drift = (+0.004 * 5000.0) / bars      # mild up day -> BPC small win
            vix, rv = 12.0, 2.0e-4
        for w in range(bars):
            spot += drift + path.normal(0.0, 0.03)
            rows.append(_common_row(date, w, mtc_top, spot, vix, rv, rng))
    return pd.DataFrame(rows)


def _expected_sortino_independent(daily_arr):
    """Recompute the M3 fix's INDEPENDENT few-losing-day Sortino from daily returns:
    semideviation-about-0 of losing days, floored at 1/3 of the OVERALL std, ×√252,
    clipped to ±_MAX_SORTINO. Used to prove the new path is taken (NOT clip(sharpe))."""
    from layer2.evaluator_vectorized import _DOWNSIDE_DEV_STD_FRACTION
    ann = math.sqrt(252.0)
    mean_daily = float(np.mean(daily_arr))
    overall_std = float(np.std(daily_arr)) if len(daily_arr) > 1 else 0.0
    downside = daily_arr[daily_arr < 0]
    emp_dd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 0.0
    dd = max(emp_dd, _DOWNSIDE_DEV_STD_FRACTION * overall_std, 1e-9)
    return float(np.clip(mean_daily / dd * ann, -_MAX_SORTINO, _MAX_SORTINO))


def test_fix5_m3_sortino_independent_of_sharpe_on_skewed_tape():
    """On a left-skewed tape with < _MIN_DOWNSIDE_OBS losing DAYS, Sortino must be a
    genuine, INDEPENDENT downside measure — NOT the old `clip(sharpe)` collinear
    fallback. We pin: (a) Sortino != Sharpe; (b) Sortino is bounded by ±_MAX_SORTINO;
    (c) Sortino EXACTLY equals the magnitude-tied downside-dev formula (proving the new
    path executed and is downside-based), and that this value is NOT clip(sharpe)."""
    df = _skewed_downside_tape()
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    tmpl = bull_put_credit_standard()
    raw = _run_raw(df, td, tmpl,
                   "GT(EphReal(1.0), EphReal(0.0))",       # always enter (hold to close)
                   "LT(MinutesToClose, EphReal(-5.0))")    # never signal-exit
    # Confirm the few-losing-day regime the fix targets (and left skew).
    daily = pd.Series(raw.returns).groupby(df["date"].values).sum().values.astype(np.float64)
    n_losing_days = int((daily < 0).sum())
    assert 0 < n_losing_days < _MIN_DOWNSIDE_OBS, (
        f"fixture must have between 1 and {_MIN_DOWNSIDE_OBS - 1} losing days to "
        f"exercise the M3 branch; got {n_losing_days}")
    assert raw.return_skew < -0.5, (
        f"tape must be left-skewed for a meaningful downside test; skew={raw.return_skew:.2f}")
    assert math.isfinite(raw.sharpe) and raw.sharpe > SENTINEL_RAW, (
        "this hold-to-close strategy should not be gated on this tape")

    # (a) INDEPENDENT of Sharpe (the headline regression: old code returned clip(sharpe)).
    assert raw.sortino != pytest.approx(raw.sharpe, abs=1e-9), (
        f"FIX #5 FAILED: Sortino collinear with Sharpe in the few-losing-day regime "
        f"(sortino={raw.sortino:.6f} == sharpe={raw.sharpe:.6f}) — clip(sharpe) "
        f"fallback not removed")
    # (b) bounded.
    assert abs(raw.sortino) <= _MAX_SORTINO + 1e-9, "Sortino must stay within the cap"
    # (c) EXACTLY the new magnitude-tied downside formula — proves the independent,
    # downside-based path executed (and is distinct from the clipped Sharpe value).
    exp = _expected_sortino_independent(daily)
    assert raw.sortino == pytest.approx(exp, rel=1e-9, abs=1e-9), (
        f"FIX #5: Sortino does not match the independent downside-dev formula "
        f"(got {raw.sortino:.6f}, expected {exp:.6f})")
    clipped_sharpe = float(np.clip(raw.sharpe, -_MAX_SORTINO, _MAX_SORTINO))
    assert abs(exp - clipped_sharpe) > 1e-6, (
        f"sanity: the independent Sortino must differ from clip(sharpe) on this tape "
        f"(indep={exp:.6f}, clip(sharpe)={clipped_sharpe:.6f}) — else the test can't "
        f"distinguish the fix")


def test_fix5_m3_no_sharpe_fallback_in_source():
    """Guard against re-introducing the `sortino = clip(sharpe)` collinear fallback in
    the few-losing-day branch."""
    import inspect
    src = inspect.getsource(ev.vectorized_backtest)
    assert "sortino = float(np.clip(sharpe," not in src, (
        "FIX #5 regression: the few-losing-day branch must NOT fall back to "
        "clip(sharpe) (that makes Sortino collinear with Sharpe)")


# ===========================================================================
# FIX #6 — L3 avg_position_size reads the SIGNAL bar (entry_bar-1), not fill bar
# ===========================================================================
def test_fix6_l3_avg_position_size_uses_signal_bar():
    """avg_position_size must equal the SIGNAL-bar size that was actually used
    (`pending_size = size_signals[entry_bar-1]`), not the fill-bar size. Build a tape
    where the size tree's value at the signal bar differs from the fill bar so the two
    readings are numerically distinguishable, then verify the reported average matches
    the signal-bar reconstruction."""
    df, _ = build_planted_edge_fixture(n_days=60, bars_per_date=80, seed=7)
    norm = compute_norm_stats_from_data(df)
    td = prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)
    tmpl = bull_put_credit_standard()
    # Size tree that VARIES bar-to-bar so signal-bar (entry_bar-1) != fill-bar
    # (entry_bar). MinutesToClose strictly decreases within a day, so a size that
    # depends on it changes every bar. Clamp into (0,1) via the evaluator's own
    # max(0,min(1,.)). Use a normalized terminal scaled so values land in-range.
    entry = "LT(RealizedVol30m, EphReal(0.0))"         # day-selective (calm) entry
    exit_ = "LT(MinutesToClose, EphReal(-5.0))"        # hold to close
    size = "Div(MinutesToClose, EphReal(2.0))"          # varies every bar -> distinguishes

    entry_t, exit_t, size_t = from_sexpr(entry), from_sexpr(exit_), from_sexpr(size)
    raw = vectorized_backtest(entry_t, exit_t, size_t, df, tmpl,
                              terminal_data=td, warmup_bars=15)
    assert raw.trades, "need trades for this assertion"

    # Reconstruct the size_signals array EXACTLY as vectorized_backtest does
    # (same day_mask / within_day_pos, NaN-guard, clip to [0,1]) — see
    # evaluator_vectorized.py lines ~1255-1303.
    from layer2.evaluator_vectorized import VectorizedTreeEvaluator
    n_bars = len(df)
    day_mask = np.ones(n_bars, dtype=np.float64)
    within_day_pos = np.zeros(n_bars, dtype=np.int64)
    dates = df["date"].values
    day_mask[0] = 0.0
    pos = 0
    for i in range(1, n_bars):
        if str(dates[i]) != str(dates[i - 1]):
            day_mask[i] = 0.0
            pos = 0
        else:
            pos += 1
        within_day_pos[i] = pos
    _evtr = VectorizedTreeEvaluator(td, n_bars, day_boundary_mask=day_mask,
                                    within_day_pos=within_day_pos)
    size_signals = _evtr.evaluate(size_t).astype(np.float64)
    size_signals = np.where(np.isnan(size_signals), 0.0, size_signals)
    size_signals = np.clip(size_signals, 0.0, 1.0)

    # Signal-bar reconstruction (the fix): size used = size_signals[entry_bar-1].
    expected = float(np.mean([
        size_signals[max(t.entry_bar - 1, 0)] for t in raw.trades
    ]))
    # Fill-bar (the OLD buggy reading) — must DIFFER, else the test can't tell them apart.
    fillbar = float(np.mean([size_signals[t.entry_bar] for t in raw.trades]))

    assert abs(expected - fillbar) > 1e-9, (
        "test tape failed to make signal-bar and fill-bar sizes differ; cannot "
        "distinguish the fix")
    assert raw.avg_position_size == pytest.approx(expected, abs=1e-9), (
        f"FIX #6 FAILED: avg_position_size={raw.avg_position_size:.6f} does not match "
        f"the SIGNAL-bar size {expected:.6f} (fill-bar reading would be {fillbar:.6f})")
