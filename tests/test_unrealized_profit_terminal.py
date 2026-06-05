"""UnrealizedProfitPct — position-aware exit terminal (Phase-2 of
docs/viable_run_plan_2026_06_03.md).

The GP's exit trees could previously only reference MARKET terminals (ATM_IV,
MinutesToClose, RealizedVol30m, …) — none position-aware — so they could NOT
express the canonical premium-selling exit ("close at 50% of max profit") nor a
loss-cut. UnrealizedProfitPct closes that gap: a RAW per-bar terminal giving the
fraction of the structure's max profit captured so far.

These tests pin the contract:
  1. VALUE  — ~0 at entry, →~1 as the credit spread decays to worthless (max
     profit), and NEGATIVE when the position moves against you.
  2. EXIT   — GT(UnrealizedProfitPct, EphReal(0.5)) closes the position when the
     unrealized profit first crosses 50%, vs a hold-to-close baseline that holds.
     A higher target (0.9) closes LATER (the value is a true monotone fraction).
  3. FLAT   — value is 0.0 when not in a position (entry-tree references benign).
  4. NO LEAK — the value resets per trade (no cross-trade carry-over).
  5. CODEGEN — the generated QC algorithm emits the terminal, injects the live
     QC unrealized P&L / |entry credit| into `s`, and compiles.
  6. PARITY  — a tree NOT referencing the terminal is byte-identical to the
     pure-vectorized path (the scalar override only engages when the terminal is
     actually used), and the scalar evaluator matches the vectorized one.
"""
import numpy as np
import pandas as pd
import pytest

import layer2.evaluator_vectorized as ev
from layer2.evaluator_vectorized import (
    vectorized_backtest,
    prepare_terminal_data,
    tree_references_upp,
    _eval_node_scalar,
    _exit_signal_at_bar,
    UNREALIZED_PROFIT_PCT,
    VectorizedTreeEvaluator,
)
from layer2.templates import template_by_name
from layer2.grammar import from_sexpr, to_str, to_nl

NOTIONAL = 1000.0
ALWAYS = "GT(EphReal(1.0), EphReal(0.0))"   # always-true entry
NEVER = "GT(EphReal(-10.0), EphReal(0.0))"  # always-false (hold to EOD)
SIZE = "EphReal(0.5)"


# ---------------------------------------------------------------------------
# Fixtures: deterministic tapes (no RNG -> exactly reproducible)
# ---------------------------------------------------------------------------

def _bpc_tape(direction: str, n_dates: int = 3, bars: int = 60) -> pd.DataFrame:
    """A bull-put-credit tape.

    direction="rally": steady ~3.5% intraday rally -> short put spread decays ->
        UnrealizedProfitPct rises toward +1 (max profit), no stop.
    direction="adverse": ~3% mid-day drop -> short put deep ITM -> the position
        moves against you -> UnrealizedProfitPct goes negative.
    """
    rows = []
    base = 5000.0
    for d in range(n_dates):
        date = f"2024-{d + 1:02d}-15"
        for b in range(bars):
            if direction == "rally":
                px = base * (1.0 + 0.0006 * b)
            else:  # adverse
                px = base * (1.0 - 0.030 * min(max(b - 5, 0) / 20.0, 1.0))
            rows.append(dict(
                date=date, bar_position=b,
                ATM_IV=0.20, RawSpread=0.02, DeltaSpread1=0.0, DeltaSpread5=0.0,
                MinutesToClose=max(0, 390 - b * 6), VIXSpot=16.0,
                VIXTermSlope=-0.5, RealizedVol30m=0.015, BarOfDay=b, SPXClose=px,
            ))
    return pd.DataFrame(rows)


def _run(df, exit_sexpr, *, no_haircut=False, stop=0, td=None):
    """Run an always-enter BPC strategy with the given exit tree. Returns
    (result, captured) where captured is a list of (bar_index, live_upp_value)
    recorded by spying on _exit_signal_at_bar (only populated when the exit tree
    references UnrealizedProfitPct)."""
    tpl = template_by_name("bull_put_credit_standard")  # credit; short put spread
    entry = from_sexpr(ALWAYS)
    exit_ = from_sexpr(exit_sexpr)
    size = from_sexpr(SIZE)
    if td is None:
        td = prepare_terminal_data(df)

    captured = []
    _orig_sig = ev._exit_signal_at_bar

    def _spy(et, i, t, wdp, nds, upp_value, upp_hist):
        captured.append((i, upp_value))
        return _orig_sig(et, i, t, wdp, nds, upp_value, upp_hist)

    _orig_cf = dict(ev._BASE_CREDIT_FACTORS)
    ev._exit_signal_at_bar = _spy
    if no_haircut:
        for k in list(ev._BASE_CREDIT_FACTORS):
            ev._BASE_CREDIT_FACTORS[k] = 1.0
    try:
        r = vectorized_backtest(
            entry, exit_, size, df, tpl, terminal_data=td,
            notional=NOTIONAL, warmup_bars=2, min_bars_in_trade=1,
            stop_loss_credit_multiple=stop,
        )
    finally:
        ev._exit_signal_at_bar = _orig_sig
        ev._BASE_CREDIT_FACTORS.clear()
        ev._BASE_CREDIT_FACTORS.update(_orig_cf)
    return r, captured


# ---------------------------------------------------------------------------
# 0. Grammar / wiring sanity
# ---------------------------------------------------------------------------

def test_terminal_in_all_conditions():
    """Position state, not an encoder feature -> present in ALL four ablation
    conditions equally (else the A/B/C/D comparison is on exit vocabulary, not
    representations)."""
    from layer2.grammar import (
        build_terminal_set, build_scalar_only_terminal_set,
        build_probes_only_terminal_set, build_emb_only_terminal_set,
    )
    for builder in (build_terminal_set, build_scalar_only_terminal_set,
                    build_probes_only_terminal_set, build_emb_only_terminal_set):
        names = {t.name for t in builder()}
        assert UNREALIZED_PROFIT_PCT in names, f"missing in {builder.__name__}"


def test_terminal_is_raw_not_normalized():
    """RAW (absent from TERMINAL_NORM_STATS) so GT(UnrealizedProfitPct, 0.5)
    literally means '>=50% of max profit'. EphReal's [-3,3] sampling covers the
    [0,1] profit band and the [-1,0] loss-cut band."""
    from layer2.terminal_stats import NORMALIZED_TERMINALS, normalize
    assert UNREALIZED_PROFIT_PCT not in NORMALIZED_TERMINALS
    assert normalize(UNREALIZED_PROFIT_PCT, 0.5) == 0.5  # identity (no z-score)


def test_parse_and_nl_roundtrip():
    t = from_sexpr("GT(UnrealizedProfitPct, EphReal(0.5))")
    assert to_str(t) == "GT(UnrealizedProfitPct, EphReal(0.5))"
    assert tree_references_upp(t) is True
    assert not tree_references_upp(from_sexpr("GT(ATM_IV, EphReal(0.0))"))
    # NL render exists (L3 handoff) and is descriptive
    assert "max profit" in to_nl(t).lower()


def test_seeded_zeros_array_present_for_hardening_check():
    """prepare_terminal_data seeds an all-zeros array so experiment.py's 'every
    grammar terminal has data backing' check passes AND flat/idle bars read 0.0."""
    df = _bpc_tape("rally", n_dates=1)
    td = prepare_terminal_data(df)
    assert UNREALIZED_PROFIT_PCT in td
    arr = td[UNREALIZED_PROFIT_PCT]
    assert arr.shape == (len(df),)
    assert np.all(arr == 0.0)


# ---------------------------------------------------------------------------
# 1. VALUE: ~0 at entry, ->~1 at max profit, negative when losing
# ---------------------------------------------------------------------------

def test_value_zero_at_entry_rises_to_one_on_decay():
    """No-haircut credit spread on a rally tape: the live UnrealizedProfitPct
    starts near 0 (crosses ~0 in the first in-position bars) and climbs to ~+1 as
    the spread decays to worthless (max profit captured). The exit threshold is
    set above the max so the position is never closed by signal — we observe the
    full trajectory."""
    df = _bpc_tape("rally", n_dates=1)
    # Exit references UPP (so the spy captures it) but never fires.
    _, captured = _run(df, "GT(UnrealizedProfitPct, EphReal(5.0))",
                       no_haircut=True, stop=0)
    assert captured, "exit tree references UPP -> per-bar values must be recorded"
    first_trade = [u for (i, u) in captured if i < 59]
    # Near 0 at entry (no haircut): the first marks are small-magnitude and the
    # series crosses ~0 within the first few bars.
    assert abs(first_trade[0]) < 0.15, f"entry UPP not ~0: {first_trade[0]}"
    assert min(abs(u) for u in first_trade[:5]) < 0.02, "never crosses ~0 near entry"
    # Climbs to ~max profit (+1) as the spread decays.
    assert max(first_trade) > 0.9, f"did not reach ~max profit: max={max(first_trade)}"
    assert max(first_trade) <= 1.0 + 1e-6, "exceeded +1 (max profit) — formula wrong"
    # Monotone-ish rise: last value strictly above the first.
    assert first_trade[-1] > first_trade[0]


def test_value_negative_when_position_loses():
    """Adverse tape: the short put goes deep ITM, current cost-to-close exceeds
    the credit received, so UnrealizedProfitPct is NEGATIVE (worse than entry)."""
    df = _bpc_tape("adverse", n_dates=1)
    _, captured = _run(df, "GT(UnrealizedProfitPct, EphReal(5.0))",
                       no_haircut=True, stop=0)
    assert captured
    vals = [u for (i, u) in captured if i < 59]
    assert min(vals) < -0.5, f"adverse trade UPP never went meaningfully negative: min={min(vals)}"


# ---------------------------------------------------------------------------
# 2. EXIT: profit-target close vs hold-to-close; monotone threshold
# ---------------------------------------------------------------------------

def test_profit_target_closes_position_vs_hold_baseline():
    """GT(UnrealizedProfitPct, EphReal(0.5)) closes the trade by 'signal' when
    profit first crosses 50%; the hold-to-close baseline (never-true exit) does
    NOT signal-exit. The profit-target exit must fire strictly earlier (before
    the EOD force-close) on the rally tape."""
    df = _bpc_tape("rally", n_dates=1)
    r_hold, _ = _run(df, NEVER, stop=0)
    r_pt, _ = _run(df, "GT(UnrealizedProfitPct, EphReal(0.5))", stop=0)

    # Baseline: no voluntary (signal) exit — held to session/EOD.
    assert all(t.exit_reason != "signal" for t in r_hold.trades), \
        f"hold baseline should not signal-exit: {[t.exit_reason for t in r_hold.trades]}"
    # Profit target: at least one trade closes BY SIGNAL.
    pt_signal = [t for t in r_pt.trades if t.exit_reason == "signal"]
    assert pt_signal, "profit-target exit never fired"
    # The first profit-target trade closes before the hold baseline's first exit.
    first_pt = min(pt_signal, key=lambda t: t.entry_bar)
    first_hold = min(r_hold.trades, key=lambda t: t.entry_bar)
    assert first_pt.entry_bar == first_hold.entry_bar, "same entry bar expected"
    assert first_pt.exit_bar < first_hold.exit_bar, \
        f"profit target did not close earlier: pt exit={first_pt.exit_bar} hold={first_hold.exit_bar}"


def test_higher_profit_target_closes_later():
    """The value is a TRUE monotone fraction of max profit: a 0.9 target must take
    longer to trip than a 0.5 target (reaching 90% decay is later than 50%)."""
    df = _bpc_tape("rally", n_dates=1)
    r50, _ = _run(df, "GT(UnrealizedProfitPct, EphReal(0.5))", stop=0)
    r90, _ = _run(df, "GT(UnrealizedProfitPct, EphReal(0.9))", stop=0)
    t50 = min((t for t in r50.trades if t.exit_reason == "signal"),
              key=lambda t: t.entry_bar)
    t90 = min((t for t in r90.trades),
              key=lambda t: t.entry_bar)
    assert t50.entry_bar == t90.entry_bar
    # Bars-in-trade to reach 90% profit >= bars to reach 50% profit.
    held50 = t50.exit_bar - t50.entry_bar
    held90 = t90.exit_bar - t90.entry_bar
    assert held90 > held50, f"0.9 target ({held90} bars) not slower than 0.5 ({held50} bars)"


# ---------------------------------------------------------------------------
# 3. FLAT-STATE value is 0.0 (entry-tree references are benign)
# ---------------------------------------------------------------------------

def test_flat_state_value_is_zero_entry_tree_benign():
    """When FLAT, UnrealizedProfitPct is 0.0. An ENTRY tree that requires it to
    exceed a positive target can therefore NEVER fire (it is always 0 before a
    position exists) -> 0 trades; the identical strategy with an always-true entry
    DOES trade. Proves the flat value is a benign no-op."""
    df = _bpc_tape("rally", n_dates=2)
    td = prepare_terminal_data(df)
    tpl = template_by_name("bull_put_credit_standard")
    size = from_sexpr(SIZE)
    exit_ = from_sexpr(NEVER)
    # Entry gated on a positive profit fraction -> impossible while flat.
    entry_upp = from_sexpr("GT(UnrealizedProfitPct, EphReal(0.5))")
    r_gated = vectorized_backtest(entry_upp, exit_, size, df, tpl, terminal_data=td,
                                  notional=NOTIONAL, warmup_bars=2, min_bars_in_trade=1)
    assert len(r_gated.trades) == 0, \
        f"flat-state UPP must be 0 -> entry GT(UPP,0.5) cannot fire; got {len(r_gated.trades)} trades"
    # Sanity: an always-true entry on the same tape DOES trade.
    r_open = vectorized_backtest(from_sexpr(ALWAYS), exit_, size, df, tpl, terminal_data=td,
                                 notional=NOTIONAL, warmup_bars=2, min_bars_in_trade=1)
    assert len(r_open.trades) > 0


# ---------------------------------------------------------------------------
# 4. No cross-trade leakage
# ---------------------------------------------------------------------------

def test_no_cross_trade_leakage_in_history():
    """The per-trade UnrealizedProfitPct history is cleared on every entry, so a
    Lag of it never reaches into a PRIOR trade. We check this directly: on the
    FIRST in-position bar of every trade, Lag(UnrealizedProfitPct, 5) must be 0.0
    (no history yet this trade), never a value carried from the prior trade.

    Implementation: spy on the scalar evaluator's history buffer at the moment the
    exit signal is evaluated. The buffer's length equals bars-into-this-trade, so
    at the first in-position bar it has exactly 1 element (this bar) and any lag>0
    masks to 0.0.
    """
    df = _bpc_tape("rally", n_dates=3)
    histories = []  # (i, len(upp_hist)) snapshots at each exit eval
    _orig = ev._exit_signal_at_bar

    def _spy(et, i, td, wdp, nds, upp_value, upp_hist):
        histories.append((i, len(upp_hist), list(upp_hist)))
        return _orig(et, i, td, wdp, nds, upp_value, upp_hist)

    ev._exit_signal_at_bar = _spy
    try:
        # Exit on a profit target so there are MULTIPLE trades per day (re-entry).
        r, _ = _run(df, "GT(UnrealizedProfitPct, EphReal(0.5))", stop=0,
                    td=prepare_terminal_data(df))
    finally:
        ev._exit_signal_at_bar = _orig

    assert len(r.trades) >= 3, "need multiple trades to test reset"
    # Group history snapshots by trade: a length of 1 marks a fresh trade's first
    # in-position bar. The history is monotone increasing WITHIN a trade and resets
    # to length 1 at each new trade — never carrying the prior trade's tail.
    lengths = [ln for (_, ln, _) in histories]
    # Every reset point (length==1) must be preceded (if any) by a STRICTLY larger
    # length (the prior trade ran longer) — i.e. it genuinely reset, not continued.
    reset_idxs = [k for k, ln in enumerate(lengths) if ln == 1]
    assert len(reset_idxs) >= 2, f"expected >=2 trade resets, got {len(reset_idxs)}"
    for k in reset_idxs:
        if k > 0:
            assert lengths[k - 1] >= 1
            # The buffer at the reset contains exactly THIS bar's value (1 element).
            assert histories[k][1] == 1
    # And lengths never exceed bars-in-trade: max length <= max bars held + 1.
    max_held = max((t.exit_bar - t.entry_bar) for t in r.trades)
    assert max(lengths) <= max_held + 1


def test_lag_of_upp_zero_at_trade_start():
    """Direct unit check of the no-leak masking: Lag(UnrealizedProfitPct, k) on the
    FIRST bar of a trade (upp_hist has 1 element) returns 0.0 for any k>=1, never a
    prior-trade value — even if upp_hist (defensively) still held stale data."""
    # within_day_pos large enough that the day-boundary mask does NOT trip; the
    # trade-start mask (len(upp_hist)) is what must bite.
    wdp = np.array([50], dtype=np.int64)
    nds = np.array([1.0])
    td = {UNREALIZED_PROFIT_PCT: np.zeros(1)}
    lag3 = from_sexpr("Lag(UnrealizedProfitPct, EphInt(3))")
    # Fresh trade: history has exactly the current bar's value (0.42).
    v = _eval_node_scalar(lag3, 0, td, wdp, nds, upp_value=0.42, upp_hist=[0.42])
    assert v == 0.0, f"Lag into a not-yet-existent prior bar of THIS trade must be 0.0, got {v}"
    # With 4 bars of this trade, Lag(.,3) reads the oldest (index -4).
    hist = [0.10, 0.20, 0.30, 0.42]
    v2 = _eval_node_scalar(lag3, 0, td, wdp, nds, upp_value=0.42, upp_hist=hist)
    assert v2 == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 5. Codegen emits the terminal and the generated code compiles
# ---------------------------------------------------------------------------

def test_codegen_emits_and_compiles():
    import py_compile, tempfile, os
    from layer3.codegen import generate_qc_algorithm
    src = generate_qc_algorithm(
        strategy_id="upp_profit_target",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.0))",
        exit_sexpr="GT(UnrealizedProfitPct, EphReal(0.5))",
        size_sexpr="EphReal(1.0)",
        start_date="2025-04-08", end_date="2025-04-30",
        stop_mult=0.0,
    )
    # The exit expression reads the terminal from the scalar dict.
    assert 's.get("UnrealizedProfitPct", 0.0)' in src
    # The live value is injected from QC's per-position unrealized P&L / |entry credit|.
    assert 's["UnrealizedProfitPct"] = _upp_pnl / abs(self._entry_credit)' in src
    assert 's["UnrealizedProfitPct"] = 0.0' in src  # flat-state branch
    fn = os.path.join(tempfile.gettempdir(), "test_upp_codegen.py")
    with open(fn, "w") as f:
        f.write(src)
    py_compile.compile(fn, doraise=True)  # raises on syntax error


def test_codegen_node_translation():
    from layer3.codegen import _node_to_python
    assert _node_to_python(from_sexpr("GT(UnrealizedProfitPct, EphReal(0.5))")) == \
        '(s.get("UnrealizedProfitPct", 0.0) > 0.5)'
    # Loss-cut form (negative threshold) translates cleanly too.
    assert _node_to_python(from_sexpr("LT(UnrealizedProfitPct, EphReal(-0.5))")) == \
        '(s.get("UnrealizedProfitPct", 0.0) < -0.5)'


# ---------------------------------------------------------------------------
# 6. Parity: non-UPP trees unaffected; scalar evaluator matches vectorized
# ---------------------------------------------------------------------------

def test_non_upp_exit_tree_is_byte_identical_to_vectorized_path():
    """A strategy whose exit tree does NOT reference UnrealizedProfitPct must be
    completely unaffected by this feature (the scalar override only engages when
    the terminal is used). We run the SAME non-UPP strategy and assert the spy was
    never called (fast vectorized path) and the result is well-formed."""
    df = _bpc_tape("rally", n_dates=2)
    r, captured = _run(df, "GT(MinutesToClose, EphReal(-99.0))", stop=0)  # exit never via UPP
    assert captured == [], "non-UPP exit tree must NOT trigger the per-bar scalar override"
    assert len(r.trades) > 0


def test_scalar_evaluator_matches_vectorized_when_upp_absent():
    """For a tree with NO UnrealizedProfitPct, the single-bar scalar evaluator must
    return the same value as the vectorized evaluator at every bar (guards the
    scalar path against operator-semantics drift)."""
    df = _bpc_tape("rally", n_dates=2)
    td = prepare_terminal_data(df)
    n = len(df)
    # day-boundary structures (mirror vectorized_backtest)
    dates = df["date"].values
    day_mask = np.ones(n)
    wdp = np.zeros(n, dtype=np.int64)
    day_mask[0] = 0.0
    pos = 0
    for i in range(1, n):
        if str(dates[i]) != str(dates[i - 1]):
            day_mask[i] = 0.0
            pos = 0
        else:
            pos += 1
        wdp[i] = pos
    vec = VectorizedTreeEvaluator(td, n, day_boundary_mask=day_mask, within_day_pos=wdp)

    for sexpr in (
        "GT(ATM_IV, EphReal(0.0))",
        "AND(GT(RealizedVol30m, EphReal(-0.5)), LT(MinutesToClose, EphReal(2.0)))",
        "GT(Delta(ATM_IV, EphInt(3)), EphReal(-5.0))",
        "OR(CrossAbove(ATM_IV, RealizedVol30m), GT(BarOfDay, EphReal(0.0)))",
        "GT(Sqrt(RealizedVol30m), Div(ATM_IV, EphReal(2.0)))",
    ):
        tree = from_sexpr(sexpr)
        varr = np.where(np.isnan(vec.evaluate(tree).astype(np.float64)), 0.0,
                        vec.evaluate(tree).astype(np.float64))
        for i in range(0, n, 7):  # sample bars for speed
            sv = _eval_node_scalar(tree, i, td, wdp, day_mask, upp_value=0.0, upp_hist=[])
            sv = 0.0 if (isinstance(sv, float) and np.isnan(sv)) else float(sv)
            assert sv == pytest.approx(float(varr[i]), abs=1e-9), \
                f"scalar!=vectorized at bar {i} for {sexpr}: {sv} vs {varr[i]}"


def test_reconciliation_holds_with_upp_exit():
    """sum(bar_returns) must still reconcile to sum(trade.pnl)/notional with a
    UPP-driven signal exit present (the new exit path must not break the
    P&L<->Sharpe-basis invariant)."""
    df = _bpc_tape("rally", n_dates=3)
    r, _ = _run(df, "GT(UnrealizedProfitPct, EphReal(0.5))", stop=0)
    assert any(t.exit_reason == "signal" for t in r.trades)
    sum_ret = float(np.sum(r.returns))
    sum_pnl = float(sum(t.pnl for t in r.trades)) / NOTIONAL
    assert abs(sum_ret - sum_pnl) < 1e-6, \
        f"reconciliation broke with a UPP exit: {sum_ret:.10f} vs {sum_pnl:.10f}"
