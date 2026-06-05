"""Residual-1: the sizing-GROSS realism factor (_GROSS_REALISM_FACTORS).

The proxy's _gross_pos_value over-states the real-fill gross (the cheap OTM long
wing prices far above its intraday fill), inflating the sizing basis and shrinking
proxy per-trade n vs QC's chain-priced (b)-sizing. _GROSS_REALISM_FACTORS haircuts
the SIZING basis (abs_val) only — never PnL/credit — so it is a constant
per-template sizing scale (Sharpe-direction-preserving) that restores n parity.

These tests prove (1) the measured/pending factor state, (2) the factor is wired
into abs_val so non-floored n scales ~1/factor, and (3) the per-trade NET credit
(PnL) is byte-identical with and without the haircut (it touches sizing only).
"""
import warnings

import pytest

from layer2.evaluator_vectorized import _GROSS_REALISM_FACTORS

warnings.filterwarnings("ignore")

_PARQUET = "raw_data/local_store/l1_minute_scalars.parquet"
_TRAIN_END = "2024-09-27"
_WARMUP, _END = "2024-12-01", "2025-01-31"
_ENTRY = "GT(VIXChange, Lag(Lag(Lag(VIXChange, EphInt(3)), EphInt(30)), EphInt(30)))"
_EXIT = ("CrossBelow(RealizedVol30m_5m, Lag(Sub(Lag(EphReal(-0.7563), EphInt(3)), "
         "Delta(DeltaSpread5, EphInt(30))), EphInt(3)))")
_SIZE = "Delta(ATM_IV_5m, EphInt(1))"
_TPL = "bear_call_credit_standard"


def test_factors_calibrated_all_templates():
    """All five are calibrated from the full-range gross-collector (n=378/template,
    same dataset/method as _BASE_CREDIT_FACTORS) — none left at the 1.00 stub, none
    a guessed value. Each must be a real haircut in (0, 1] (proxy over-states gross)."""
    # Recalibrated 2026-06-01 on the CLAMPED proxy (B2-fin) vs cached QC low-regime gross.
    expect = {"bull_put_credit": 0.71, "bear_call_credit": 0.72,
              "iron_condor": 0.70, "iron_butterfly": 0.65,
              "ratio_put_backspread": 0.70}
    assert _GROSS_REALISM_FACTORS == expect
    # all are genuine haircuts (real fills below model gross), none the 1.0 stub
    assert all(0.0 < v < 1.0 for v in _GROSS_REALISM_FACTORS.values())


def _run(sizing_log, monkeypatch=None, factor=None):
    import pandas as pd  # noqa: F401 (ensures pandas available)
    from layer2.io import load_minute_parquet
    from layer2.evaluator_vectorized import (
        vectorized_backtest, prepare_terminal_data)
    from layer2.terminal_stats import compute_norm_stats_from_data
    from layer2.templates import template_by_name
    from layer2.grammar import from_sexpr
    import layer2.evaluator_vectorized as ev

    if factor is not None:
        monkeypatch.setitem(ev._GROSS_REALISM_FACTORS, "bear_call_credit", factor)

    df, _ = load_minute_parquet(_PARQUET)
    norm = compute_norm_stats_from_data(df[df["date"] <= _TRAIN_END])
    sl = df[(df["date"] >= _WARMUP) & (df["date"] <= _END)].reset_index(drop=True)
    td = prepare_terminal_data(sl, norm_stats_override=norm)
    res = vectorized_backtest(
        from_sexpr(_ENTRY), from_sexpr(_EXIT), from_sexpr(_SIZE), sl,
        template_by_name(_TPL), delta_tree=None, terminal_data=td,
        _sizing_log=sizing_log)
    return res


@pytest.mark.skipif(not __import__("pathlib").Path(_PARQUET).exists(),
                    reason="minute parquet not present")
def test_gross_factor_scales_n_but_not_pnl(monkeypatch):
    """factor 0.60 vs 1.00 on bcc_f2 (size=Delta(ATM_IV_5m,1), n=1-9 -- the
    abs_val-BINDING regime, NOT the conc-cap regime). Proves two invariants:
      (1) per-trade NET credit is byte-identical (haircut touches sizing only);
      (2) where abs_val binds, n scales ~1/0.60.
    The conc-cap regime is a deliberate NO-OP (n set by shared strike-width
    margin -> no gross gap there; see _GROSS_REALISM_FACTORS doc / H1). This test
    asserts the small-n scaling because that is where the gross gap actually
    lives; it does NOT claim a uniform 1.67x across all strategies."""
    log_haircut, log_unit = [], []
    res_h = _run(log_haircut, monkeypatch, factor=0.60)
    res_u = _run(log_unit, monkeypatch, factor=1.00)

    # Same entries (same trade dates) — sizing must not change WHICH bars trade.
    assert len(log_haircut) == len(log_unit) and len(log_haircut) > 5
    pairs = list(zip(log_unit, log_haircut))

    # (1) NET credit per trade is identical — the haircut never touches PnL/credit.
    for u, h in pairs:
        assert abs(u["net"] - h["net"]) < 1e-6, "gross factor must not move net/PnL"
        assert abs(u["gross"] - h["gross"]) < 1e-6, "model gross itself unchanged"

    # (2) On non-floored trades (n_unit >= 2) the count scales ~1/0.60 = 1.67x.
    ratios = [h["n_contracts"] / u["n_contracts"]
              for u, h in pairs if u["n_contracts"] >= 2]
    assert ratios, "need at least one non-floored trade to test scaling"
    import statistics as st
    assert 1.4 <= st.median(ratios) <= 1.9, f"median n ratio {st.median(ratios):.2f} != ~1.67"


def test_codegen_forward_fills_iv_until_chain_ready():
    """Residual-2: generated code must forward-fill ATM_IV until the live chain is
    populated (>=50), gating BOTH _last_valid_iv latch and the T2 session-open
    capture — otherwise the forming-chain morning ramp leaks into ATM_IV_5m."""
    from layer3.diagnostics.operator_parity import build_codegen_instance, StrategySpec
    spec = StrategySpec(
        strategy_id="r2", template_name=_TPL,
        entry_sexpr="GT(VIXChange, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="Delta(ATM_IV_5m, EphInt(1))", is_credit=True)
    _, code = build_codegen_instance(spec, ("2025-01-02", "2025-01-31"), norm_serial={})
    # R2 v4: the settling window is TIME-based AND anchored on chain-ready (not
    # 09:30) -- the chain fills within a minute so a pure chain-count gate was a
    # no-op, and a 09:30-anchored window releases too early on slow-chain days.
    assert "_SESSION_IV_WARMUP = 6" in code
    assert "_chain_ready_bar" in code
    # exact integer-bar arithmetic (float bar_of_day could land on 5.9999)
    assert "(int(bar_of_day) - self._chain_ready_bar) >= _SESSION_IV_WARMUP" in code
    assert "_iv_reliable" in code
    # the chain-ready anchor must re-arm at the day boundary (per-session window)
    assert "self._chain_ready_bar = -1.0" in code
    # the latch + the T2 open-capture are BOTH inside the iv-reliable branch
    i_gate = code.index("if atm_iv > 0.001 and _iv_reliable:")
    i_latch = code.index("self._last_valid_iv = atm_iv", i_gate)
    i_else = code.index("            atm_iv = self._last_valid_iv", i_gate)
    assert i_gate < i_latch < i_else, "latch must be inside the iv-reliable branch"
    # the chain-ready anchor is set BEFORE the reliability test that reads it,
    # and is initialized (-1.0) before the anchor-set can run -> never read unset
    i_init = code.index("self._chain_ready_bar = -1.0")
    i_anchor = code.index("self._chain_ready_bar = int(bar_of_day)")
    assert i_init < i_anchor < i_gate, "init < anchor-set < gate read ordering"


def test_codegen_never_reads_invalid_contract_price_attr():
    """OptionContract has NO .Price attribute (that is Securities/holdings); reading
    `_c.Price` threw a Runtime Error the instant a both-sides-missing contract was
    sized (crashed the backtest mid-run). The leg-pricing fallback must use the
    one-sided quote then _c.LastPrice -- never `_c.Price`."""
    from layer3.diagnostics.operator_parity import build_codegen_instance, StrategySpec
    spec = StrategySpec(
        strategy_id="px", template_name=_TPL,
        entry_sexpr="GT(VIXChange, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="Delta(ATM_IV_5m, EphInt(1))", is_credit=True)
    _, code = build_codegen_instance(spec, ("2025-01-02", "2025-01-31"), norm_serial={})
    # the buggy attribute must be gone everywhere it was applied to a contract
    assert "_c.Price" not in code, "OptionContract.Price is invalid -- crashes at runtime"
    # the correct chain-side fallback chain must be present
    assert "_price = max(_bid, _ask)" in code, "one-sided-quote fallback missing"
    assert "_c.LastPrice" in code, "LastPrice fallback missing"
