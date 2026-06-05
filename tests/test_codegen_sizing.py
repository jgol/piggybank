"""Position-sizing parity: codegen ↔ proxy (2026-05-31).

The proxy (layer2/evaluator_vectorized.py) is the GP objective; codegen sizes the
deployed QC position. The prior codegen sized off an IV-blind `wing_pct·spot`
constant (notional 10000), diverging 5-20x per template and breaking Sharpe
scale-invariance. The fix transcribes the proxy's $/share sizing (NOTIONAL=1000,
entry_gross basis, equity-floor de-levering from QC's portfolio value); with
SetCash(100000)=NOTIONAL×100 (SPX mult) the proxy's contract count maps 1:1.

Two bugs fixed here:
  1. The proxy's `_total_ratio > 2` ratio-cap fired for any 4-leg structure (IC/IB
     sum 4 unit ratios > 2), forcing them to n=1. Gate on `any(ratio>1)` instead.
  2. Gross is priced from QC's LIVE chain (b) with a per-leg BS fallback, matching
     the proxy's real-quote-based surface better than a BS model.
"""
import inspect
import re
from datetime import datetime

from layer2.evaluator_vectorized import vectorized_backtest
from layer3.codegen import generate_qc_algorithm, validate_generated_code

_IC = dict(strategy_id="size_ic", template_name="iron_condor_standard",
           entry_sexpr="GT(ATM_IV, EphReal(0.18))",
           exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
           size_sexpr="EphReal(0.5)", start_date="2025-01-02", end_date="2025-01-31")


# --- proxy ratio-condition bug fix ------------------------------------------
def test_proxy_ratio_branch_gates_on_genuine_ratio_leg():
    """The ratio-adjustment + 5% cap must gate on a real ratio leg (any ratio>1),
    NOT on sum>2 — else IC/IB (4 unit-ratio legs) trip it and get sized to 1."""
    src = inspect.getsource(vectorized_backtest)
    assert "_has_ratio_leg = any(l.ratio > 1 for l in active_legs)" in src
    assert "if _has_ratio_leg:" in src
    # the old buggy guard must be gone from the sizing block
    assert "if _total_ratio > 2:" not in src, (
        "the sum>2 ratio guard must be replaced by _has_ratio_leg"
    )


# --- codegen sizing structure -----------------------------------------------
def test_codegen_sizing_transcribes_proxy():
    code = generate_qc_algorithm(**_IC)
    ok, err = validate_generated_code(code)
    assert ok, err
    assert "def _size_n_contracts" in code
    assert "NOTIONAL = 1000.0" in code            # proxy notional, $/share
    assert "self.Portfolio.TotalPortfolioValue / (NOTIONAL * 100.0)" in code  # equity floor
    assert "_has_ratio = any(_rt > 1" in code     # bug fix mirrored
    assert "_approx_abs_val" not in code          # old IV-blind sizing gone
    assert "_c.BidPrice" in code and "_c.AskPrice" in code  # (b) chain pricing
    # per-leg BS fallback present
    assert "per-leg BS fallback" in code


def _bare_instance(**kw):
    from layer3.diagnostics.operator_parity import build_codegen_instance, StrategySpec
    spec = StrategySpec(strategy_id=kw["strategy_id"], template_name=kw["template_name"],
                        entry_sexpr=kw["entry_sexpr"], exit_sexpr=kw["exit_sexpr"],
                        size_sexpr=kw["size_sexpr"], is_credit=True)
    inst, _ = build_codegen_instance(spec, (kw["start_date"], kw["end_date"]), norm_serial={})
    inst.Time = datetime(2025, 1, 2, 14, 15, 0)   # -> mtc=120
    inst._last_valid_iv = 0.15
    inst._derived_vix = 15.0
    inst._cached_chain = None                      # force the BS fallback path
    class _Pf:
        TotalPortfolioValue = 100000.0
    inst.Portfolio = _Pf()
    return inst


def test_codegen_ic_not_sized_to_one():
    """Regression for the ratio-branch bug: IC must NOT collapse to n=1.
    (No chain -> BS fallback prices the legs; the ratio cap must not apply.)"""
    inst = _bare_instance(**_IC)
    n = inst._size_n_contracts(5000.0, 0.5)
    assert n > 5, f"IC sized to {n} — the 4-leg ratio-cap bug has regressed (expected ~tens)"


def test_codegen_chain_price_preferred_over_bs():
    """When the live chain has a real quote, gross uses it (b), not the BS model."""
    inst = _bare_instance(**dict(_IC, template_name="bear_call_credit_standard"))
    # mock a 2-contract chain with fat quotes so chain-mid >> BS value
    class _C:
        def __init__(self, strike, right, bid, ask):
            self.Strike = strike; self.Right = right
            self.BidPrice = bid; self.AskPrice = ask; self.Price = (bid + ask) / 2
    from layer3.diagnostics.operator_parity import _Dummy  # OptionRight stub
    # bcc legs are calls; _find_contract filters on c.Right == OptionRight.Call
    import sys
    Call = sys.modules["AlgorithmImports"].OptionRight.Call
    Put = sys.modules["AlgorithmImports"].OptionRight.Put
    chain = [_C(5015, Call, 40.0, 42.0), _C(5030, Call, 38.0, 40.0)]
    inst._cached_chain = chain
    n_chain = inst._size_n_contracts(5000.0, 0.5)
    inst._cached_chain = None
    n_bs = inst._size_n_contracts(5000.0, 0.5)
    # fat chain quotes (gross ~80) -> far fewer contracts than the small BS gross
    assert n_chain < n_bs, (
        f"chain-priced n ({n_chain}) should differ from BS-fallback n ({n_bs}) "
        f"when chain quotes are present (proves (b) path is used)"
    )
