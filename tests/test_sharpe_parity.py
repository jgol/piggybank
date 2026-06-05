"""Phase 0 (proxy<->QC Sharpe calibration): the QC-side Sharpe must be computed
with the EXACT formula the proxy uses, so A1/A2 sign+rank transfer compare
like-for-like. These tests pin `qc_sharpe_from_orders.daily_sharpe` to the proxy's
inline arithmetic (evaluator_vectorized.py:1657-1659) and verify the QC daily-PnL
reconstruction (0DTE realized PnL, 0 on no-trade days, scale-invariant).
"""
import math

import numpy as np
import pytest

from scripts.qc_sharpe_from_orders import (
    daily_sharpe,
    reconstruct_daily_pnl,
    qc_sharpe_from_orders,
    trading_days_in_window,
    UnclosedLegError,
)


def _proxy_inline_sharpe(daily):
    """The proxy formula transcribed verbatim from evaluator_vectorized.py:1658-1659
    (the oracle these tests pin against)."""
    arr = np.asarray(daily, dtype=np.float64)
    returns_std = np.std(arr) if len(arr) > 1 else 1e-9
    return float(np.mean(arr) / max(returns_std, 1e-9) * math.sqrt(252.0))


@pytest.mark.parametrize("series", [
    [0.01, -0.005, 0.02, 0.0, -0.01, 0.015, 0.0, 0.008],
    [0.001] * 50,                       # constant positive -> std 0 -> 1e-9 floor
    [0.02, -0.02, 0.02, -0.02, 0.02],   # zero-mean
    list(np.linspace(-0.05, 0.05, 30)),
    [-0.03, -0.01, -0.02, 0.0, -0.015], # all-negative-ish
    [5.0],                              # len-1 edge (proxy: std:=1e-9 -> huge)
])
def test_daily_sharpe_matches_proxy_formula_exactly(series):
    """Identical daily series => identical Sharpe (the Phase 0 deliverable)."""
    got = daily_sharpe(series)
    want = _proxy_inline_sharpe(series)
    assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12), (
        f"daily_sharpe={got} != proxy inline {want} for {series[:4]}..."
    )


def test_uses_population_std_not_sample():
    """The proxy uses numpy default ddof=0 (population). A ddof=1 (sample) impl
    would diverge on small samples — pin it."""
    series = [0.01, 0.03, -0.02, 0.04]
    pop = np.mean(series) / np.std(series) * math.sqrt(252.0)         # ddof=0
    sample = np.mean(series) / np.std(series, ddof=1) * math.sqrt(252.0)
    assert not math.isclose(pop, sample)                              # they differ
    assert math.isclose(daily_sharpe(series), pop, rel_tol=1e-12)     # we are ddof=0


def test_no_risk_free_subtraction():
    """Proxy treats excess == raw (no RF term). Adding a positive constant to
    every day must change the Sharpe (it would not if an RF were subtracted to
    cancel the shift)."""
    base = [0.01, -0.005, 0.02, 0.0, -0.01]
    shifted = [r + 0.01 for r in base]
    assert not math.isclose(daily_sharpe(base), daily_sharpe(shifted))


def test_scale_invariance_notional_cancels():
    """Sharpe is scale-invariant, so dollar-PnL and return-unit series give the
    same number — this is why reconstruct_daily_pnl never needs the notional."""
    series = np.array([120.0, -55.0, 200.0, 0.0, -80.0, 140.0])
    assert math.isclose(daily_sharpe(series), daily_sharpe(series / 73421.0),
                        rel_tol=1e-12)


def test_reconstruct_0dte_realized_pnl_and_no_trade_days_zero():
    """0DTE daily realized PnL == -sum(order values that day); no-fill trading
    days are 0.0 so the series spans the full calendar (matches the proxy)."""
    days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    orders = [
        # 2025-01-02: receive 5.00 credit, pay 1.50 to close -> realized +3.50
        {"time": "2025-01-02T10:00:00", "value": -5.00},  # value<0 = cash received
        {"time": "2025-01-02T15:45:00", "value": 1.50},   # value>0 = cash paid
        # 2025-01-06: net loss -> realized -2.00
        {"time": "2025-01-06T10:00:00", "value": -4.00},
        {"time": "2025-01-06T15:45:00", "value": 6.00},
        # 2025-01-03 and 2025-01-07: no fills -> 0.0
    ]
    pnl = reconstruct_daily_pnl(orders, days)
    assert np.allclose(pnl, [3.50, 0.0, -2.00, 0.0]), pnl.tolist()


def test_reconstruct_tolerates_capitalized_keys():
    days = ["2025-01-02"]
    pnl = reconstruct_daily_pnl([{"Time": "2025-01-02T10:00:00", "Value": -7.0}], days)
    assert np.allclose(pnl, [7.0])


def test_end_to_end_qc_sharpe_from_orders():
    """Full path: fills + calendar -> daily PnL -> proxy-formula Sharpe."""
    days = ["2025-01-02", "2025-01-03", "2025-01-06"]
    orders = [
        {"time": "2025-01-02T10:00:00", "value": -5.0},
        {"time": "2025-01-02T15:45:00", "value": 2.0},   # +3.0
        {"time": "2025-01-06T10:00:00", "value": -5.0},
        {"time": "2025-01-06T15:45:00", "value": 1.0},   # +4.0
    ]                                                     # 2025-01-03 -> 0.0
    got = qc_sharpe_from_orders(orders, days)
    want = _proxy_inline_sharpe([3.0, 0.0, 4.0])
    assert math.isclose(got, want, rel_tol=1e-12)


# --- SPXW cash-settlement guard (Blocker 1) -------------------------------------

def test_settlement_order_raises_unclosed_leg():
    """A type-6 SPXW settlement order (value=±strike, ITM) must raise rather than
    poison -sum(value) with ~±$strike. Mirrors generated_qc/qc_orders_calib_debit."""
    days = ["2025-12-24"]
    orders = [
        {"time": "2025-12-24T10:00:00", "value": -3.0, "quantity": -1,
         "status": 3, "type": 0, "symbol": {"value": "SPXW  251224C06925000"}},
        # ITM settlement booked as underlying-at-strike, NEXT UTC day, type=6:
        {"time": "2025-12-25T06:00:00", "value": -6925.0, "quantity": -1,
         "status": 3, "type": 6, "symbol": {"value": "SPXW  251224C06925000"}},
    ]
    with pytest.raises(UnclosedLegError):
        reconstruct_daily_pnl(orders, days)


def test_unclosed_net_quantity_raises():
    """An open fill with no closing fill (net qty != 0) must raise."""
    days = ["2025-01-02"]
    orders = [{"time": "2025-01-02T10:00:00", "value": -5.0, "quantity": -1,
               "status": 3, "type": 0, "symbol": {"value": "SPXW  X"}}]
    with pytest.raises(UnclosedLegError):
        reconstruct_daily_pnl(orders, days)


def test_strict_false_computes_despite_settlement():
    """strict=False returns the (known-approximate) number instead of raising."""
    days = ["2025-01-02"]
    orders = [
        {"time": "2025-01-02T10:00:00", "value": -5.0, "quantity": -1,
         "status": 3, "type": 0, "symbol": {"value": "SPXW  X"}},
        {"time": "2025-01-02T15:45:00", "value": 2.0, "quantity": 1,
         "status": 3, "type": 0, "symbol": {"value": "SPXW  X"}},
    ]
    pnl = reconstruct_daily_pnl(orders, days, strict=False)
    assert np.allclose(pnl, [3.0])  # -(-5+2) = 3, net qty 0 -> no raise even strict


def test_nonfilled_orders_excluded():
    """status != 3 (e.g. Canceled=5) orders must not enter the PnL sum."""
    days = ["2025-01-02"]
    orders = [
        {"time": "2025-01-02T10:00:00", "value": -5.0, "quantity": -1,
         "status": 3, "type": 0, "symbol": {"value": "SPXW  X"}},
        {"time": "2025-01-02T15:45:00", "value": 2.0, "quantity": 1,
         "status": 3, "type": 0, "symbol": {"value": "SPXW  X"}},
        # canceled order with a phantom value must be ignored:
        {"time": "2025-01-02T11:00:00", "value": 999.0, "quantity": 0,
         "status": 5, "type": 0, "symbol": {"value": "SPXW  X"}},
    ]
    assert np.allclose(reconstruct_daily_pnl(orders, days), [3.0])


# --- calendar parity (Blocker 2) ------------------------------------------------

def test_calendar_matches_proxy_parquet_length():
    """trading_days_in_window must come from l1_cnn.parquet (the proxy's own
    calendar), NOT spx_daily_ohlc.csv (which over-counts ~3 half-days/window).
    Oct2024-May2025 must be the proxy's 163, not the CSV's 166."""
    import pandas as pd
    from scripts.qc_sharpe_from_orders import _PROXY_PARQUET
    days = trading_days_in_window("2024-10-01", "2025-05-31")
    parquet_dates = {str(d)[:10] for d in
                     pd.read_parquet(_PROXY_PARQUET, columns=["date"])["date"]}
    expected = sorted(d for d in parquet_dates if "2024-10-01" <= d <= "2025-05-31")
    assert days == expected
    assert len(days) == 163, f"expected proxy's 163 trading days, got {len(days)}"


# --- daily_sharpe edge contracts ------------------------------------------------

def test_daily_sharpe_empty_is_zero():
    assert daily_sharpe([]) == 0.0


def test_daily_sharpe_len1_replicates_proxy_huge_edge():
    """len==1 mirrors the proxy's std:=1e-9 (sign-preserving, huge) for parity."""
    assert daily_sharpe([5.0]) == _proxy_inline_sharpe([5.0])
    assert daily_sharpe([-5.0]) == _proxy_inline_sharpe([-5.0])
    assert daily_sharpe([5.0]) > 0 and daily_sharpe([-5.0]) < 0
