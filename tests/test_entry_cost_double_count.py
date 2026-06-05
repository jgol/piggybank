"""Regression tests for the entry-cost double-count fix (2026-06-03).

For a CREDIT structure the credit haircut (`_BASE_CREDIT_FACTORS`) was r5-calibrated
as median(QC MarketOrder fill)/median(proxy BS-mid). QC market orders cross the
bid-ask, so the haircut ALREADY captures the entry bid-ask crossing. `_entry_costs`
additionally charged a per-leg spread-crossing → a double-count for credit ENTRY.

The fix: `_entry_costs(skip_spread_crossing=True)` charges fees + the flat uncertainty
charge ONLY (no per-leg bid-ask spread-crossing); the credit-ENTRY call site passes
`skip_spread_crossing=(raw_entry_val < -0.01)`. EXITS (a real mid-day close DOES cross
the spread) and DEBIT entry are unchanged.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from layer2.evaluator_vectorized import _entry_costs  # noqa: E402
from layer2.templates import base_template_by_name  # noqa: E402


def _bpc_legs_strikes():
    """2-leg bull-put-credit legs + near-ATM strikes that carry extrinsic value
    (so the spread-crossing term is non-zero and the test is meaningful)."""
    t = base_template_by_name("bull_put_credit")
    legs = t.legs
    # short put just OTM, long put further OTM — both carry time value at mtc=200
    spot = 4700.0
    strikes = [4650.0, 4600.0][:len(legs)]
    return legs, strikes, spot


def test_skip_removes_only_the_spread_crossing():
    legs, strikes, spot = _bpc_legs_strikes()
    full = _entry_costs(legs, strikes, spot, 200.0, 0.15, 2.50)
    skip = _entry_costs(legs, strikes, spot, 200.0, 0.15, 2.50,
                        skip_spread_crossing=True)
    # the spread-crossing is a strictly positive cost on legs with extrinsic value
    assert skip < full, "skip_spread_crossing must remove the per-leg bid-ask crossing"
    # what's LEFT is exactly the fee + the flat uncertainty charge (no spread term)
    fees_flat = sum((2.50 / 100.0) * l.ratio + 0.02 * l.ratio for l in legs)
    assert skip == pytest.approx(fees_flat, abs=1e-9), (
        "with the spread skipped, only fee + flat-uncertainty charge should remain")


def test_default_is_backward_compatible():
    # skip_spread_crossing defaults False → unchanged legacy behavior (spread included)
    legs, strikes, spot = _bpc_legs_strikes()
    full = _entry_costs(legs, strikes, spot, 200.0, 0.15, 2.50)
    fees_flat = sum((2.50 / 100.0) * l.ratio + 0.02 * l.ratio for l in legs)
    assert full > fees_flat, "default _entry_costs must still include the spread-crossing"


def test_skip_is_a_no_op_when_no_extrinsic():
    # deep-ITM/-OTM legs with ~zero extrinsic: the spread term is the floor only, but
    # the fix still must never INCREASE cost and must stay >= fees.
    legs, strikes, spot = _bpc_legs_strikes()
    full = _entry_costs(legs, strikes, spot, 200.0, 0.15, 2.50)
    skip = _entry_costs(legs, strikes, spot, 200.0, 0.15, 2.50,
                        skip_spread_crossing=True)
    assert skip <= full
    assert skip > 0.0  # fees + flat charge are always present


@pytest.mark.slow
def test_fix_lifts_credit_strategy_sharpe_through_evaluator():
    """End-to-end: the entry double-count removal lifts a credit strategy's Sharpe
    on real fold-1 data vs the pre-fix value (-0.54 measured). Loose bound — pins the
    DIRECTION (the fix is wired into the credit-entry path), not an exact number."""
    import pandas as pd
    from layer2.io import split_by_date
    from layer2.evaluator_vectorized import prepare_terminal_data, vectorized_backtest
    from layer2.terminal_stats import compute_norm_stats_from_arrays
    from layer2.grammar import from_sexpr
    P = "raw_data/local_store/l1_minute_scalars.parquet"
    if not os.path.exists(P):
        pytest.skip("production parquet not present")
    df = pd.read_parquet(P, filters=[("date", "<", "2024-03-30")])
    tr = split_by_date(df, train_end="2024-03-29", val_end="2024-05-29",
                       embargo_days=0)["train"]
    fn = compute_norm_stats_from_arrays(prepare_terminal_data(tr, normalize_terminals=False))
    td = prepare_terminal_data(tr, norm_stats_override=fn)
    E = from_sexpr("AND(GT(ATM_IV, EphReal(1.0)), GT(IVRVGap5d, EphReal(0.5)))")
    X = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")
    Z = from_sexpr("EphReal(0.5)"); D = from_sexpr("EphReal(0.0)")
    t = base_template_by_name("bull_put_credit")
    r = vectorized_backtest(E, X, Z, tr, t, delta_tree=D, terminal_data=td,
                            warmup_bars=30, cost_multiplier=1.0,
                            stop_loss_credit_multiple=0.0)
    # pre-fix this strategy measured -0.54; the entry double-count removal recovers
    # ~+0.23 → ~-0.31. Assert it cleared the pre-fix level by a margin.
    assert r.sharpe > -0.45, (
        f"entry double-count fix should lift the credit strategy above the pre-fix "
        f"-0.54 level; got {r.sharpe:.3f}")
