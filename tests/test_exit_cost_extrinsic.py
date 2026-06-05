"""Exit/entry cost must charge the bid-ask spread on EXTRINSIC value only,
floored at a modeled per-leg minimum that scales with time-of-day.

Two coupled behaviors are pinned here:

1. EXTRINSIC-ONLY spread (F-fix). A deep-ITM option settles at parity — you do
   not cross the spread on its intrinsic value. The pre-fix `_entry_costs`
   multiplied the relative spread by the FULL mid (intrinsic + extrinsic), which
   fabricated phantom exit costs on big-move days: an RPB FOMC gap-down was
   charged ~$1136/share vs the correct ~$0.14, over-stating proxy losses up to 8x.

2. FIXED-$ FLOOR (2026-06-01 audit) scaled by tod_mult. Charging spread on
   extrinsic-only collapsed deep-ITM exits to the fee-floor — OPTIMISTIC on the
   losing tail. Real SPX spreads are a roughly fixed DOLLAR width, so an active
   close still crosses the bid-ask even at extrinsic≈0. The floor is
   `_MIN_SPREAD_COST_PER_LEG * tod_mult` so a late-day deep-ITM exit (floor binds)
   pays the wider near-close spread, never the midday minimum.
"""
from layer2.evaluator_vectorized import (
    _entry_costs, _option_value, _MIN_SPREAD_COST_PER_LEG)
from layer2.templates import Leg

# Per-leg fixed components (in $/share): exchange fee + flat uncertainty charge.
_FEE = 2.50 / 100.0   # $2.50/contract ÷ 100 SPX multiplier
_FLAT = 0.02


def test_deep_itm_leg_cost_is_extrinsic_bounded():
    """A deep-ITM put (spot far below strike) has ~zero extrinsic -> spread cost
    is the floor, NOT a fraction of its large intrinsic-heavy mid (~150)."""
    spot, strike = 5730.0, 5880.0          # put 150 pts ITM
    legs = (Leg("put", -0.40, qty_sign=-1, ratio=1),)
    # mtc=25 -> last-30-min tod_mult=2.0 -> floor = 0.10*2.0 = 0.20
    cost = _entry_costs(legs, [strike], spot, 25.0, 0.30, 2.50, atm_spread=0.03)
    expected = _FEE + _MIN_SPREAD_COST_PER_LEG * 2.0 + _FLAT  # 0.245
    assert abs(cost - expected) < 1e-6, (
        f"deep-ITM cost {cost:.4f} should be fee+floor(x2.0)+flat={expected:.4f} "
        f"(extrinsic≈0, NOT a fraction of the 150 intrinsic)")


def test_floor_scales_with_time_of_day():
    """The fixed-$ floor must widen toward the close (tod_mult U-shape), so a
    deep-ITM exit where the floor BINDS is never under-charged at the midday rate."""
    spot, strike = 5730.0, 5880.0
    legs = (Leg("put", -0.40, qty_sign=-1, ratio=1),)
    midday = _entry_costs(legs, [strike], spot, 200.0, 0.30, 2.50, atm_spread=0.03)
    close = _entry_costs(legs, [strike], spot, 10.0, 0.30, 2.50, atm_spread=0.03)
    assert abs(midday - (_FEE + _MIN_SPREAD_COST_PER_LEG * 1.0 + _FLAT)) < 1e-6, \
        f"midday floor should be x1.0: {midday:.4f}"
    assert abs(close - (_FEE + _MIN_SPREAD_COST_PER_LEG * 2.5 + _FLAT)) < 1e-6, \
        f"last-15-min floor should be x2.5: {close:.4f}"
    assert close > midday, "near-close deep-ITM exit must cost MORE than midday (no optimism)"


def test_outlier_spread_does_not_explode_on_deep_itm():
    """The phantom-cost case: outlier relative spread (1.468) on a deep-ITM leg
    must NOT produce a huge cost (pre-fix: ~1136 $/share)."""
    spot, strike = 5730.0, 5880.0
    legs = (Leg("put", -0.40, qty_sign=-1, ratio=1),
            Leg("put", -0.20, qty_sign=+1, ratio=2))
    cost = _entry_costs(legs, [strike, 5865.0], spot, 25.0, 0.30, 2.50, atm_spread=1.468)
    assert cost < 1.0, f"deep-ITM outlier-spread cost {cost:.3f} must stay bounded (was ~1136)"


def test_otm_leg_cost_retains_full_spread_on_mid():
    """An OTM leg has extrinsic == mid (intrinsic 0), so the spread is charged on
    the FULL mid (the fix must NOT collapse it to the floor)."""
    spot, strike = 5900.0, 5850.0          # put 50 pts OTM
    legs = (Leg("put", -0.25, qty_sign=-1, ratio=1),)
    mid = _option_value(spot, strike, 200.0, 0.15, False)
    assert mid > 0.05, "test setup: OTM put should have nonzero mid"
    cost = _entry_costs(legs, [strike], spot, 200.0, 0.15, 2.50, atm_spread=0.03)
    # cost = fee + max(spread_frac*rel_spread*mid, floor) + flat; the variable
    # term on the full mid must clear the midday floor for this nonzero-mid leg.
    assert cost > _FEE + _MIN_SPREAD_COST_PER_LEG + _FLAT - 1e-9, \
        "OTM cost should retain the full spread-on-mid term (>= floor)"
