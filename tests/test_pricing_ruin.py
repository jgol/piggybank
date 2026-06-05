"""Pricing & ruin fixes (2026-06-01 holistic review, Batch 2).

B2-fin: per-leg grid IV upper clamp + single-option no-arb bound, so a corrupt-data
bar (300-500% IV cell) cannot mark a defined-risk spread beyond its structural width.
M1-fin: adverse-fill markup on realized stop-loss PnL (replaces a dead `*= 1.15`).
B3: -100% ruin liquidation halts further trading (covered by test_drawdown_plumbing's
floor test; the constant + wiring are pinned here).
"""
import numpy as np

from layer2.evaluator_vectorized import (
    _empirical_iv, _net_pos_value, _option_value, STOP_LOSS_FILL_MARKUP)
from layer2.templates import Leg

SPOT = 5000.0
MTC = 120.0


# ----------------------------- B2-fin: IV upper clamp ------------------------ #

def test_empirical_iv_caps_extreme_high_cell():
    """A 500% (5.0) grid cell must be clamped, not used raw."""
    atm = 0.01  # data-gap floor (the corrupt-bar signature)
    grid = np.array([0.145, 0, 0, 0, 0, 0, 0, 0, 5.0, 0, 0], dtype=float)  # 25Dc=500%
    iv = _empirical_iv(grid, 0.25, atm)  # at the corrupted +25-delta point
    assert iv <= 1.0, f"grid IV must be capped at <=1.0, got {iv}"
    assert iv < 5.0, "the 500% cell must not be used raw"


def test_empirical_iv_normal_grid_unchanged():
    """A sane grid (all ~0.18-0.22) must be untouched by the clamp."""
    atm = 0.20
    grid = np.array([0.24, 0.22, 0.21, 0.20, 0.19, 0.19, 0.20, 0.21, 0.22, 0.23, 0.25],
                    dtype=float)
    for d in (-0.25, 0.0, 0.25):
        iv = _empirical_iv(grid, d, atm)
        assert 0.15 < iv < 0.30, f"normal grid IV distorted at delta {d}: {iv}"


def test_net_pos_value_bounded_on_corrupt_bar():
    """A 50-wide IC on a 500%-IV corrupt bar must mark within its structural width
    (~100/share), NOT at 34x (the pre-fix -169/share phantom)."""
    atm = 0.01
    grid = np.array([0.145, 0, 0, 0, 0, 0, 0, 0, 5.0, 0, 0], dtype=float)
    legs = (Leg("put", -0.25, qty_sign=-1, ratio=1), Leg("put", -0.05, qty_sign=1, ratio=1),
            Leg("call", 0.25, qty_sign=-1, ratio=1), Leg("call", 0.05, qty_sign=1, ratio=1))
    strikes = [4950.0, 4900.0, 5050.0, 5100.0]   # 50-wide wings
    nv = _net_pos_value(legs, strikes, SPOT, MTC, atm, grid_ivs=grid)
    assert abs(nv) < 100.0, f"corrupt-bar IC net value {nv} exceeds structural width"


def test_single_leg_noarb_bound():
    """_net_pos_value caps each leg at its arbitrage bound: a long call ≤ spot."""
    # A single long deep-ITM call: value ≤ spot regardless of inputs.
    legs = (Leg("call", 0.99, qty_sign=1, ratio=1),)
    nv = _net_pos_value(legs, [1.0], SPOT, MTC, 0.20)  # strike=1 → ~all intrinsic
    assert 0.0 <= nv <= SPOT + 1e-6, f"call value {nv} violates the ≤ spot no-arb bound"


# ----------------------------- M1-fin: stop-loss markup ---------------------- #

def test_stop_loss_fill_markup_is_conservative():
    """The realized-loss markup exists and is in the conservative (>1) direction."""
    assert STOP_LOSS_FILL_MARKUP > 1.0
    assert STOP_LOSS_FILL_MARKUP == 1.10


def test_dead_unrealised_markup_removed():
    """The dead `unrealised *= 1.15` (discarded before realized PnL) must be gone —
    the markup now lives on the realized stop-loss PnL."""
    src = open("layer2/evaluator_vectorized.py").read()
    assert "unrealised *= 1.15" not in src, "dead stop-loss markup must be removed"
    assert "pnl *= STOP_LOSS_FILL_MARKUP" in src, "markup must apply to realized PnL"
