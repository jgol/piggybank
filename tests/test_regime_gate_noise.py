"""Noise-aware per-regime gate (2026-06-04).

The per-regime gate (`_regime_gate_fails`) used a HARD threshold: reject if the
strategy's per-regime trade-level Sharpe < the random-entry bar. Both are noisy
estimates (Lo 2002 SE ≈ sqrt((1+0.5·SR²)/n)), so the gate decided feasibility
WITHIN sampling noise — a ~0.3-SE miss sentineled the whole iron_condor population
(folds 2/3 aborted, 0/256). The fix rejects only when worse than the bar by more
than margin_k × the combined SE. These tests pin that behavior with the measured
iron_condor (noise miss → must pass at k=1) and ratio_put_backspread (genuine
−2.3-SE miss → must still fail) numbers.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.fitness import _regime_gate_fails  # noqa: E402


def _pnls(n, target_sharpe):
    """Deterministic pnl vector with trade-level Sharpe == target (mean/std)."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std()          # mean 0, std 1
    return z + target_sharpe              # mean=target, std=1 → sharpe=target


def _result(high_pnls):
    """All trades land in the 'high' regime (entry_bar=50, where ATM_IV is high)."""
    trades = [types.SimpleNamespace(entry_bar=50, pnl=float(p)) for p in high_pnls]
    return types.SimpleNamespace(trades=trades)


def _td():
    iv = np.zeros(100, dtype=np.float64)
    iv[50] = 5.0                          # bar 50 → above q67 → 'high' regime
    return {"ATM_IV": iv}


TERC = (-1.0, 1.0)                        # q33, q67


def test_iron_condor_noise_miss_passes_at_k1_but_old_threshold_rejects():
    # Measured: IC fold-2 high-vol seed Sharpe −0.214 (n=45) vs random bar −0.171.
    res = _result(_pnls(45, -0.214))
    bars = {"high": -0.171}
    ses = {"high": 0.119}                 # bar SE from its own ~72 trades
    common = dict(regime_terciles=TERC, random_entry_regime_sharpes=bars,
                  random_entry_regime_se=ses)
    # OLD hard threshold (k=0): −0.214 < −0.171 → rejected (the bug).
    assert _regime_gate_fails(res, _td(), margin_k=0.0, **common) is True
    # NOISE-AWARE (k=1): the 0.043 gap is < 1 combined SE → NOT rejected.
    assert _regime_gate_fails(res, _td(), margin_k=1.0, **common) is False


def test_rpb_genuine_miss_still_rejected_at_k1():
    # Measured: RPB fold-2 mid seed Sharpe −0.274 (n=135) vs bar −0.073 → −2.3 SE.
    res = _result(_pnls(135, -0.274))
    # put these trades in 'high' for the harness (regime label irrelevant to the math)
    bars = {"high": -0.073}
    ses = {"high": 0.09}
    assert _regime_gate_fails(
        res, _td(), regime_terciles=TERC, random_entry_regime_sharpes=bars,
        random_entry_regime_se=ses, margin_k=1.0) is True


def test_margin_scales_monotonically_with_k():
    res = _result(_pnls(45, -0.30))
    bars = {"high": -0.171}
    ses = {"high": 0.119}
    common = dict(regime_terciles=TERC, random_entry_regime_sharpes=bars,
                  random_entry_regime_se=ses)
    assert _regime_gate_fails(res, _td(), margin_k=0.0, **common) is True
    # a large enough margin never rejects
    assert _regime_gate_fails(res, _td(), margin_k=5.0, **common) is False


def test_se_widens_with_fewer_trades_so_small_samples_are_harder_to_reject():
    # Same Sharpe miss, but n=12 (wide SE) vs n=200 (tight SE). At k=1 the
    # small sample is INSIDE noise (pass); the large sample is a real miss (fail).
    bars = {"high": -0.10}
    ses = {"high": 0.0}
    common = dict(regime_terciles=TERC, random_entry_regime_sharpes=bars,
                  random_entry_regime_se=ses, margin_k=1.0)
    small = _regime_gate_fails(_result(_pnls(12, -0.30)), _td(), **common)
    large = _regime_gate_fails(_result(_pnls(200, -0.30)), _td(), **common)
    assert small is False, "n=12 miss is within sampling noise → should pass"
    assert large is True, "n=200 miss is statistically real → should fail"


def test_below_min_regime_trades_skips_gate():
    # Fewer than REGIME_MIN_TRADES (10) high-regime trades → gate does not apply.
    res = _result(_pnls(8, -5.0))         # catastrophic but only 8 trades
    bars = {"high": -0.10}
    assert _regime_gate_fails(
        res, _td(), regime_terciles=TERC, random_entry_regime_sharpes=bars,
        random_entry_regime_se={"high": 0.0}, margin_k=1.0) is False


def test_returns_bool_when_no_baseline_provided():
    # Backward-compat: no se dict, no margin → falls back to −0.5 floor with k=1.
    res = _result(_pnls(40, -0.9))        # well below −0.5 even with margin
    assert _regime_gate_fails(res, _td(), regime_terciles=TERC) is True
