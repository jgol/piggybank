"""Regression test for the HIGH (2026-06-03 code review): codegen
Lag/Delta(UnrealizedProfitPct) returned a CONSTANT 0 in QC because the UPP value
was injected into `s` AFTER the per-bar buffer-update loop, so it was never
pushed into self._buffers — the temporal read saw an empty deque. The proxy
maintains a dedicated PER-TRADE `upp_hist` (cleared on each entry,
evaluator_vectorized.py:693/1775/2243). The fix gives codegen the same:
self._upp_hist (init in Initialize, cleared in _open_position, appended each
in-position bar in _on_bar) read by _lag/_delta via a UnrealizedProfitPct
special-case.

These tests exercise the REAL generated methods via build_codegen_instance, so a
regression that silently stops populating the buffer (back to constant 0) fails.
"""
import os
import sys
from collections import deque

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer3.diagnostics.operator_parity import (  # noqa: E402
    StrategySpec, build_codegen_instance,
)

_WINDOW = ("2025-01-02", "2025-01-31")

# Exit tree references BOTH Lag and Delta of UnrealizedProfitPct so the codegen
# must emit self._lag/_delta("UnrealizedProfitPct", ...).
_SPEC = StrategySpec(
    strategy_id="upp_buf", template_name="bull_put_credit_standard",
    entry_sexpr="GT(ATM_IV, EphReal(0.0))",
    exit_sexpr="OR(GT(Lag(UnrealizedProfitPct, EphInt(2)), EphReal(0.5)), "
               "GT(Delta(UnrealizedProfitPct, EphInt(3)), EphReal(0.1)))",
    size_sexpr="EphReal(0.3)", is_credit=True)


def _inst():
    inst, code = build_codegen_instance(_SPEC, _WINDOW, norm_serial={})
    return inst, code


# --------------------------------------------------------------------------
# 1) WIRING — the buffer must be initialized, cleared on entry, appended in-loop.
# (Catches the original "never populated" bug at the text level.)
# --------------------------------------------------------------------------
def test_upp_hist_initialized_cleared_and_appended():
    _, code = _inst()
    init = "self._upp_hist = deque(maxlen=self.MAX_LAG + 1)"
    clear = "self._upp_hist.clear()"
    append = "self._upp_hist.append(s[\"UnrealizedProfitPct\"])"
    assert init in code, "self._upp_hist must be initialized in Initialize"
    assert clear in code, "self._upp_hist.clear() must run on each new entry"
    assert append in code, "self._upp_hist.append must run each in-position bar"
    # init BEFORE both clear and append (no read-before-init)
    assert code.index(init) < code.index(append)
    assert code.index(init) < code.index(clear)
    # the append must be guarded by the in-position branch (not flat bars):
    # it sits AFTER the `s["UnrealizedProfitPct"] = _upp_pnl / ...` assignment.
    assert code.index('s["UnrealizedProfitPct"] = _upp_pnl') < code.index(append)


def test_lag_delta_special_case_upp_in_source():
    _, code = _inst()
    # _lag/_delta must branch on the UPP terminal to read self._upp_hist...
    assert 'if name == "UnrealizedProfitPct":' in code, (
        "_lag/_delta must special-case UnrealizedProfitPct")
    assert "buf = self._upp_hist" in code, (
        "_lag must read the per-trade self._upp_hist for UnrealizedProfitPct")
    # ...and gate that read on position state (UPP=0 baseline while flat).
    assert "if not self._position_open:" in code, (
        "_lag/_delta UPP branch must return 0 while flat (proxy parity)")


# --------------------------------------------------------------------------
# 2) BEHAVIORAL — the REAL generated _lag reads the per-trade history.
# --------------------------------------------------------------------------
def test_lag_upp_reads_per_trade_history_not_constant_zero():
    inst, _ = _inst()
    inst._position_open = True           # in-position: UPP reads the live history
    # The original bug: _lag read self._buffers (empty for UPP) -> 0. Poison the
    # generic buffer to PROVE the read path now uses self._upp_hist instead.
    inst._buffers["UnrealizedProfitPct"] = deque([99.0, 99.0, 99.0],
                                                 maxlen=inst.MAX_LAG + 1)
    inst._upp_hist.clear()
    for v in (0.1, 0.2, 0.3, 0.4):       # 4 in-position bars, most-recent last
        inst._upp_hist.append(v)
    assert inst._lag("UnrealizedProfitPct", 0) == pytest.approx(0.4)
    assert inst._lag("UnrealizedProfitPct", 1) == pytest.approx(0.3)
    assert inst._lag("UnrealizedProfitPct", 3) == pytest.approx(0.1)
    # k >= len(upp_hist) -> 0.0 ("before this trade started", proxy line 733)
    assert inst._lag("UnrealizedProfitPct", 4) == 0.0
    # NOT the poisoned generic-buffer value
    assert inst._lag("UnrealizedProfitPct", 0) != 99.0


def test_non_upp_terminal_still_reads_generic_buffer():
    inst, _ = _inst()
    inst._buffers["ATM_IV"] = deque([0.10, 0.20, 0.30], maxlen=inst.MAX_LAG + 1)
    assert inst._lag("ATM_IV", 0) == pytest.approx(0.30)
    assert inst._lag("ATM_IV", 2) == pytest.approx(0.10)


# --------------------------------------------------------------------------
# 3) BEHAVIORAL — Delta(UPP) uses the WITHIN-DAY mask, not trade-history length.
# This is the subtle proxy divergence: in the first bars of a TRADE (but past
# the first k bars of the DAY), Delta(UPP,k) == cur - 0 == cur, NOT 0.
# --------------------------------------------------------------------------
def _set_within_day_pos(inst, n_bars):
    """within_day_pos+1 == len(_raw_RawSpread); the _delta UPP mask reads it.
    Also marks the instance in-position — every caller below tests the IN-position
    UPP path (the proxy uses UPP=0 while flat; that flat gate is tested in §5)."""
    inst._buffers["_raw_RawSpread"] = deque([0.02] * n_bars, maxlen=inst.MAX_LAG + 1)
    inst._position_open = True


def test_delta_upp_mid_trade_returns_cur_not_zero():
    inst, _ = _inst()
    _set_within_day_pos(inst, 50)        # well past the first k bars of the day
    inst._upp_hist.clear()
    inst._upp_hist.append(0.1)
    inst._upp_hist.append(0.2)           # only 2 bars into the trade
    # k=3 >= len(upp_hist)=2: lagged term is 0 (before trade), but the DAY mask
    # does NOT fire (within_day_pos=49 >= 3) -> Delta == cur - 0 == cur == 0.2.
    # The pre-fix generic `len(buf) <= k` mask would have wrongly returned 0.0.
    assert inst._delta("UnrealizedProfitPct", 3) == pytest.approx(0.2)


def test_delta_upp_normal_mid_trade_difference():
    inst, _ = _inst()
    _set_within_day_pos(inst, 50)
    inst._upp_hist.clear()
    for v in (0.1, 0.2, 0.3, 0.4):
        inst._upp_hist.append(v)
    # cur(0.4) - lag2(0.2) == 0.2
    assert inst._delta("UnrealizedProfitPct", 2) == pytest.approx(0.2)


def test_delta_upp_day_boundary_mask_fires():
    inst, _ = _inst()
    _set_within_day_pos(inst, 2)         # only the 2nd bar of the DAY
    inst._upp_hist.clear()
    inst._upp_hist.append(0.1)
    inst._upp_hist.append(0.2)
    # within_day_pos=1 < k=3 -> masked to 0.0 (proxy line 760)
    assert inst._delta("UnrealizedProfitPct", 3) == 0.0


# --------------------------------------------------------------------------
# 4) BEHAVIORAL — clearing on entry prevents cross-trade leakage.
# --------------------------------------------------------------------------
def test_no_cross_trade_leak_after_clear():
    inst, _ = _inst()
    _set_within_day_pos(inst, 50)
    inst._upp_hist.clear()
    for v in (0.1, 0.2, 0.3, 0.4):       # trade 1
        inst._upp_hist.append(v)
    inst._upp_hist.clear()               # new entry (proxy upp_hist.clear())
    inst._upp_hist.append(0.9)           # trade 2, bar 1
    # Lag(UPP,1) must be 0 (only 1 bar into trade 2), NOT trade 1's 0.4.
    assert inst._lag("UnrealizedProfitPct", 1) == 0.0
    assert inst._lag("UnrealizedProfitPct", 0) == pytest.approx(0.9)
    # Delta back into the prior trade must also be 0-anchored, not a leak.
    assert inst._delta("UnrealizedProfitPct", 2) == pytest.approx(0.9)  # cur-0


# --------------------------------------------------------------------------
# 5) BEHAVIORAL — UPP is the 0 baseline while FLAT (review HIGH, 2026-06-03).
# The proxy substitutes the live per-trade UPP ONLY into the EXIT tree
# in-position; entry/size/delta trees see the zeros array. So Lag/Delta(UPP)
# must read 0 while flat — NOT the prior trade's stale _upp_hist tail.
# --------------------------------------------------------------------------
def test_lag_upp_is_zero_while_flat_despite_stale_history():
    inst, _ = _inst()                    # harness default: _position_open = False
    inst._upp_hist.extend([0.1, 0.2, 0.3])   # prior trade's residual tail
    assert inst._position_open is False
    assert inst._lag("UnrealizedProfitPct", 0) == 0.0
    assert inst._lag("UnrealizedProfitPct", 1) == 0.0
    assert inst._lag("UnrealizedProfitPct", 2) == 0.0


def test_delta_upp_is_zero_while_flat_despite_stale_history():
    inst, _ = _inst()
    _set_within_day_pos(inst, 50)        # (also sets in-position)
    inst._upp_hist.extend([0.1, 0.2, 0.3, 0.4])
    inst._position_open = False          # override: FLAT
    assert inst._delta("UnrealizedProfitPct", 2) == 0.0


def test_entry_tree_lag_upp_sees_zero_while_flat_not_stale_tail():
    """The review's exact gap: Lag(UPP) in an ENTRY tree, evaluated while FLAT, must
    read 0 (proxy's entry-tree UPP=0), not the prior trade's tail — else QC enters on
    a stale signal the proxy never saw, breaking proxy↔QC trade-timing parity."""
    spec = StrategySpec(
        strategy_id="upp_entry", template_name="bull_put_credit_standard",
        # entry fires iff Lag(UPP,2) > 0.5
        entry_sexpr="GT(Lag(UnrealizedProfitPct, EphInt(2)), EphReal(0.5))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.3)", is_credit=True)
    inst, _ = build_codegen_instance(spec, _WINDOW, norm_serial={})
    inst._upp_hist.extend([0.6, 0.7, 0.8])   # stale tail; lag2 would be 0.6 > 0.5
    # FLAT (the bar where entry is actually decided): must NOT misfire.
    inst._position_open = False
    assert bool(inst._eval_entry({})) is False, (
        "flat entry tree saw the stale 0.6 (>0.5) instead of the proxy's UPP=0")
    # Sanity: in-position the same history reads live (0.6 at lag2 -> fires). This
    # bar is never ACTED on for entry (we're already in a position) but confirms the
    # gate is position-conditional, not a blanket zero.
    inst._position_open = True
    assert bool(inst._eval_entry({})) is True
