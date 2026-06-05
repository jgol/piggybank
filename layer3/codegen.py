"""Deterministic GP tree → QuantConnect Python code generator.

Translates evolved s-expression trees + template leg definitions into complete
QCAlgorithm classes WITHOUT any LLM involvement. The generated code is a
self-contained Python file that can be uploaded to QuantConnect, compiled,
and backtested.

Condition A (scalar-only) first — no encoder/embedding infrastructure.

Architecture:
    1. Code templates (string) for QC boilerplate (Initialize, OnData, etc.)
    2. Per-template leg definitions for multi-leg option execution
    3. Mechanical tree-walker that emits Python expressions from Node trees

Usage:
    from layer3.codegen import generate_qc_algorithm
    from layer2.grammar import from_sexpr

    code = generate_qc_algorithm(
        strategy_id="ic_001",
        template_name="iron_condor",
        entry_sexpr="GT(ATM_IV, EphReal(0.18))",
        exit_sexpr="LT(MinutesToClose, EphReal(30))",
        size_sexpr="EphReal(1.0)",
    )
    # code is a complete QCAlgorithm Python file string
"""
from __future__ import annotations

import ast
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from layer2.grammar import (
    FuncNode, GType, Node, TermNode, from_sexpr, to_str,
)
from layer2.terminal_stats import TERMINAL_NORM_STATS
from layer3.bs_iv import qc_method_source as _bs_qc_method_source

# ---------------------------------------------------------------------------
# Template leg definitions (mirror layer2/templates.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QCLeg:
    option_type: str   # "call" or "put"
    delta_target: float
    qty_sign: int      # +1 long, -1 short
    ratio: int = 1


TEMPLATE_LEGS: Dict[str, List[QCLeg]] = {
    # Standard width
    "iron_condor_standard": [
        QCLeg("call", +0.25, -1), QCLeg("call", +0.10, +1),
        QCLeg("put",  -0.25, -1), QCLeg("put",  -0.10, +1),
    ],
    "iron_butterfly_standard": [
        QCLeg("call", +0.50, -1), QCLeg("call", +0.10, +1),
        QCLeg("put",  -0.50, -1), QCLeg("put",  -0.10, +1),
    ],
    "bull_put_credit_standard": [
        QCLeg("put", -0.25, -1), QCLeg("put", -0.10, +1),
    ],
    "bear_call_credit_standard": [
        QCLeg("call", +0.25, -1), QCLeg("call", +0.10, +1),
    ],
    # Narrow width
    "iron_condor_narrow": [
        QCLeg("call", +0.35, -1), QCLeg("call", +0.15, +1),
        QCLeg("put",  -0.35, -1), QCLeg("put",  -0.15, +1),
    ],
    "iron_butterfly_narrow": [
        QCLeg("call", +0.50, -1), QCLeg("call", +0.20, +1),
        QCLeg("put",  -0.50, -1), QCLeg("put",  -0.20, +1),
    ],
    "bull_put_credit_narrow": [
        QCLeg("put", -0.35, -1), QCLeg("put", -0.15, +1),
    ],
    "bear_call_credit_narrow": [
        QCLeg("call", +0.35, -1), QCLeg("call", +0.15, +1),
    ],
    # Wide width
    "iron_condor_wide": [
        QCLeg("call", +0.15, -1), QCLeg("call", +0.10, +1),
        QCLeg("put",  -0.15, -1), QCLeg("put",  -0.10, +1),
    ],
    "bull_put_credit_wide": [
        QCLeg("put", -0.15, -1), QCLeg("put", -0.10, +1),
    ],
    "bear_call_credit_wide": [
        QCLeg("call", +0.15, -1), QCLeg("call", +0.10, +1),
    ],
    # Debit (unchanged)
    "bull_call_debit": [
        QCLeg("call", +0.50, +1), QCLeg("call", +0.20, -1),
    ],
    "bear_put_debit": [
        QCLeg("put", -0.50, +1), QCLeg("put", -0.20, -1),
    ],
}

# Backward-compatible aliases (old names without _standard suffix)
TEMPLATE_LEGS["iron_condor"] = TEMPLATE_LEGS["iron_condor_standard"]
TEMPLATE_LEGS["iron_butterfly"] = TEMPLATE_LEGS["iron_butterfly_standard"]
TEMPLATE_LEGS["bull_put_credit"] = TEMPLATE_LEGS["bull_put_credit_standard"]
TEMPLATE_LEGS["bear_call_credit"] = TEMPLATE_LEGS["bear_call_credit_standard"]

# Encoder-augmented _enc templates removed (2026-05-11) for ablation fairness.
# See layer2/templates.py module docstring for rationale.

# Level B base template delta ranges: (min_delta, max_delta, wing_offset, delta_fixed)
# Keys match Template.name values from templates.py base factories.
BASE_TEMPLATE_DELTA_RANGES: Dict[str, tuple] = {
    "iron_condor":           (0.15, 0.40, 0.15, False),
    "iron_butterfly":        (0.10, 0.30, None,  True),  # short fixed at 0.50
    "bull_put_credit":       (0.15, 0.40, 0.15, False),
    "bear_call_credit":      (0.15, 0.40, 0.15, False),
    "ratio_put_backspread":  (0.25, 0.50, 0.20, False),
}


def _build_dynamic_legs(template_name: str, delta_value: float) -> List[QCLeg]:
    """Build legs from base template + GP-discovered delta_value [0,1]."""
    min_d, max_d, wing_offset, delta_fixed = BASE_TEMPLATE_DELTA_RANGES[template_name]
    short_delta = min_d + delta_value * (max_d - min_d)

    if template_name == "iron_condor":
        long_delta = max(short_delta - wing_offset, 0.05)
        return [
            QCLeg("call", +round(short_delta, 4), -1),
            QCLeg("call", +round(long_delta, 4),  +1),
            QCLeg("put",  -round(short_delta, 4), -1),
            QCLeg("put",  -round(long_delta, 4),  +1),
        ]
    elif template_name == "iron_butterfly":
        # Short fixed at ATM (0.50); delta_tree controls wing delta
        wing_delta = short_delta  # maps to (0.10, 0.30) range
        return [
            QCLeg("call", +0.50, -1),
            QCLeg("call", +round(wing_delta, 4), +1),
            QCLeg("put",  -0.50, -1),
            QCLeg("put",  -round(wing_delta, 4), +1),
        ]
    elif template_name == "bull_put_credit":
        long_delta = max(short_delta - wing_offset, 0.05)
        return [
            QCLeg("put", -round(short_delta, 4), -1),
            QCLeg("put", -round(long_delta, 4),  +1),
        ]
    elif template_name == "bear_call_credit":
        long_delta = max(short_delta - wing_offset, 0.05)
        return [
            QCLeg("call", +round(short_delta, 4), -1),
            QCLeg("call", +round(long_delta, 4),  +1),
        ]
    elif template_name == "ratio_put_backspread":
        # Sell 1 near-ATM put, buy 2 OTM puts.
        # short_delta = GP-controlled via delta_range (0.25-0.50)
        # long_delta = short_delta - wing_offset (0.20 below short)
        long_delta = max(short_delta - wing_offset, 0.05)
        return [
            QCLeg("put", -round(short_delta, 4), -1, ratio=1),  # sell 1
            QCLeg("put", -round(long_delta, 4),  +1, ratio=2),  # buy 2
        ]
    else:
        raise ValueError(f"Unknown base template: {template_name}")


# ---------------------------------------------------------------------------
# Tree-to-Python expression walker
# ---------------------------------------------------------------------------

def _node_to_python(node: Node) -> str:
    """Convert a Node tree to a Python expression string.

    Maps each GP operator to its Python equivalent, matching the semantics
    in layer2/evaluator.py exactly.
    """
    if isinstance(node, TermNode):
        name = node.name
        v = node.value
        # Ephemeral constants
        if name == "EphReal":
            return repr(float(v))
        if name == "EphInt":
            return repr(int(v))
        # Side/Regime enum literals
        if name in ("CALL", "PUT", "NEUTRAL"):
            return repr(name)
        if name in ("LOW_VOL", "MID_VOL", "HIGH_VOL_STRESSED", "HIGH_VOL_PREMIUM"):
            return repr(name)
        # BLOCKER B2-int (2026-06-01 holistic review; EMB_ added per the campaign
        # review): FAIL LOUD on every L1-encoder output terminal QuantConnect cannot
        # compute live. Three families, all encoder-derived: Pred* (PredRV15/30,
        # PredSpread, PredGammaAccel, PredSmileConvexity, PredJump, PredFlowToxicity,
        # PredRegime), Regime* (RegimeAboveLow/IsHigh/IsPremium, RegimeProb0..3), and
        # EMB_* (the bare typed-vector embeddings EMB_GRID/EMB_SPX/EMB_VAR_*, which
        # parse to a TermNode and so BYPASS the FuncNode-level EmbProj fail-loud). The
        # prior `s.get(name, 0.0)` silently resolved them to RAW 0.0 in QC while the
        # proxy used the NORMALIZED value — a silent train-serve break that froze the
        # terminal and broke proxy↔QC transfer for probes-only / emb-only / real-l1
        # champions. A strategy depending on any encoder terminal MUST NOT be
        # code-generated until the QC side computes+normalizes it. The prefix test is
        # safe: no QC-computable scalar starts with Pred / Regime / EMB_.
        if name.startswith("Pred") or name.startswith("Regime") or name.startswith("EMB_"):
            raise NotImplementedError(
                f"Terminal '{name}' is an L1-encoder/probe/embedding output that "
                f"QuantConnect cannot compute live (no encoder at serving time). "
                f"Deploying it would silently resolve to raw 0.0 vs the normalized proxy "
                f"value — a train-serve break. Strip encoder terminals (scalar-only) or "
                f"implement+normalize them on the QC side before deploying."
            )
        # Real market scalars — read from self.scalars dict (computed + normalized in
        # QC's _compute_scalars). .get() default is a defensive fallback only.
        return f's.get("{name}", 0.0)'

    assert isinstance(node, FuncNode)
    name = node.name
    ch = node.children

    # Comparison / Boolean
    if name == "GT":
        return f"({_node_to_python(ch[0])} > {_node_to_python(ch[1])})"
    if name == "LT":
        return f"({_node_to_python(ch[0])} < {_node_to_python(ch[1])})"
    if name == "AND":
        return f"(float({_node_to_python(ch[0])}) > 0.5 and float({_node_to_python(ch[1])}) > 0.5)"
    if name == "OR":
        return f"(float({_node_to_python(ch[0])}) > 0.5 or float({_node_to_python(ch[1])}) > 0.5)"
    if name == "NOT":
        return f"(float({_node_to_python(ch[0])}) <= 0.5)"

    # Arithmetic
    if name == "Add":
        return f"({_node_to_python(ch[0])} + {_node_to_python(ch[1])})"
    if name == "Sub":
        return f"({_node_to_python(ch[0])} - {_node_to_python(ch[1])})"
    if name == "Mul":
        return f"({_node_to_python(ch[0])} * {_node_to_python(ch[1])})"
    if name == "Div":
        # Analytic quotient: a / sqrt(1 + b^2) (Ni et al. 2013)
        a = _node_to_python(ch[0])
        b = _node_to_python(ch[1])
        return f"({a} / math.sqrt(1.0 + ({b}) ** 2))"
    if name == "Sqrt":
        # Protected sqrt = sqrt(|x|): closure-preserving protected operator
        # (Koza, 1992; Poli, Langdon & McPhee, 2008). MUST match both proxy
        # evaluators (evaluator.py:716, evaluator_vectorized.py:331) — using
        # max(0,x) instead would make the deployed program compute a different
        # function than the one that was evolved/selected, i.e. training–serving
        # skew (Sculley et al., 2015). Normalized terminals are routinely
        # negative, so the two diverge on a large fraction of bars.
        return f"math.sqrt(abs({_node_to_python(ch[0])}))"

    # Temporal — Lag/Delta on terminals use rolling buffer.
    # For FuncNode first-child (computed expressions), we generate code that
    # stores the expression result in the buffer under a synthetic key derived
    # from the s-expression, then retrieves lagged/delta from that buffer.
    if name == "Lag":
        lag_val = _node_to_python(ch[1])
        if isinstance(ch[0], TermNode) and ch[0].name in ("EphReal", "EphInt"):
            return _node_to_python(ch[0])
        if isinstance(ch[0], TermNode) and ch[0].ret_type == GType.REAL:
            return f'self._lag("{ch[0].name}", {lag_val})'
        # FuncNode: compute, buffer, and lag
        expr = _node_to_python(ch[0])
        buf_key = f"_expr_{hash(to_str(ch[0])) & 0xFFFFFFFF:08x}"
        return f'self._lag_expr("{buf_key}", {expr}, {lag_val})'

    if name == "Delta":
        lag_val = _node_to_python(ch[1])
        if isinstance(ch[0], TermNode) and ch[0].name in ("EphReal", "EphInt"):
            return "0.0"
        if isinstance(ch[0], TermNode) and ch[0].ret_type == GType.REAL:
            return f'self._delta("{ch[0].name}", {lag_val})'
        # FuncNode: compute, buffer, and delta
        expr = _node_to_python(ch[0])
        buf_key = f"_expr_{hash(to_str(ch[0])) & 0xFFFFFFFF:08x}"
        return f'self._delta_expr("{buf_key}", {expr}, {lag_val})'

    # CrossAbove / CrossBelow — require prev-bar cache
    if name == "CrossAbove":
        ka = to_str(ch[0])
        kb = to_str(ch[1])
        a = _node_to_python(ch[0])
        b = _node_to_python(ch[1])
        return (f'self._cross_above("{_escape(ka)}", "{_escape(kb)}", '
                f'{a}, {b})')
    if name == "CrossBelow":
        ka = to_str(ch[0])
        kb = to_str(ch[1])
        a = _node_to_python(ch[0])
        b = _node_to_python(ch[1])
        return (f'self._cross_below("{_escape(ka)}", "{_escape(kb)}", '
                f'{a}, {b})')

    # Conditional
    if name in ("IfThenElse", "IfSide"):
        cond = _node_to_python(ch[0])
        then = _node_to_python(ch[1])
        else_ = _node_to_python(ch[2])
        return f"({then} if {cond} else {else_})"

    # Regime (Condition A won't have these, but handle for completeness)
    if name == "InRegime":
        return f'(self._current_regime == {_node_to_python(ch[0])})'
    if name == "RegimeIs":
        return f'({_node_to_python(ch[0])} == {_node_to_python(ch[1])})'

    # Unknown — fail loudly rather than silently returning 0.0
    raise NotImplementedError(
        f"Operator {name!r} not supported in codegen. "
        f"Embedding operators require encoder infrastructure (Conditions C/D)."
    )


def _escape(s: str) -> str:
    """Escape a string for use as a Python dict key."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# QC algorithm code template
# ---------------------------------------------------------------------------

_QC_TEMPLATE = '''\
# Auto-generated by layer3/codegen.py — DO NOT EDIT
# Strategy: {strategy_id}
# Template: {template_name}
# Entry:    {entry_sexpr_short}
# Exit:     {exit_sexpr_short}

import math
from collections import deque
from datetime import timedelta

from AlgorithmImports import *


def _norm_ppf(p):
    """Inverse normal CDF (Acklam rational approximation). Pure math, no scipy."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - (c0 + c1 * t + c2 * t * t) /
                 (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (
            1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


class GP_{safe_id}(QCAlgorithm):
    """GP-evolved 0DTE strategy: {template_name}."""

    MAX_LAG = 450  # Must span at least one full trading day (405 bars)
    # so that cross-day Lag/Delta references work throughout the session.
    # The proxy evaluator uses vectorized numpy arrays with no lag cap.
    # With MAX_LAG=30, day-constant terminals like OvernightGap produce
    # Delta()=0 after 31 bars, breaking cross-day entry conditions.
    WARMUP_BARS = 30
    BASE_CONTRACTS = {base_contracts}

    def Initialize(self):
        self.SetStartDate({start_year}, {start_month}, {start_day})
        self.SetEndDate({end_year}, {end_month}, {end_day})
        self.SetCash(100000)

        # SPX index + 0DTE options
        self.spx = self.AddIndex("SPX", Resolution.Minute).Symbol
        option = self.AddIndexOption(self.spx, "SPXW", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-50, 50).Expiration(0, 0))
        self.option_symbol = option.Symbol

        # P0-5: real CBOE VIX + VIX9D — the SAME source the L1 collector used
        # (research_collector.py:164-165) and that VIXSpot/VIXTermSlope/VIXChange/
        # VIXMean5d were normalized against. Proven available in QC backtest
        # (generated_qc/gp_condA_iron_condor.py:53-54). The earlier ATM_IV*100
        # proxy mis-scaled VIXSpot (~10 vs ~17) and froze VIXTermSlope dead --
        # training-serving skew (Sculley et al., 2015).
        self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        self.vix9d = self.AddData(CBOE, "VIX9D", Resolution.Daily).Symbol
        self._curr_day_vix = 0.0
        self._prev_day_vix = 0.0
        # _derived_vix (ATM_IV*100) retained ONLY as an internal IV-scale fallback
        # for spline-fail ATM_IV and BS strike sizing -- never for VIX terminals.
        self._derived_vix = 0.0
        # P0-5b: CBOE VIX source-health guard. If the AddData(CBOE,"VIX")
        # subscription never delivers data in this backtest, vix_spot stays <=0 and
        # the VIX terminals degrade silently -- VIXSpot reverts to the ATM_IV*100
        # proxy, VIXTermSlope/VIXChange/VIXMean5d to frozen constants -- re-injecting
        # the training-serving skew P0-5 removed, with NO error. These flags let us
        # DETECT that (vix_spot forward-fills its last good value, so a
        # never-populated subscription is a clean, latchable signal) and FAIL LOUD
        # (Error + a machine-detectable runtime statistic the L3 pipeline reads)
        # instead of reporting a silently-invalid backtest.
        self._vix_ever_populated = False  # real CBOE VIX seen at least once
        self._vix_source_failed = False   # latched once a dead subscription is confirmed
        self._vix_fallback_bars = 0       # bars that fell back to ATM_IV*100 for VIXSpot
        self._day_count = 0               # distinct trading days elapsed (warmup-robust gate)

        # Fee model: $2.50/contract all-in (IBKR commission + exchange/regulatory
        # + estimated half-spread crossing). Matches proxy evaluator's fee_per_leg.
        # Previous $0.65 was commission-only, causing QC to understate costs by ~12x
        # vs real execution (audit finding: +0.3-0.8 Sharpe optimistic in QC).
        # QC's default margin model bypassed — see NullBuyingPowerModel below.
        def _init_security(sec):
            sec.SetFeeModel(ConstantFeeModel(2.50))
            if sec.Type == SecurityType.IndexOption:
                sec.SetBuyingPowerModel(NullBuyingPowerModel())
        self.SetSecurityInitializer(_init_security)

        # Rolling buffer for Lag/Delta
        self._buffers = {{}}  # name -> deque(maxlen=MAX_LAG+1)
        # Per-TRADE UnrealizedProfitPct history for Lag/Delta(UnrealizedProfitPct).
        # Separate from self._buffers (which is day-cleared) because UPP is a
        # position-aware terminal: the proxy clears it on EACH new entry so a
        # temporal op never reaches back into a PRIOR trade within the same day
        # (evaluator_vectorized.py:693, 2243). Cleared in _open_position, appended
        # each in-position bar in _on_bar.
        self._upp_hist = deque(maxlen=self.MAX_LAG + 1)
        self._prev_eval = {{}}
        self._curr_eval = {{}}
        self._bar_count = 0
        self._cached_chain = None
        self._position_open = False
        self._bars_in_trade = 0      # track holding period
        self._open_fail_cooldown = 0  # cooldown after failed open
        self._trades_today = 0       # daily trade cap (P0 fix: prevent runaway re-entry)
        self.MAX_TRADES_PER_DAY = 8  # raised from 5: enables post-stop re-entry
        self._last_valid_spread = 0.02  # P1: forward-fill for zero-spread bars
        self._last_valid_iv = 0.15      # P1: forward-fill for zero-IV bars (seed)
        self._iv_ever_latched = False   # M1: True once a chain-ready IV is latched;
                                        # until then, keep computed IV over the seed
        self._chain_ready_bar = -1.0    # R2 v4: bar_of_day the chain first populated
                                        # this session; the IV settling window counts
                                        # from HERE, not 09:30 (#2 slow-chain fix)
        # Inter-day rolling buffers — initialized HERE so no code path can hit an
        # AttributeError (both _on_bar's day-boundary roll and _get_interday read
        # them; lazy creation in one missed the other). _daily_atm_iv (T2) feeds
        # IVRVGap5d; _day_open_atm_iv is the session-open ATM_IV captured per day.
        self._daily_closes = deque(maxlen=10)
        self._daily_ivs = deque(maxlen=10)
        self._daily_atm_iv = deque(maxlen=10)
        self._day_open_atm_iv = 0.0
        self._close_1600_px = 0.0       # G11: 16:00 cash close for daily-close roll
        self._pending_entry = False     # 1-bar execution delay (matches proxy)
        self._pending_size = 0.0
        # PARITY: per-bar signal cache. _on_bar evaluates ALL trees every bar
        # (continuous Lag/Delta buffers) and stores the results here; the action
        # gates read these instead of re-evaluating (which would gap the buffers).
        self._sig_entry = False
        self._sig_exit = False
        self._sig_size = 0.0
        self._sig_delta = 0.5
        self._entry_credit = 0.0        # credit received at entry (for stop-loss)
        self._entry_gross = 0.0         # gross |premium| at entry (B-3 debit risk basis)
        self._ruined = False            # B3: account blown up (≤ -100% equity)
        self._stop_loss_base = 2.5      # fallback; overridden per-trade by delta-dependent formula
        self._entry_short_delta = 0.25  # actual short delta, set at entry time
        self._session_open_price = None # for SessionReturn computation
        self._raw_history = {{}}         # for 5-bar smoothed terminals

        # Schedule: evaluate every minute (matches GP proxy's 1-min resolution)
        self.Schedule.On(
            self.DateRules.EveryDay(self.spx),
            self.TimeRules.Every(timedelta(minutes=1)),
            self._on_bar,
        )

    def OnData(self, data):
        # Cache option chain (required: chain unavailable in scheduled functions)
        if data.OptionChains:
            for kvp in data.OptionChains:
                chain = kvp.Value
                if chain:
                    today = self.Time.date()
                    self._cached_chain = [
                        c for c in chain
                        if c.Expiry.date() == today
                    ]

    def _on_bar(self):
        """Main strategy logic — called every minute during market hours."""
        if self.IsWarmingUp:
            return

        # Market hours guard: only run 9:30-15:59 ET (code runs until 16:00; MTC ref is 16:15 SPXW settlement)
        t = self.Time
        if t.hour < 9 or (t.hour == 9 and t.minute < 30) or t.hour >= 16:
            # G11 RV5d parity (2026-06-01): capture the 16:00 cash-session close
            # for the inter-day daily-close roll BEFORE returning. The proxy
            # defines each day's "close" as the 16:00 bar (MinutesToClose==15;
            # evaluator_vectorized.prepare_terminal_data), but this guard freezes
            # _last_spx_price at the 15:59 bar (~1.85pt/day off -> RV5d MAE blew
            # up). Recording the index Price at exactly 16:00 here, and rolling
            # THAT (not _last_spx_price) into _daily_closes at the session
            # boundary, aligns the daily-close DEFINITION with the proxy. Only the
            # 16:00 minute is captured; no trading/eval logic runs past 15:59.
            if t.hour == 16 and t.minute == 0:
                _close_px = self.Securities[self.spx].Price
                if _close_px > 0:
                    self._close_1600_px = _close_px
            return

        # Session boundary reset (0DTE: each day is independent)
        today = self.Time.date()
        if not hasattr(self, "_last_date") or self._last_date != today:
            if hasattr(self, "_last_date") and self._last_date is not None:
                # Force close any overnight position
                if self._position_open:
                    self._close_position("Session boundary close")
                # Reset cross-above/below state at the day boundary (those track
                # bar-to-bar crossings, which must not span overnight). The per-bar
                # Lag/Delta rolling buffers (self._buffers) are cleared just below by
                # the P0-7b parity block — superseding the earlier "keep buffers
                # continuous across days" design: the proxy MASKS within_day_pos<lag
                # to 0 (evaluator_vectorized.py:296), i.e. it resets per day, so the
                # codegen must clear per day to match. bar_count persists across days.
                self._prev_eval = {{}}
                self._curr_eval = {{}}
                self._cached_chain = None
                self._chain_ready_bar = -1.0  # R2 v4: re-arm the chain-ready anchor
                self._trades_today = 0
                self._pending_entry = False
                self._raw_history = {{}}  # reset 5-bar smoothing at day boundary
                # Session-reset the Parkinson RV buffer (0DTE: no overnight carry;
                # the prior-day H/L leak was lookahead-flavored and broke RV parity).
                if hasattr(self, '_rv_log_hl_buf'):
                    self._rv_log_hl_buf.clear()
                # P0-5: capture prior-day real CBOE VIX for VIXChange + VIXMean5d
                # (mirrors gp_condA_iron_condor.py:135). _curr_day_vix still holds
                # yesterday's value here; today's is set after _last_date below.
                self._prev_day_vix = getattr(self, '_curr_day_vix', 0.0)
                # Track daily closes / real VIX for inter-day terminals.
                if not hasattr(self, '_daily_closes'):
                    self._daily_closes = deque(maxlen=10)
                    self._daily_ivs = deque(maxlen=10)
                    self._daily_atm_iv = deque(maxlen=10)
                # G11 RV5d parity: roll the 16:00 cash-session close (captured in
                # the market-hours guard at MTC==15) to MATCH the proxy's daily-
                # close definition. Fall back to _last_spx_price (the 15:59 bar)
                # only if the 16:00 print was missed (first day / short session).
                _day_close_px = getattr(self, '_close_1600_px', 0.0)
                if _day_close_px <= 0:
                    _day_close_px = getattr(self, '_last_spx_price', 0.0)
                if _day_close_px > 0:
                    self._daily_closes.append(_day_close_px)
                self._close_1600_px = 0.0  # re-arm for the new day
                if getattr(self, '_curr_day_vix', 0.0) > 0:
                    self._daily_ivs.append(self._curr_day_vix)  # real VIX -> VIXMean5d
                # T2 fix (2026-05-31): IVRVGap5d uses ATM_IV, NOT VIX. The proxy's
                # IVRVGap5d = mean(ATM_IV[d-5:d]) - RV5d (evaluator_vectorized.py),
                # but codegen used VIXMean5d/100 (~0.18 vs ATM_IV ~0.136) -> +1.38σ
                # offset (VIX carries the variance-risk premium). Roll the prior
                # day's session-open ATM_IV in here (mirrors the VIX/close rolls).
                if getattr(self, '_day_open_atm_iv', 0.0) > 0:
                    self._daily_atm_iv.append(self._day_open_atm_iv)
                self._day_open_atm_iv = 0.0  # captured at first valid-IV bar of the new day
                # OvernightGap / SessionReturn / SessionPosition anchor: the true
                # session-open price is captured from the first fresh index Price in
                # _compute_scalars (Securities.Open is stale at day-boundary
                # detection for an index). Here we only stash yesterday's close
                # (= _last_spx_price, not yet updated to today) and re-arm capture.
                self._prev_session_close = getattr(self, '_last_spx_price', None)
                self._session_open_price = None   # captured at the first bar below
                # Reset session high/low for SessionPosition (re-seeded at first bar)
                self._session_high = None
                self._session_low = None
            self._last_date = today
            self._day_count += 1  # P0-5b: distinct trading days, for the VIX-dead gate
            # P0-7b PARITY FIX (2026-05-31): clear the per-bar Lag/Delta buffers at
            # each new trading day so cross-day Lag/Delta references return 0,
            # MATCHING the proxy (evaluator_vectorized.py:286 masks within_day_pos
            # < lag -> 0). Without this the deques carried ACROSS days, so Lag
            # reached into the PRIOR day's value -> the bcc_f2 entry
            # GT(VIXChange, Lag^3(VIXChange)) fired ~13x more often in QC than in
            # the proxy (8 trades/day vs ~1; QC Sharpe -4.0 vs proxy +1.5). Runs
            # BEFORE this bar's _compute_scalars/appends (order verified). Only
            # self._buffers (per-bar Lag/Delta); the daily deques
            # (_daily_closes/_daily_ivs) span days by design and are untouched.
            for _lag_buf in self._buffers.values():
                _lag_buf.clear()
            # P0-5: current real CBOE VIX, refreshed once per day at the session
            # boundary (the CBOE series is daily -> constant within a day); consumed
            # by VIXSpot/VIXTermSlope/VIXChange in _compute_scalars (mirrors
            # gp_condA_iron_condor.py:151). Forward-fills the last good value via the
            # else branch, so once populated it never reverts to 0.
            self._curr_day_vix = (self.Securities[self.vix].Price
                                  if (self.vix in self.Securities and self.Securities[self.vix].Price > 0)
                                  else getattr(self, '_curr_day_vix', 0.0))
            # P0-5b: latch source-health here too (warmup-independent), so the
            # OnEndOfAlgorithm backstop is honest for ANY run length -- not only
            # after WARMUP_BARS, where the in-loop latch lives.
            if self._curr_day_vix > 0:
                self._vix_ever_populated = True
            # (SessionReturn's open anchor is captured from the first fresh index
            # Price in _compute_scalars — see the session-open capture there.)

        # End-of-day: no new entries after 15:45, force close at 15:50
        _eod_no_entry = (self.Time.hour == 15 and self.Time.minute >= 45)
        if ((self.Time.hour == 15 and self.Time.minute >= 50) or self.Time.hour >= 16) and self._position_open:
            self._close_position("EOD force close")
            return

        # Compute bypass scalars (base values, before delta computation)
        s = self._compute_scalars()
        if s is None:
            return
        # Track SPX price for OvernightGap computation
        spx_px = self.Securities[self.spx].Price
        if spx_px > 0:
            self._last_spx_price = spx_px

        # Track raw RawSpread for delta computation (avoids double-normalization)
        if "_raw_RawSpread" not in self._buffers:
            self._buffers["_raw_RawSpread"] = deque(maxlen=self.MAX_LAG + 1)
        self._buffers["_raw_RawSpread"].append(s.get("_raw_RawSpread", 0.0))

        # Update rolling buffers with normalized values
        for name, val in s.items():
            if name.startswith("_raw_") or name in ("DeltaSpread1", "DeltaSpread5"):
                continue  # skip internal raw values and deltas (computed below)
            if name not in self._buffers:
                self._buffers[name] = deque(maxlen=self.MAX_LAG + 1)
            self._buffers[name].append(val)
        self._bar_count += 1

        # Compute DeltaSpread from RAW (un-normalized) RawSpread, then normalize ONCE
        _raw_buf = self._buffers["_raw_RawSpread"]
        _raw_now = _raw_buf[-1] if _raw_buf else 0.0
        _raw_1 = _raw_buf[-2] if len(_raw_buf) >= 2 else _raw_now
        _raw_5 = _raw_buf[-6] if len(_raw_buf) >= 6 else _raw_now
        s["DeltaSpread1"] = self._normalize("DeltaSpread1", _raw_now - _raw_1) if self._bar_count > 1 else 0.0
        s["DeltaSpread5"] = self._normalize("DeltaSpread5", _raw_now - _raw_5) if self._bar_count > 5 else 0.0
        # Push normalized deltas into buffer so Lag(DeltaSpread1, k) works
        for _ds_name in ("DeltaSpread1", "DeltaSpread5"):
            if _ds_name not in self._buffers:
                self._buffers[_ds_name] = deque(maxlen=self.MAX_LAG + 1)
            self._buffers[_ds_name].append(s[_ds_name])

        if self._bar_count < self.WARMUP_BARS:
            return

        # P0-5b: CBOE VIX source-health check (post-warmup). The live security
        # price is the earliest signal that the subscription is alive; latch it.
        if self.vix in self.Securities and self.Securities[self.vix].Price > 0:
            self._vix_ever_populated = True
        elif (not self._vix_source_failed) and self._day_count >= 3:
            # >=3 distinct trading days elapsed and real CBOE VIX has NEVER been
            # seen -> the AddData(CBOE,"VIX") subscription is dead. The VIX
            # terminals are degraded (VIXSpot -> ATM_IV*100 proxy; the others ->
            # frozen constants), re-injecting the P0-5 skew. Fail loud, once, and
            # mark the run machine-detectably invalid.
            self._vix_source_failed = True
            self.Error(
                "P0-5b VIX SOURCE FAILURE: CBOE VIX never populated after "
                f"{{self._day_count}} days / {{self._bar_count}} bars. VIX terminals "
                "are degraded -- VIXSpot uses the ATM_IV*100 proxy; VIXTermSlope/"
                "VIXChange/VIXMean5d fall back to frozen constants -- re-injecting "
                "the training-serving skew P0-5 removed. This backtest is INVALID "
                "for proxy->QC calibration."
            )
            self.SetRuntimeStatistic("vix_source_failed", "1")

        # Diagnostic: log scalars and entry eval periodically
        # More frequent logging early to diagnose "not trading" issues
        _log_freq = 30 if self._bar_count < 500 else 100
        if self._bar_count == self.WARMUP_BARS or self._bar_count % _log_freq == 0:
            self.Debug(f"Bar {{self._bar_count}} chain={{len(self._cached_chain) if self._cached_chain else 0}} "
                       f"ATM_IV={{s['ATM_IV']:.4f}} "
                       f"VIX={{s['VIXSpot']:.2f}} VIXChg={{s.get('VIXChange',0):.3f}} "
                       f"RV={{s['RealizedVol30m']:.4f}} Spread={{s['RawSpread']:.4f}} "
                       f"MTC={{s['MinutesToClose']:.0f}} BoD={{s['BarOfDay']:.1f}} "
                       f"OGap={{s.get('OvernightGap',0):.4f}} SRet={{s.get('SessionReturn',0):.4f}} "
                       f"chain={{len(self._cached_chain) if self._cached_chain else 0}} "
                       f"pos={{self._position_open}} trades={{self._trades_today}}")

        # Evaluate strategy trees (with min/max hold period matching proxy)
        MIN_BARS_IN_TRADE = 15  # proxy: min_bars_in_trade=15 (prevents spread-crossing churning)
        # MUST equal the proxy's max_bars_in_trade default (evaluator_vectorized.py:922).
        # A shorter QC cap than the proxy used during evolution realizes different
        # theta/PnL on the SAME strategy = training-serving skew (Sculley et al., 2015);
        # the proxy held to bar 330, so QC must too. Single source of truth, asserted
        # by tests/test_codegen_exec_parity.py. (Was 180 with a false "proxy=180" comment.)
        MAX_BARS_IN_TRADE = 330  # proxy: max_bars_in_trade=330 (evaluator_vectorized.py:922)
        try:
            # PARITY FIX (2026-05-31): evaluate ALL strategy trees EVERY bar so
            # their nested Lag/Delta expr-buffers (_lag_expr/_delta_expr) and the
            # Cross prev-bar state stay CONTINUOUS — matching the proxy, which
            # evaluates entry/exit/size/delta signals over ALL bars
            # (evaluator_vectorized.py:1017-1051). Evaluating a tree only while its
            # branch was active (entry-when-flat / exit-when-in-position) starved the
            # OTHER branch's buffers, so Lag^k reached across the gap to a STALE
            # value: the bcc_f2 entry GT(VIXChange, Lag^3(VIXChange)) re-fired
            # ~6.4x/active-day in QC vs the proxy's ~1.5 (proven offline,
            # layer3/diagnostics/operator_parity.py: 51 vs 12 entries -> identical
            # after this fix). The ACTION stays gated by position state below; only
            # the EVALUATION is now unconditional.
            #
            # UnrealizedProfitPct (position-aware exit terminal) — PARITY with the
            # proxy (evaluator_vectorized.py): the fraction of MAX PROFIT captured =
            # (unrealised P&L) / |entry premium|. The proxy computes
            # (curr_val - entry_net_value)/|entry_net_value| per share; in dollars
            # that is exactly unrealised_pnl / |entry_credit| (the n_contracts and
            # the x100 multiplier cancel). self._entry_credit is the cash delta at
            # entry: > 0 for a CREDIT spread (received), < 0 for a DEBIT (paid), so
            # abs() is the premium magnitude for both. 0.0 when FLAT (entry-tree /
            # idle bars) — benign no-op, matching the proxy's flat baseline. RAW
            # (no _normalize), so GT(UnrealizedProfitPct, 0.5) fires at >=50% of max
            # profit, identical to the proxy.
            if self._position_open and abs(getattr(self, '_entry_credit', 0.0)) > 1e-6:
                _upp_pnl = sum(
                    h.UnrealizedProfit for h in self.Portfolio.Values
                    if h.Invested and h.Symbol != self.spx
                )
                s["UnrealizedProfitPct"] = _upp_pnl / abs(self._entry_credit)
                # Per-trade UPP history for Lag/Delta(UnrealizedProfitPct) — append
                # this in-position bar (most-recent last), BEFORE the trees are
                # evaluated below, so self._lag("UnrealizedProfitPct", k) sees this
                # bar as upp_hist[-1]. Mirrors the proxy's upp_hist.append at
                # evaluator_vectorized.py:1775 (also pre-exit-eval). The entry bar
                # itself appends nothing (position not yet open here), matching the
                # proxy where the append starts the bar AFTER entry.
                self._upp_hist.append(s["UnrealizedProfitPct"])
            else:
                s["UnrealizedProfitPct"] = 0.0
            self._sig_entry = bool(self._eval_entry(s))
            self._sig_exit = bool(self._eval_exit(s))
            self._sig_size = max(0.0, min(1.0, self._eval_size(s)))
            self._sig_delta = self._eval_delta(s)
{diag_terminal_block}
            # B3 (campaign sweep): detect ruin — a blown-up cash account (total value
            # at or below 0 = the proxy's equity ≤ -100%). Sticky for the rest of the
            # backtest window, mirroring the proxy's _ruined halt (which takes no
            # further entries once cumulative equity crosses -100%).
            if self.Portfolio.TotalPortfolioValue <= 0.0:
                self._ruined = True
            if not self._position_open:
                # B3: ruined account takes no further entries this window.
                if self._ruined:
                    self._pending_entry = False
                # Daily trade cap (P0: prevent runaway re-entry)
                elif self._trades_today >= self.MAX_TRADES_PER_DAY:
                    self._pending_entry = False
                # No new entries near EOD (would just get force-closed)
                elif _eod_no_entry:
                    self._pending_entry = False
                # Cooldown after failed open attempt
                elif self._open_fail_cooldown > 0:
                    self._open_fail_cooldown -= 1
                    self._pending_entry = False  # cancel stale signal during cooldown
                # 1-bar execution delay: signal at bar N, fill at bar N+1
                # (matches proxy's pending_entry mechanism — prevents
                # look-ahead bias from entering at the price that triggered)
                elif self._pending_entry:
                    self._pending_entry = False
                    self.Debug(f"DELAYED ENTRY FILL at bar {{self._bar_count}}")
                    self._open_position(s)
                else:
                    # Regime guard: skip entry in low-vol regimes (Risk Manager recommendation)
                    _regime_ok = {regime_guard_iv_val} <= 0 or self._last_valid_iv >= {regime_guard_iv_val}
                    if not _regime_ok:
                        if self._bar_count % 200 == 0:
                            self.Debug(f"Regime guard: ATM_IV={{self._last_valid_iv:.4f}} < {regime_guard_iv_val} -- skipping entry")
                    elif True:
                        # Signals already computed up-front this bar (continuous
                        # buffers); the action gate just reads the stored values.
                        entry_signal = self._sig_entry
                        if self._bar_count % 100 == 0:
                            self.Debug(f"Entry eval bar {{self._bar_count}}: signal={{entry_signal}}")
                        if entry_signal:
                            self._pending_entry = True
                            self._pending_size = self._sig_size
                            self.Debug(f"ENTRY SIGNAL at bar {{self._bar_count}} (fill next bar)")
            else:
                self._bars_in_trade += 1
                # Stop-loss / max-loss gate (must match proxy evaluator logic).
                # Credits: exit when loss > N× credit received.
                # Debits: exit when loss > 80% of debit paid, or underwater after 40 bars.
                # Stop-loss slippage: proxy applies 1.2-1.5× time-dependent slippage on
                # stop exits. In QC, market fills naturally include slippage, but to match
                # the proxy's conservative trigger we tighten the threshold by dividing by
                # the expected slippage factor (exits earlier to account for worse fills).
                # Stop-loss base — emitted by codegen to match the proxy's
                # V1-vs-Level-B branch (evaluator_vectorized.py:1197-1200):
                #   V1 / scalar-only      -> stop_loss_credit_multiple * 0.85
                #   Level B (delta_tree)  -> (1.5 + short_delta * 2.0)   * 0.85
                # (P0-3: previously always used the Level-B formula with a default
                #  0.25 delta -> 1.70 for V1, vs the proxy's 2.125 = training-serving skew.)
                # 0.85x execution discount: QC v5 data shows actual stop-loss
                # fills average 0.3x credit WORSE than trigger level due to
                # 1-minute execution lag + bid-ask slippage on exit. Tightening
                # trigger by 0.85x keeps actual losses under the intended cap.
                # 0.85x calibrated from 37 QC v5 losing trades (P90 loss ratio
                # 2.86x at a 2.5x trigger -> ~0.36x slippage). E.g. the V1 base
                # 2.5 x 0.85 = 2.125x trigger; actual fills land near the cap.
                _stop_loss_multiple = {stop_base_expr}
                if _stop_loss_multiple > 0:
                    # Use per-position unrealized PnL from invested securities,
                    # not Portfolio.TotalUnrealisedProfit which aggregates across
                    # all holdings including cash. Matches proxy's per-trade M2M.
                    # (Codegen divergence #2 fix, 2026-05-19)
                    unrealised_pnl = sum(
                        h.UnrealizedProfit for h in self.Portfolio.Values
                        if h.Invested and h.Symbol != self.spx
                    )
                    # Time-dependent stop: matches proxy formula exactly.
                    # Wider early (recovery possible), tighter late (gamma concentrated).
                    # mult = base × (0.75 + 0.50 × mtc/400)
                    _market_close = self.Time.replace(hour=16, minute=15, second=0)
                    _mtc_now = max((_market_close - self.Time).total_seconds() / 60.0, 0.0)
                    _time_factor = 0.75 + 0.50 * min(_mtc_now / 400.0, 1.0)
                    if {is_credit} and self._entry_credit > 0.01:
                        # Credit: exit when loss > time-adjusted × credit received
                        _adjusted_mult = _stop_loss_multiple * _time_factor
                        if unrealised_pnl < -_adjusted_mult * self._entry_credit:
                            self._close_position(f"STOP LOSS at bar {{self._bar_count}} (loss={{unrealised_pnl:.0f}} > {{_stop_loss_multiple:.1f}}x credit={{self._entry_credit:.0f}})")
                            return
                    elif not {is_credit} and self._entry_credit < -0.01:
                        # Debit max-loss = 80% of the RISK BASIS = max(entry_gross, net
                        # debit) (B-3: matches the proxy; for a ratio backspread gross
                        # >> net so a net-only basis stops ~9x too early).
                        debit_paid = abs(self._entry_credit)
                        _risk_basis = max(self._entry_gross, debit_paid)
                        if unrealised_pnl < -0.80 * _risk_basis:
                            self._close_position(f"DEBIT MAX LOSS at bar {{self._bar_count}} (loss={{unrealised_pnl:.0f}} > 80% of risk_basis={{_risk_basis:.0f}})")
                            return
                        # Time-decay gate: still underwater after N bars (B-2: 240 for
                        # ratio/3-leg, 40 otherwise — matches the proxy, was hardcoded 40)
                        if self._bars_in_trade >= {debit_underwater_bars} and unrealised_pnl < 0:
                            self._close_position(f"DEBIT TIME DECAY at bar {{self._bar_count}} (underwater after {{self._bars_in_trade}} bars)")
                            return
                # Force close after max holding period
                if self._bars_in_trade >= MAX_BARS_IN_TRADE:
                    self._close_position(f"MAX HOLD at bar {{self._bar_count}} ({{self._bars_in_trade}} bars)")
                # Only evaluate exit after min holding period
                elif self._bars_in_trade >= MIN_BARS_IN_TRADE:
                    if self._sig_exit:
                        self._close_position(f"EXIT at bar {{self._bar_count}} (held {{self._bars_in_trade}} bars)")
        except Exception as e:
            import traceback
            self._tree_eval_errors = getattr(self, '_tree_eval_errors', 0) + 1
            if self._tree_eval_errors <= 5:
                self.Debug(f"Tree eval error ({{self._tree_eval_errors}}): {{e}}\\n{{traceback.format_exc()[:500]}}")
            elif self._tree_eval_errors == 6:
                self.Debug("Suppressing further tree eval error details (>5 errors)")

        # Rotate cross-detection cache
        self._prev_eval = dict(self._curr_eval)
        self._curr_eval = {{}}

    # L2 grammar fix: normalization stats (center, scale) per terminal.
    # SINGLE SOURCE OF TRUTH: must match layer2/terminal_stats.py exactly.
    _NORM_STATS = {norm_stats_literal}

    def _normalize(self, name, raw):
        """Normalize a terminal value to ~N(0,1) using frozen training stats.
        OOD clamp [-5, +5] matches proxy evaluator to prevent extreme outliers
        from producing divergent tree outputs during market dislocations."""
        stats = self._NORM_STATS.get(name)
        if stats is None:
            return raw
        center, scale = stats
        if scale < 1e-12:
            return 0.0
        return max(-5.0, min(5.0, (raw - center) / scale))

    def _compute_scalars(self):
        """Compute bypass scalar values from QC market data.

        L2 grammar fix: all REAL terminals are normalized to ~N(0,1) using
        training-split statistics. This matches the normalization applied in
        the GP proxy evaluator (layer2/evaluator.py).
        """
        spx_price = self.Securities[self.spx].Price
        if spx_price <= 0:
            if self._bar_count % 100 == 0:
                self.Debug(f"Bar {{self._bar_count}}: SPX price=0, skipping")
            return None

        # Session-open capture (2026-06-01 parity fix). Securities.Open is STALE at
        # day-boundary detection for an index, so log(Open/prev_close) collapsed to
        # ~0 -> OvernightGap stuck ~-0.08 and SessionReturn/SessionPosition
        # mis-anchored. Capture the TRUE session open from the first fresh index
        # Price of the day (matches the proxy's bar-0 reconstructed open), and use
        # it for OvernightGap, SessionReturn, and SessionPosition together.
        if self._session_open_price is None and spx_price > 0:
            self._session_open_price = spx_price
            _pc = getattr(self, '_prev_session_close', None)
            self._overnight_gap = math.log(spx_price / _pc) if (_pc and _pc > 0) else 0.0
            if getattr(self, '_session_high', None) is None:
                self._session_high = spx_price
                self._session_low = spx_price

        # P0-5: VIXSpot from real CBOE VIX (self._curr_day_vix, refreshed each bar
        # in _on_bar). Fallback to the ATM_IV*100 proxy only during VIX warmup so
        # the terminal is never zero. Finalized after ATM_IV is computed below.
        vix_spot = getattr(self, '_curr_day_vix', 0.0)

        # Time — C2 fix: use 16:15 ET (SPXW PM settlement) to match proxy.
        # Was 16:00 — 15-min systematic offset caused every bar's MTC to
        # differ between proxy and QC, shifting threshold-dependent strategies.
        market_close = self.Time.replace(hour=16, minute=15, second=0)
        minutes_to_close = max(0, min(404, (market_close - self.Time).total_seconds() / 60))
        market_open = self.Time.replace(hour=9, minute=30, second=0)
        # BarOfDay = minute index within session (0-404 at 1-min resolution)
        bar_of_day = max(0, min(404, (self.Time - market_open).total_seconds() / 60.0))

        # ATM IV, spread, and skew from the option chain via cubic spline.
        # Matches the proxy's research_collector.py _fill_side(): fits a natural
        # cubic spline on (delta, IV) and (delta, spread) across the chain and
        # evaluates at target deltas (0.50 ATM, 0.25 skew). Both the IV and the
        # delta fed into the spline are SELF-COMPUTED Black-Scholes values (see
        # loop below), identical to how the collector built the L1 grid.
        atm_iv = 0.0
        raw_spread = 0.0
        _put_atm_iv = 0.0
        _call_atm_iv = 0.0
        _put_25d_iv = 0.0
        _call_25d_iv = 0.0
        _puts = []
        _calls = []
        _put_atm_spread = 0.0
        _call_atm_spread = 0.0
        if self._cached_chain:
            # P0-4: self-compute Black-Scholes IV + delta from the bid/ask mid,
            # matching the L1 collector (research_collector._bs_iv/_bs_greeks,
            # r=0.05, expiry 16:15 ET) instead of QC-native _c.ImpliedVolatility /
            # _c.Greeks.Delta. QC-native IV uses different conventions (~1.9x higher
            # on 0DTE) than the IV the GP was evolved on, which forced the old
            # 0.3869x+0.0292 patch. Using the SAME IV definition at train (L1/proxy)
            # and serve (QC) removes that training-serving skew at the source
            # (Sculley et al., 2015). mid>0.05 gate matches research_collector.py:958.
            _puts = []
            _calls = []
            _mins_left = max((16 * 60 + 15) - (self.Time.hour * 60 + self.Time.minute), 1.0)
            for _c in self._cached_chain:
                _bid = _c.BidPrice if _c.BidPrice and _c.BidPrice > 0 else 0.0
                _ask = _c.AskPrice if _c.AskPrice and _c.AskPrice > 0 else 0.0
                if _bid <= 0 or _ask <= 0:
                    continue
                _mid = (_bid + _ask) / 2.0
                if _mid <= 0.05:
                    continue
                _dte = max((_c.Expiry.date() - self.Time.date()).days, 0)
                _T = max(1e-6, (_dte + _mins_left / 390.0) / 365.0)
                _is_call_c = (_c.Right == OptionRight.Call)
                _iv = self._bs_iv(_mid, spx_price, _c.Strike, _T, 0.05, _is_call_c)
                if _iv is None or _iv <= 0.001:
                    continue
                _d = self._bs_delta(spx_price, _c.Strike, _T, 0.05, _iv, _is_call_c)
                if _d is None or abs(_d) <= 0.001:
                    continue
                _ba = (_ask - _bid) / _mid
                if _is_call_c:
                    _calls.append((_d, _iv, _ba))
                else:
                    _puts.append((abs(_d), _iv, _ba))
            # Sort by delta and remove duplicates (matching proxy)
            _puts.sort(key=lambda x: x[0])
            _calls.sort(key=lambda x: x[0])
            _puts = self._dedup_by_delta(_puts)
            _calls = self._dedup_by_delta(_calls)
            # Spline interpolation per side at target deltas: 0.50 (ATM), 0.25 (skew)
            if len(_puts) >= 4:
                _put_atm_iv = self._cubic_spline_eval([p[0] for p in _puts], [p[1] for p in _puts], 0.50)
                _put_25d_iv = self._cubic_spline_eval([p[0] for p in _puts], [p[1] for p in _puts], 0.25)
                _put_atm_spread = self._cubic_spline_eval([p[0] for p in _puts], [p[2] for p in _puts], 0.50)
            elif len(_puts) >= 2:
                _put_atm_iv = self._linear_interp([p[0] for p in _puts], [p[1] for p in _puts], 0.50)
                _put_25d_iv = self._linear_interp([p[0] for p in _puts], [p[1] for p in _puts], 0.25)
                _put_atm_spread = self._linear_interp([p[0] for p in _puts], [p[2] for p in _puts], 0.50)
            else:
                _put_atm_spread = 0.0
            if len(_calls) >= 4:
                _call_atm_iv = self._cubic_spline_eval([c[0] for c in _calls], [c[1] for c in _calls], 0.50)
                _call_25d_iv = self._cubic_spline_eval([c[0] for c in _calls], [c[1] for c in _calls], 0.25)
                _call_atm_spread = self._cubic_spline_eval([c[0] for c in _calls], [c[2] for c in _calls], 0.50)
            elif len(_calls) >= 2:
                _call_atm_iv = self._linear_interp([c[0] for c in _calls], [c[1] for c in _calls], 0.50)
                _call_25d_iv = self._linear_interp([c[0] for c in _calls], [c[1] for c in _calls], 0.25)
                _call_atm_spread = self._linear_interp([c[0] for c in _calls], [c[2] for c in _calls], 0.50)
            else:
                _call_atm_spread = 0.0
            # ATM IV = average of put-side and call-side spline at delta=0.50
            if _put_atm_iv > 0.001 and _call_atm_iv > 0.001:
                atm_iv = (_put_atm_iv + _call_atm_iv) / 2.0
            elif _put_atm_iv > 0.001:
                atm_iv = _put_atm_iv
            elif _call_atm_iv > 0.001:
                atm_iv = _call_atm_iv
            # Fallback: spline failed (too few valid contracts). Self-compute BS IV
            # on the nearest-to-ATM call mid (same model as the grid above; never
            # QC-native ImpliedVolatility). If still unavailable, use the prior
            # derived VIX (~last valid), then the P1 forward-fill below.
            if atm_iv <= 0.001:
                _atm = min(
                    (c for c in self._cached_chain
                     if c.Right == OptionRight.Call and c.BidPrice and c.AskPrice
                     and c.BidPrice > 0 and c.AskPrice > 0),
                    key=lambda c: abs(c.Strike - spx_price),
                    default=None,
                )
                if _atm is not None:
                    _amid = (_atm.BidPrice + _atm.AskPrice) / 2.0
                    _adte = max((_atm.Expiry.date() - self.Time.date()).days, 0)
                    _aT = max(1e-6, (_adte + _mins_left / 390.0) / 365.0)
                    _aiv = self._bs_iv(_amid, spx_price, _atm.Strike, _aT, 0.05, True) if _amid > 0.05 else None
                    if _aiv is not None and _aiv > 0.001:
                        atm_iv = _aiv
                if atm_iv <= 0.001 and self._derived_vix > 0:
                    atm_iv = self._derived_vix / 100.0
            # Raw spread = average of put-side and call-side spline at delta=0.50
            if _put_atm_spread > 0.0001 and _call_atm_spread > 0.0001:
                raw_spread = (_put_atm_spread + _call_atm_spread) / 2.0
            elif _put_atm_spread > 0.0001:
                raw_spread = _put_atm_spread
            elif _call_atm_spread > 0.0001:
                raw_spread = _call_atm_spread
            else:
                # Fallback: nearest-to-ATM contract spread
                _atm_any = min(
                    self._cached_chain,
                    key=lambda c: abs(c.Strike - spx_price),
                    default=None,
                )
                if _atm_any and _atm_any.AskPrice and _atm_any.AskPrice > 0 and _atm_any.BidPrice and _atm_any.BidPrice > 0:
                    _mid = (_atm_any.AskPrice + _atm_any.BidPrice) / 2.0
                    raw_spread = (_atm_any.AskPrice - _atm_any.BidPrice) / _mid if _mid > 0 else 0.0
            # Clamp spline output to valid range (extrapolation can produce negatives)
            atm_iv = max(atm_iv, 0.0)
            raw_spread = max(raw_spread, 0.0)
        # P0-4: the 0.3869*atm_iv+0.0292 ATM_IV correction is REMOVED. It was an
        # empirical single-month (Jan 2025) linear patch that bridged QC-native IV
        # to the collector's IV and did not generalize across regimes. Now that the
        # spline is fed self-computed Black-Scholes IV with the SAME inversion as
        # the collector (see grid loop above), the two IV definitions agree by
        # construction and no fitted correction is needed — train/serve skew is
        # eliminated at the source rather than patched (Sculley et al., 2015).

        # P1 + R2-open: forward-fill ATM_IV from the last RELIABLE value through the
        # SESSION-OPEN SETTLING window. The bcc_f2 QC-diag log showed ATM_IV_5m
        # ramping for ~6 bars after 09:30 (e.g. -1.05 -> 0.62 on Jan 3), whereas the
        # collector that built the proxy parquet saw a stable IV from bar 0. The
        # cause is TIME, not chain count: the chain fills 0 -> ~200 within ONE
        # minute (the QC-diag `Bar ... chain=0` line appears only at 09:30; chain
        # ~full by 09:31), so a chain-count gate only ever caught the masked first
        # bar and was a no-op -- two deploys at thresholds 50 and 150 produced byte-
        # identical stats. The ramp is the 5-bar ATM_IV_5m smoothing window
        # dragging the unsettled opening-print IV (bars 0-1, wide opening spreads)
        # across the average. So gate on TIME: hold the IV unreliable for the
        # first _SESSION_IV_WARMUP bars after the chain populates (v4 anchor below)
        # and forward-fill from the last reliable value -- matching the proxy's
        # stable open (note: the forward-fill is FLAT, so it kills the proxy's
        # small real morning Delta too -> QC under-sizes vs proxy on early entries
        # whose size reads morning micro-IV; that residual is non-transferable
        # because QC's opening mids cannot resolve ~0.002 IV changes) and killing the
        # QC-only Delta(ATM_IV_5m,1) spike that oversized the 09:35 entries (the
        # 37x/19x/21x n outliers). 6 == the 5m window length + 1, so every unsettled
        # print ages out before the IV is trusted. The `len(chain) >= 50` clause is
        # retained ONLY as a degenerate near-empty-chain guard (NOT the primary
        # gate, which is the time window -- by bar 6 the chain is always full).
        # (R2 fix v3, 2026-06-01: chain-gate deploys were byte-identical; the 09:35
        # size bar has chain ~full but IV still settling -> needs a time gate.
        # _SESSION_IV_WARMUP is a named constant for an easy diagnostic re-tune.)
        _SESSION_IV_WARMUP = 6
        # R2 v4 (#2 slow-chain fix): count the settling window from when the CHAIN
        # first populated this session, NOT from 09:30. On slow-chain days (e.g.
        # 2025-02-28: chain=0 until ~09:40) the opening-print IV keeps settling for
        # ~6 bars AFTER the chain arrives, so a 09:30-anchored window releases too
        # early and a late (09:40) entry reads the ramp -> the n=14 outlier. Anchor
        # the window on chain-ready so every day gets the full settling allowance.
        _chain_ok = bool(self._cached_chain) and len(self._cached_chain) >= 50
        if _chain_ok and self._chain_ready_bar < 0.0:
            # int() so the window is exact integer-bar arithmetic: bar_of_day is a
            # float (total_seconds()/60) that can land on 5.9999 and silently delay
            # release by a bar against the `>= 6` test.
            self._chain_ready_bar = int(bar_of_day)
        _iv_reliable = (_chain_ok and self._chain_ready_bar >= 0.0
                        and (int(bar_of_day) - self._chain_ready_bar) >= _SESSION_IV_WARMUP)
        if atm_iv > 0.001 and _iv_reliable:
            self._last_valid_iv = atm_iv
            self._iv_ever_latched = True
            # T2: capture the session-OPEN ATM_IV (first RELIABLE-IV bar of day)
            # for IVRVGap5d's prior-days mean (proxy uses _day_iv at session open).
            if getattr(self, '_day_open_atm_iv', 0.0) <= 0.0:
                self._day_open_atm_iv = atm_iv
        elif getattr(self, '_iv_ever_latched', False) or atm_iv <= 0.001:
            # Forward-fill from the last RELIABLE value: (a) a prior reliable IV
            # exists -> use it (suppresses the forming-chain open ramp); or (b)
            # there is no live value at all -> the seed/last value. M1 fix: do NOT
            # overwrite a usable partial-chain IV with the 0.15 seed before ANY
            # reliable value has ever been latched -- on the first-ever morning we
            # keep the computed atm_iv (better than the seed) rather than leaking
            # 0.15 into ATM_IV / IVRVGap5d / strike sizing. Only pre-first-latch
            # warmup bars take the implicit `else` (keep computed atm_iv).
            atm_iv = self._last_valid_iv
        if raw_spread > 0.0001:
            self._last_valid_spread = raw_spread
        else:
            raw_spread = self._last_valid_spread

        # Keep the internal IV-scale VIX proxy up to date (used ONLY for the
        # spline-fail ATM_IV fallback and BS strike sizing -- never a terminal).
        if atm_iv > 0.001:
            self._derived_vix = atm_iv * 100.0
        # P0-5: VIXSpot terminal = real CBOE VIX; fall back to the ATM_IV*100
        # proxy only while the CBOE VIX series is warming up (value still 0).
        # P0-5b: count every fall-back bar so OnEndOfAlgorithm can report whether
        # the proxy was used briefly (warmup, benign) or for the whole run (dead
        # subscription -> invalid). _vix_ever_populated disambiguates the two.
        if vix_spot <= 0.0:
            vix_spot = self._derived_vix
            self._vix_fallback_bars += 1

        # RealizedVol30m — Parkinson estimator over last 30 bars
        rv30m = self._compute_rv30m()

        # P0-5: VIXTermSlope = VIX9D - VIX from the real CBOE indices (collector
        # raw variate v120 - v119; research_collector.compute_vix_features). CBOE
        # VIX9D IS available in QC backtest (gp_condA_iron_condor.py:53,499-501) --
        # the prior "unavailable, freeze to constant" comment was false and made
        # this a dead terminal in QC while the GP evolved against a live +/-2sigma
        # signal. Fall back to the frozen median (-> normalized 0) only during warmup.
        _vix9d = (self.Securities[self.vix9d].Price
                  if (self.vix9d in self.Securities and self.Securities[self.vix9d].Price > 0)
                  else 0.0)
        if _vix9d > 0.0 and vix_spot > 0.0:
            vix_term_slope = _vix9d - vix_spot
        else:
            vix_term_slope = -0.6875  # frozen median (normalizes to 0) during warmup

        # SessionReturn: cumulative intraday SPX return since session open
        session_return = 0.0
        if self._session_open_price and self._session_open_price > 0:
            session_return = (spx_price - self._session_open_price) / self._session_open_price

        # 5-bar smoothed terminals: trailing mean of raw values with day-boundary respect
        for _base, _val in [("ATM_IV", atm_iv), ("RealizedVol30m", rv30m), ("RawSpread", raw_spread)]:
            _key = f"_raw_5m_{{_base}}"
            if _key not in self._raw_history:
                self._raw_history[_key] = deque(maxlen=5)
            self._raw_history[_key].append(_val)
        atm_iv_5m = sum(self._raw_history.get("_raw_5m_ATM_IV", [atm_iv])) / max(len(self._raw_history.get("_raw_5m_ATM_IV", [1])), 1)
        rv30m_5m = sum(self._raw_history.get("_raw_5m_RealizedVol30m", [rv30m])) / max(len(self._raw_history.get("_raw_5m_RealizedVol30m", [1])), 1)
        raw_spread_5m = sum(self._raw_history.get("_raw_5m_RawSpread", [raw_spread])) / max(len(self._raw_history.get("_raw_5m_RawSpread", [1])), 1)

        # PutCallSkew: IV(25-delta put) - IV(25-delta call)
        # Uses spline-interpolated IV at delta=0.25 from both sides, computed
        # above alongside ATM_IV. Matches proxy's _fill_side() which evaluates
        # the same spline at target_delta=0.25 for the 25Dp and 25Dc grid points.
        # Old approach used _find_contract for single nearest 25d contract.
        #
        # G11 reliability gate (2026-06-01): the collector (research_collector.py
        # _reliability) zeroes any grid point whose NEAREST observed delta is
        # >0.08 from the target, and build_variate_vector then sets that grid IV
        # to 0.0 -> the parquet (generate_minute_parquet.py) records
        # PutCallSkew=0.0 whenever either 25Δ leg is unreliable. QC's spline
        # EXTRAPOLATES the steep 25Δ put wing even with no live contract near
        # 0.25Δ, so QC emitted a +0.44 skew the proxy recorded as 0 (MAE 0.55).
        # Mirror the collector: require BOTH 25Δ legs to have a live contract
        # within 0.08 delta of 0.25 (same threshold as the GridReliability block
        # below and _reliability's n<0.08), else the skew is 0.0.
        _put_25d_reliable = (len(_puts) >= 2 and
                             min(abs(0.25 - p[0]) for p in _puts) < 0.08)
        _call_25d_reliable = (len(_calls) >= 2 and
                              min(abs(0.25 - c[0]) for c in _calls) < 0.08)
        put_call_skew = 0.0
        if (_put_25d_iv > 0.001 and _call_25d_iv > 0.001
                and _put_25d_reliable and _call_25d_reliable):
            put_call_skew = _put_25d_iv - _call_25d_iv

        # SessionPosition: (price - session_low) / (session_high - session_low)
        if not hasattr(self, '_session_high') or self._session_high is None:
            self._session_high = spx_price
            self._session_low = spx_price
        self._session_high = max(self._session_high, spx_price)
        self._session_low = min(self._session_low, spx_price)
        session_range = self._session_high - self._session_low
        session_position = (spx_price - self._session_low) / session_range if session_range > 0.01 else 0.5

        # GridReliability: fraction of 11 delta grid points with valid
        # spline-interpolated IV. Matches proxy's _reliability() which checks
        # if spline-interpolated IV at each target delta is > 0.001.
        # Uses the same put/call spline data built above for ATM_IV.
        # 11 grid points: 5Dp,10Dp,25Dp,40Dp,ATMp, ATM, ATMc,40Dc,25Dc,10Dc,5Dc
        # Put-side targets (abs delta): 0.05, 0.10, 0.25, 0.40, 0.50
        # Call-side targets (delta): 0.50, 0.40, 0.25, 0.10, 0.05
        # ATM = average of put ATM and call ATM (index 5)
        grid_reliability = 0.0
        if self._cached_chain:
            _grid_put_deltas = [0.05, 0.10, 0.25, 0.40, 0.50]
            _grid_call_deltas = [0.50, 0.40, 0.25, 0.10, 0.05]
            _n_valid = 0
            # Put side (5 points)
            if len(_puts) >= 2:
                _put_d = [p[0] for p in _puts]
                _put_iv_arr = [p[1] for p in _puts]
                for _td in _grid_put_deltas:
                    # Check if target delta is within range of observed deltas
                    # (proxy's _reliability checks min distance < 0.08)
                    _min_dist = min(abs(_td - d) for d in _put_d)
                    if _min_dist < 0.08:
                        if len(_puts) >= 4:
                            _giv = self._cubic_spline_eval(_put_d, _put_iv_arr, _td)
                        else:
                            _giv = self._linear_interp(_put_d, _put_iv_arr, _td)
                        if _giv > 0.001:
                            _n_valid += 1
                    # (no elif — 0.08 threshold already covers tight matches)
            # Call side (5 points)
            if len(_calls) >= 2:
                _call_d = [c[0] for c in _calls]
                _call_iv_arr = [c[1] for c in _calls]
                for _td in _grid_call_deltas:
                    _min_dist = min(abs(_td - d) for d in _call_d)
                    if _min_dist < 0.08:
                        if len(_calls) >= 4:
                            _giv = self._cubic_spline_eval(_call_d, _call_iv_arr, _td)
                        else:
                            _giv = self._linear_interp(_call_d, _call_iv_arr, _td)
                        if _giv > 0.001:
                            _n_valid += 1
                    elif _min_dist < 0.02:
                        _n_valid += 1
            # ATM point: valid if either side has ATM IV
            if _put_atm_iv > 0.001 or _call_atm_iv > 0.001:
                _n_valid += 1
            grid_reliability = min(_n_valid / 11.0, 1.0)

        # Apply normalization to all REAL terminals.
        # ROOT-CAUSE FIXES ONLY (no empirical curve-fitting):
        # - ATM_IV/PutCallSkew/RawSpread/GridReliability: cubic spline interpolation
        #   over SELF-COMPUTED Black-Scholes IV+delta (P0-4), matching the collector's
        #   _bs_iv/_fill_side. The 0.3869x+0.0292 IV correction is removed: same IV
        #   model train and serve, so no fitted bias is needed.
        # - VIXChange: real CBOE VIX day-over-day (P0-5). Prior bug: differenced the ATM_IV*100 proxy.
        # - RealizedVol30m: 60x scale correction. Root cause: QC backtest H/L bars differ from Research.
        # - VIXTermSlope: real CBOE VIX9D - VIX (P0-5). Prior bug: false "unavailable" premise -> dead constant.
        # - RV5d: ddof=0 matching proxy. Root cause: statistical formula difference.
        # - OvernightGap/SessionReturn: Securities.Open. Root cause: wrong price source.
        return {{
            "ATM_IV": self._normalize("ATM_IV", atm_iv),
            "VIXSpot": self._normalize("VIXSpot", vix_spot),
            "VIXTermSlope": self._normalize("VIXTermSlope", vix_term_slope),
            "RealizedVol30m": self._normalize("RealizedVol30m", rv30m),
            "RawSpread": self._normalize("RawSpread", raw_spread),
            "_raw_RawSpread": raw_spread,  # internal: un-normalized for delta computation
            "DeltaSpread1": 0.0,  # computed in _on_bar after buffer update
            "DeltaSpread5": 0.0,  # computed in _on_bar after buffer update
            "MinutesToClose": self._normalize("MinutesToClose", minutes_to_close),
            "BarOfDay": self._normalize("BarOfDay", bar_of_day),
            "ThetaUrgency": self._normalize("ThetaUrgency", 1.0 / math.sqrt(max(minutes_to_close, 1.0))),
            "SessionReturn": self._normalize("SessionReturn", session_return),
            "ATM_IV_5m": self._normalize("ATM_IV_5m", atm_iv_5m),
            "RealizedVol30m_5m": self._normalize("RealizedVol30m_5m", rv30m_5m),
            "RawSpread_5m": self._normalize("RawSpread_5m", raw_spread_5m),
            # P0-5: VIXChange = day-over-day change in REAL CBOE VIX (today's VIX
            # minus the prior day's), matching the proxy's VIXSpot day-over-day
            # difference (VIX-scale ~+/-1-2 pts; norm center=0, scale=1.5). Replaces
            # the prior ATM_IV*100 day-over-day, which differenced a ~10-scale
            # self-derived proxy rather than the ~17-scale CBOE VIX the GP trained on.
            "VIXChange": self._normalize("VIXChange", (vix_spot - getattr(self, '_prev_day_vix', vix_spot)) if getattr(self, '_prev_day_vix', 0.0) > 0.0 else 0.0),
            "PutCallSkew": self._normalize("PutCallSkew", put_call_skew),
            "SessionPosition": self._normalize("SessionPosition", session_position),
            "OvernightGap": self._normalize("OvernightGap", getattr(self, '_overnight_gap', 0.0)),
            "GridReliability": self._normalize("GridReliability", grid_reliability),
            "SPXClose": spx_price,  # NOT normalized — used for strike pricing, not GP terminal
            # Inter-day terminals: multi-day context (3-5 day lookback).
            # Computed from daily close prices stored in _daily_closes buffer.
            "SPXReturn3d": self._normalize("SPXReturn3d", self._get_interday("SPXReturn3d")),
            "VIXMean5d": self._normalize("VIXMean5d", self._get_interday("VIXMean5d")),
            "RV5d": self._normalize("RV5d", self._get_interday("RV5d")),
            "IVRVGap5d": self._normalize("IVRVGap5d", self._get_interday("IVRVGap5d")),
        }}

    def _compute_rv30m(self):
        """30-bar Parkinson realized volatility (single-window, session-reset).

        Parkinson std: sqrt(sum(ln(H/L)^2) / (4*ln2*N)) over the in-session rolling
        30-bar window. Proxy parity (2026-06-01):
          - the buffer is CLEARED at the day boundary (a fresh 0DTE book has no
            overnight realized path; the carry was a lookahead-flavored leak);
          - PARTIAL windows are returned during warmup (any N>=1) so the morning
            ramps continuously instead of flooring to a fallback — this is what
            fixed the RealizedVol30m_5m -1.7 morning pin;
          - the proxy column is now the SINGLE-sqrt Parkinson std (the old
            sqrt(v112) double-sqrt was removed in generate_minute_parquet.py), so
            QC's raw Parkinson matches it directly and the former 60x scale
            collapses to ~1.0 (RV_SCALE, confirmed/refined by the G11 diagnostic).
        """
        if not hasattr(self, '_rv_log_hl_buf'):
            self._rv_log_hl_buf = deque(maxlen=30)
        bar = self.Securities[self.spx]
        h, l = bar.High, bar.Low
        if h > 0 and l > 0 and h >= l:
            self._rv_log_hl_buf.append(math.log(h / l))
        if not self._rv_log_hl_buf:
            return 0.0
        try:
            vals = list(self._rv_log_hl_buf)
            raw_rv = math.sqrt(sum(x * x for x in vals) / (4 * math.log(2) * len(vals)))
            return raw_rv * 1.0   # RV_SCALE
        except Exception:
            return 0.0

    # VIXTermSlope (VIX9D - VIX) is computed in _compute_scalars from the real
    # CBOE VIX/VIX9D subscriptions added in Initialize (P0-5), matching the
    # collector. VIX futures were never the right source (futures basis is not
    # the term-structure slope).

    def _compute_expected_credit(self, spx_price):
        """Compute expected credit/debit from BS model matching proxy.

        Uses the same skew-adjusted BS pricing as the proxy's Edgeworth
        evaluator, with the per-template credit correction factor.
        Enables credit guard to reject fills that diverge >40% from
        expected. (Codegen divergence #3 fix, 2026-05-19)
        """
        # T3: single-sourced from the proxy's _BASE_CREDIT_FACTORS at gen time
        # (was a stale hardcoded literal 0.73-0.81 vs the proxy's recalibrated
        # 0.88-0.90 -> drift). Baked here so the served code can't diverge.
        _CREDIT_FACTORS = {credit_factors_literal}
        # _derived_vix is VIX-scale (×100); BS needs IV-scale (÷100)
        iv = getattr(self, '_derived_vix', 15.0) / 100.0
        _market_close = self.Time.replace(hour=16, minute=15, second=0)
        mtc = max(1.0, (_market_close - self.Time).total_seconds() / 60.0)
        total = 0.0
        for leg_type, delta, sign, ratio in {legs_tuples}:
            strike = self._bs_delta_to_strike(delta, spx_price, iv, mtc)
            tau = max(mtc, 1.0) / (252.0 * 390.0)
            sigma_sqrt_tau = iv * math.sqrt(max(tau, 1e-12))
            d1 = (math.log(spx_price / max(strike, 1.0)) + 0.5 * sigma_sqrt_tau**2) / max(sigma_sqrt_tau, 1e-9)
            d2 = d1 - sigma_sqrt_tau
            nd1 = 0.5 * math.erfc(-d1 / math.sqrt(2.0))
            nd2 = 0.5 * math.erfc(-d2 / math.sqrt(2.0))
            if leg_type == "call":
                val = spx_price * nd1 - strike * nd2
            else:
                val = strike * (1 - nd2) - spx_price * (1 - nd1)
            total += sign * ratio * val
        # Apply per-template credit correction
        factor = _CREDIT_FACTORS.get("{template_name}", 0.80)
        return total * factor * 100  # ×100 for SPX multiplier

    def _size_n_contracts(self, spx_price, size_mult):
        """Position sizing — faithful transcription of the proxy
        (evaluator_vectorized.py:1600-1699), in $/share with NOTIONAL=1000.

        Units: the contract-count formula is IDENTICAL to the proxy's
        (n = int(NOTIONAL*size/abs_val), NOTIONAL=1000 both sides). Equity scaling
        also matches exactly: with SetCash(100000) the same trade earns
        pnl_pershare*n*100 real $ (SPX x100 mult), so equity = TotalPortfolioValue/
        (NOTIONAL*100)-1 = pnl_pershare*n/1000 = the proxy's `equity += pnl/notional`
        (the 100x in PnL and the 100x divisor cancel). So n_QC ~= n_proxy and the
        equity-floor de-levering transfers. The ONE residual: abs_val here uses QC
        LIVE-CHAIN gross (option b) while the proxy uses its modeled BS/Edgeworth
        surface; these differ by ~the same chain-vs-model gap the credit factors
        (0.73-0.81) bridge (~20-27%), so n_QC ~= n_proxy up to that bounded factor,
        NOT exactly. The per-template QC trade-by-trade MEASURES this residual
        (logic-check re-derives n from QC's own fills). Replaces the prior IV-blind
        wing_pct sizing (diverged 5-20x; audit 2026-05-31).

        CAVEAT (Level B only): for dynamic-delta templates this sizes off the STATIC
        {legs_tuples} midpoint deltas, while the order trades the GP's dynamic delta
        (_sig_delta) — so RPB / Level-B sizing is approximate (bounded ~1.5-2x at the
        delta-range extremes). V1 (no delta tree) is EXACT (static legs == ordered
        legs == proxy legs). The dynamic path is fixed before the RPB calibration."""
        NOTIONAL = 1000.0
        iv = (self._last_valid_iv if getattr(self, '_last_valid_iv', 0.0) > 0.0
              else (getattr(self, '_derived_vix', 15.0) / 100.0))
        _mc = self.Time.replace(hour=16, minute=15, second=0)
        mtc = max(1.0, (_mc - self.Time).total_seconds() / 60.0)
        tau = max(mtc, 1.0) / (252.0 * 390.0)
        sst = iv * math.sqrt(max(tau, 1e-12))
        gross = 0.0
        net = 0.0
        strikes = []
        for _lt, _dl, _sg, _rt in {legs_tuples}:
            # (b) gross alignment: price each leg from QC's LIVE chain (real quotes
            # — the SAME market the proxy's empirical IV surface was built from, so
            # this matches the proxy's actual gross basis better than a BS model AND
            # is what a live deployment sizes on). Use the contract _find_contract
            # will actually trade, so sizing and execution agree. Fall back to a
            # skew-free BS value per leg ONLY when the chain quote is missing/zero
            # (e.g. $0-bid deep-OTM wings).
            _c = self._find_contract(_lt, _dl, spx_price)
            _price = 0.0
            if _c is not None:
                _bid = _c.BidPrice if (_c.BidPrice and _c.BidPrice > 0) else 0.0
                _ask = _c.AskPrice if (_c.AskPrice and _c.AskPrice > 0) else 0.0
                if _bid > 0.0 and _ask > 0.0:
                    _price = 0.5 * (_bid + _ask)
                elif _bid > 0.0 or _ask > 0.0:
                    # one-sided quote (only bid or only ask present): use it rather
                    # than dropping to the skew-free BS fallback. A $0-bid deep-OTM
                    # wing keeps _price=0 here and correctly falls through to BS.
                    _price = max(_bid, _ask)
                elif _c.LastPrice and _c.LastPrice > 0:
                    # OptionContract has NO bare Price attribute (that belongs to
                    # Securities/holdings) -- reading it threw a Runtime Error the
                    # moment a both-sides-missing contract was sized. LastPrice (last
                    # traded mark) is the correct chain-side fallback.
                    _price = float(_c.LastPrice)
                _k = float(_c.Strike)
            else:
                _k = self._bs_delta_to_strike(_dl, spx_price, iv, mtc)
            if _price <= 0.0:
                # per-leg BS fallback (chain quote unavailable)
                _d1 = (math.log(spx_price / max(_k, 1.0)) + 0.5 * sst * sst) / max(sst, 1e-9)
                _d2 = _d1 - sst
                _nd1 = 0.5 * math.erfc(-_d1 / math.sqrt(2.0))
                _nd2 = 0.5 * math.erfc(-_d2 / math.sqrt(2.0))
                _price = ((spx_price * _nd1 - _k * _nd2) if _lt == "call"
                          else (_k * (1.0 - _nd2) - spx_price * (1.0 - _nd1)))
            strikes.append(_k)
            gross += abs(_sg * _rt * _price)
            net += _sg * _rt * _price
        # abs_val basis (proxy :1611-1623)
        if {is_credit}:
            abs_val = max(gross, 2.0)
        else:
            abs_val = max(gross, abs(net), 2.0)
        _total_ratio = sum(_rt for _, _, _, _rt in {legs_tuples})
        # ratio adjustment + 5% cap apply ONLY to genuine ratio structures (a leg
        # with ratio>1, e.g. RPB) — NOT to 4-leg defined-risk IC/IB (proxy fix
        # 2026-05-31: `sum>2` mis-fired for IC/IB, forcing n=1).
        _has_ratio = any(_rt > 1 for _, _, _, _rt in {legs_tuples})
        if _has_ratio:
            abs_val *= _total_ratio / 2.0
        n = min(int(NOTIONAL * size_mult / max(abs_val, 1e-9)), int(NOTIONAL / 2.0))
        # 5% notional risk cap for ratio structures (proxy :1631-1637)
        if _has_ratio and len(strikes) >= 2:
            _mw = max(abs(strikes[_j] - strikes[_j + 1]) for _j in range(len(strikes) - 1))
            n = min(n, max(1, int(NOTIONAL * 0.05 / max(_mw, 1.0))))
        if n < 1:
            n = 1
        # leverage cap (proxy :1641)
        if n * abs_val > 2.0 * NOTIONAL:
            n = max(1, int(2.0 * NOTIONAL / max(abs_val, 1e-9)))
        # margin gate + equity floor + concentration cap (proxy :1647-1699)
        if len(strikes) >= 2:
            if len(strikes) == 4:
                _msw = max(abs(strikes[0] - strikes[1]), abs(strikes[2] - strikes[3]))
            else:
                _msw = max(abs(strikes[_j] - strikes[_j + 1]) for _j in range(len(strikes) - 1))
            # credit received reduces margin (proxy :1676): net < 0 for credit
            _mpc = max(0.0, _msw - abs(net)) if net < -0.01 else _msw
            # equity = QC realized account return; account = NOTIONAL x 100 mult.
            # Floored at -90% so available_capital never < 10% of notional (proxy :1688).
            _equity = (self.Portfolio.TotalPortfolioValue / (NOTIONAL * 100.0)) - 1.0
            _avail = NOTIONAL * max(1.0 + _equity, 0.10)
            if _mpc * n > _avail:
                _safe = int(_avail / max(_mpc, 1.0))
                if _safe < 1:
                    return 0  # insufficient margin -> skip (proxy `continue`)
                n = _safe
            _maxc = max(1, int(0.50 * _avail / max(_mpc, 1.0)))
            n = min(n, _maxc)
        return int(n)

    def _get_interday(self, name):
        """Compute inter-day terminals from daily close/IV history buffers.

        Matches proxy's inter-day terminal computation in
        evaluator_vectorized.prepare_terminal_data().

        Buffer contents (corrected at session boundary):
        - _daily_closes: raw SPX close prices
        - _daily_ivs: real CBOE VIX (daily), prior-day value rolled in at boundary
        """
        if not hasattr(self, '_daily_closes'):
            self._daily_closes = deque(maxlen=10)
            self._daily_ivs = deque(maxlen=10)
        closes = list(self._daily_closes)
        ivs = list(self._daily_ivs)
        if name == "SPXReturn3d":
            # Proxy: (close[d-1] - close[d-4]) / close[d-4]
            # Uses d >= 4 check with 0-indexed days.
            # In buffer terms: need 5 entries (indices 0..4) to get
            # entries[-1] vs entries[-4].
            if len(closes) >= 5:
                return (closes[-1] - closes[-4]) / max(closes[-4], 1.0)
            return 0.0
        elif name == "VIXMean5d":
            # Proxy: mean of VIXSpot at session opens for 5 prior days.
            # _daily_ivs stores real CBOE VIX (P0-5).
            if len(ivs) >= 5:
                return sum(ivs[-5:]) / 5.0
            return sum(ivs) / max(len(ivs), 1) if ivs else 15.0  # VIX-scale fallback
        elif name == "RV5d":
            # Proxy: np.std(daily_rets[d-6:d-1]) * sqrt(252), fired when d >= 6.
            # daily_rets = diff(log(day_closes)); the window is the 5 returns
            # ending at day d-1, drawn from the LAST 6 closes. 6 buffered closes
            # are therefore sufficient.
            #
            # G11 fix (2026-06-01) — off-by-one + negative-index wrap. The old
            # code gated on `len(closes) >= 7` and iterated
            # `for i in range(len-6, len-1): closes[i]/closes[i-1]`. With len==6
            # that range starts at i=0, so `closes[i-1]` == `closes[-1]` WRAPPED
            # to the newest close — a corrupt return. The `>= 7` gate existed only
            # to dodge that wrap, but it made QC fire one day LATER than the proxy
            # (proxy d>=6) and report the proxy's PREVIOUS-day RV5d on every day —
            # a 1-day lag compounding the daily-close-timing gap into the RV5d MAE.
            # Rewriting the window as `range(len-5, len)` over `closes[i]/
            # closes[i-1]` references closes[len-6..len-1] with NO wrap, so it is
            # valid at len==6 and reproduces the proxy's std EXACTLY for all days.
            # ddof=0 (was ddof=1, which inflated RV5d ~12% with N=5).
            if len(closes) >= 6:
                rets = [math.log(closes[i] / max(closes[i-1], 1.0))
                        for i in range(len(closes)-5, len(closes))]
                if rets:
                    mean_r = sum(rets) / len(rets)
                    # ddof=0 to match proxy (np.std default)
                    var = sum((r - mean_r)**2 for r in rets) / len(rets)
                    return math.sqrt(var) * math.sqrt(252)
            return 0.15  # proxy default for d < 6
        elif name == "IVRVGap5d":
            # T2 fix: proxy uses mean(ATM_IV[d-5:d]) - RV5d, NOT VIX. _daily_atm_iv
            # holds prior-days' session-open ATM_IV (IV-scale ~0.13). Using VIX/100
            # (~0.18) gave a +1.38σ offset (variance-risk premium). Fall back to
            # _last_valid_iv until 5 days are buffered.
            atm_ivs = list(self._daily_atm_iv) if hasattr(self, '_daily_atm_iv') else []
            if len(atm_ivs) >= 5:
                iv_mean = sum(atm_ivs[-5:]) / 5.0
            elif atm_ivs:
                iv_mean = sum(atm_ivs) / len(atm_ivs)
            else:
                iv_mean = getattr(self, '_last_valid_iv', 0.15)
            rv = self._get_interday("RV5d")
            return iv_mean - rv
        return 0.0

    # -- Self-computed Black-Scholes IV / delta (P0-4) --
    # Injected verbatim from layer3/bs_iv.py so the served QC code and the
    # importable, unit-tested reference share ONE source. Scipy-free; matches
    # research_collector._bs_iv (brentq bracket [0.01,5.0], r=0.05) and _bs_greeks.
{bs_methods}

    # -- Cubic spline interpolation (matches proxy's scipy.CubicSpline) --

    @staticmethod
    def _dedup_by_delta(pairs):
        """Remove duplicate delta entries (keep first). Matches proxy's np.diff > 1e-6."""
        if not pairs:
            return pairs
        out = [pairs[0]]
        for i in range(1, len(pairs)):
            if abs(pairs[i][0] - pairs[i-1][0]) > 1e-6:
                out.append(pairs[i])
        return out

    @staticmethod
    def _cubic_spline_eval(xs, ys, x_target):
        """Natural cubic spline interpolation at x_target. Pure Python.

        Matches scipy.CubicSpline(xs, ys, extrapolate=True) used in the proxy's
        research_collector.py _fill_side(). Solves the tridiagonal system for
        natural boundary conditions (second derivative = 0 at endpoints).

        Args:
            xs: sorted list of x values (deltas), len >= 4
            ys: corresponding y values (IV or spread)
            x_target: point to evaluate at

        Returns:
            Interpolated y value at x_target, clamped to [0, 5].
        """
        n = len(xs)
        if n < 2:
            return ys[0] if ys else 0.0
        if n == 2:
            # Linear fallback with clamping
            t = (x_target - xs[0]) / max(xs[1] - xs[0], 1e-12)
            return max(0.0, min(5.0, ys[0] + t * (ys[1] - ys[0])))
        if n == 3:
            # Quadratic fallback
            h0 = xs[1] - xs[0]
            h1 = xs[2] - xs[1]
            if h0 < 1e-12 or h1 < 1e-12:
                t = (x_target - xs[0]) / max(xs[-1] - xs[0], 1e-12)
                return ys[0] + t * (ys[-1] - ys[0])
            # Lagrange quadratic
            t0 = ((x_target - xs[1]) * (x_target - xs[2])) / ((xs[0] - xs[1]) * (xs[0] - xs[2]))
            t1 = ((x_target - xs[0]) * (x_target - xs[2])) / ((xs[1] - xs[0]) * (xs[1] - xs[2]))
            t2 = ((x_target - xs[0]) * (x_target - xs[1])) / ((xs[2] - xs[0]) * (xs[2] - xs[1]))
            val = ys[0] * t0 + ys[1] * t1 + ys[2] * t2
            return max(0.0, min(5.0, val))

        # Natural cubic spline: solve tridiagonal system for second derivatives
        # h[i] = xs[i+1] - xs[i]
        h = [xs[i+1] - xs[i] for i in range(n-1)]
        # Check for zero-width intervals
        for i in range(len(h)):
            if h[i] < 1e-12:
                h[i] = 1e-12

        # Set up tridiagonal system for M (second derivatives)
        # Natural BC: M[0] = M[n-1] = 0
        # For i=1..n-2: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1]
        #   = 6*((ys[i+1]-ys[i])/h[i] - (ys[i]-ys[i-1])/h[i-1])
        m = n - 2  # number of unknowns (M[1]..M[n-2])
        if m <= 0:
            # Shouldn't happen with n >= 4, but guard
            t = (x_target - xs[0]) / max(xs[-1] - xs[0], 1e-12)
            return ys[0] + t * (ys[-1] - ys[0])

        # Diagonal, sub-diagonal, super-diagonal, rhs
        diag = [0.0] * m
        sub = [0.0] * m   # sub[i] = h[i] for i=1..m-1
        sup = [0.0] * m   # sup[i] = h[i+1] for i=0..m-2
        rhs = [0.0] * m

        for i in range(m):
            j = i + 1  # index in original array (M[j])
            diag[i] = 2.0 * (h[j-1] + h[j])
            rhs[i] = 6.0 * ((ys[j+1] - ys[j]) / h[j] - (ys[j] - ys[j-1]) / h[j-1])
            if i > 0:
                sub[i] = h[j-1]
            if i < m - 1:
                sup[i] = h[j]

        # Thomas algorithm (tridiagonal solver)
        # Forward sweep
        for i in range(1, m):
            if abs(diag[i-1]) < 1e-15:
                diag[i-1] = 1e-15
            w = sub[i] / diag[i-1]
            diag[i] -= w * sup[i-1]
            rhs[i] -= w * rhs[i-1]

        # Back substitution
        M = [0.0] * n
        if abs(diag[m-1]) < 1e-15:
            diag[m-1] = 1e-15
        M[m] = rhs[m-1] / diag[m-1]  # M[n-2]
        for i in range(m-2, -1, -1):
            if abs(diag[i]) < 1e-15:
                diag[i] = 1e-15
            M[i+1] = (rhs[i] - sup[i] * M[i+2]) / diag[i]
        # M[0] = M[n-1] = 0 (natural BC, already initialized)

        # Find interval for x_target
        # Extrapolation: use first/last interval
        if x_target <= xs[0]:
            k = 0
        elif x_target >= xs[-1]:
            k = n - 2
        else:
            k = 0
            for i in range(n - 1):
                if xs[i] <= x_target <= xs[i+1]:
                    k = i
                    break

        # Evaluate cubic in interval k
        dx = x_target - xs[k]
        hk = h[k]
        a = (M[k+1] - M[k]) / (6.0 * hk)
        b = M[k] / 2.0
        c = (ys[k+1] - ys[k]) / hk - hk * (2.0 * M[k] + M[k+1]) / 6.0
        d = ys[k]
        val = a * dx * dx * dx + b * dx * dx + c * dx + d
        return max(0.0, min(5.0, val))

    @staticmethod
    def _linear_interp(xs, ys, x_target):
        """Piecewise-linear interpolation with extrapolation. Fallback for < 4 points."""
        n = len(xs)
        if n == 0:
            return 0.0
        if n == 1:
            return ys[0]
        # Clamp to first/last segment for extrapolation
        if x_target <= xs[0]:
            t = (x_target - xs[0]) / max(xs[1] - xs[0], 1e-12)
            return ys[0] + t * (ys[1] - ys[0])
        if x_target >= xs[-1]:
            t = (x_target - xs[-2]) / max(xs[-1] - xs[-2], 1e-12)
            return ys[-2] + t * (ys[-1] - ys[-2])
        # Find interval
        for i in range(n - 1):
            if xs[i] <= x_target <= xs[i+1]:
                t = (x_target - xs[i]) / max(xs[i+1] - xs[i], 1e-12)
                return ys[i] + t * (ys[i+1] - ys[i])
        return ys[-1]

    # -- Rolling buffer helpers (match evaluator semantics) --

    def _lag(self, name, k):
        """Get value from k bars ago. UnrealizedProfitPct reads the per-TRADE
        history (self._upp_hist, cleared on entry); the generic len-1-k indexing
        already yields the proxy's `upp_hist[-1-k] if k < len else 0.0`
        (evaluator_vectorized.py:729-733) — the per-trade clear makes the trade-
        start mask at least as tight as the day mask, so no extra day guard is
        needed here. Every other terminal reads the day-cleared generic buffer."""
        k = max(0, min(int(k), self.MAX_LAG))
        if name == "UnrealizedProfitPct":
            # Proxy parity (review 2026-06-03): UPP is the 0 baseline for ANY tree
            # evaluated while FLAT. The proxy substitutes the live per-trade value
            # ONLY into the EXIT tree IN-position (evaluator_vectorized.py:1532-1536);
            # entry/size/delta trees are vectorized with the zeros array (:1294).
            # Reading _upp_hist while flat would surface the PRIOR trade's stale tail
            # and flip an entry-tree Lag(UPP,k) that the proxy holds at 0.
            if not self._position_open:
                return 0.0
            buf = self._upp_hist
        else:
            buf = self._buffers.get(name)
        if not buf or len(buf) == 0:
            return 0.0
        idx = len(buf) - 1 - k
        return float(buf[idx]) if idx >= 0 else 0.0

    def _delta(self, name, k):
        """Current value minus k-bars-ago value. Parity with the proxy's Delta,
        which masks within_day_pos < lag -> 0 (evaluator_vectorized.py:306): until
        the buffer holds MORE than k entries this day, return 0.0. Without this
        guard the k-lagged read is 0 and Delta degrades to (current - 0) = current
        on the first k bars of each day instead of 0 -- a serving-vs-objective skew
        for any strategy containing Delta(<terminal>, k)."""
        k = max(1, min(int(k), self.MAX_LAG))
        if name == "UnrealizedProfitPct":
            # Proxy parity (review 2026-06-03): 0 while FLAT — the proxy's live UPP
            # substitution is exit-tree+in-position only; entry/size/delta trees use
            # the 0 baseline (see _lag). Without this a flat-bar Delta(UPP,k) reads
            # the prior trade's stale tail.
            if not self._position_open:
                return 0.0
            # Proxy parity (evaluator_vectorized.py:757-764): Delta(UPP) masks on
            # the WITHIN-DAY position (day boundary), then subtracts the per-trade
            # lag (0 before the trade started). So in the first bars of a TRADE
            # (but past the first k bars of the DAY) Delta(UPP,k) == cur - 0 == cur,
            # NOT 0 — the generic len(upp_hist)<=k mask would wrongly zero it.
            # within_day_pos+1 == len(_raw_RawSpread) (appended unconditionally
            # every bar at the top of _on_bar, day-cleared with the other buffers),
            # so `len(...) <= k` is exactly the proxy's `within_day_pos < k`.
            if len(self._buffers.get("_raw_RawSpread", ())) <= k:
                return 0.0
            return self._lag(name, 0) - self._lag(name, k)
        buf = self._buffers.get(name)
        if not buf or len(buf) <= k:
            return 0.0
        return self._lag(name, 0) - self._lag(name, k)

    def _lag_expr(self, key, current_val, k):
        """Lag a computed expression: store current value in buffer, return k-lagged."""
        current_val = float(current_val) if current_val == current_val else 0.0
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.MAX_LAG + 1)
        self._buffers[key].append(current_val)
        return self._lag(key, max(0, min(int(k), self.MAX_LAG)))

    def _delta_expr(self, key, current_val, k):
        """Delta of a computed expression: store current, return current - lagged.
        Same within-day mask as _delta (proxy evaluator_vectorized.py:306): until
        the buffer holds MORE than k entries this day, return 0.0 -- otherwise the
        early-day Delta returns (current - 0) = current instead of 0."""
        current_val = float(current_val) if current_val == current_val else 0.0
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.MAX_LAG + 1)
        self._buffers[key].append(current_val)
        k = max(1, min(int(k), self.MAX_LAG))
        if len(self._buffers[key]) <= k:
            return 0.0
        return self._lag(key, 0) - self._lag(key, k)

    def _cross_above(self, ka, kb, a, b):
        """Detect a crossing above b from previous bar."""
        self._curr_eval[ka] = a
        self._curr_eval[kb] = b
        pa = self._prev_eval.get(ka)
        pb = self._prev_eval.get(kb)
        if pa is None or pb is None:
            return False
        return a > b and pa <= pb

    def _cross_below(self, ka, kb, a, b):
        """Detect a crossing below b from previous bar."""
        self._curr_eval[ka] = a
        self._curr_eval[kb] = b
        pa = self._prev_eval.get(ka)
        pb = self._prev_eval.get(kb)
        if pa is None or pb is None:
            return False
        return a < b and pa >= pb

    def _safe(self, x):
        """NaN/Inf guard — matches evaluator _safe_real."""
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return float(x)
        return 0.0

    # -- Strategy tree evaluation (auto-generated) --

    def _eval_entry(self, s):
        """Entry condition — returns bool (NaN-safe)."""
        _v = {entry_expr}
        if isinstance(_v, float) and math.isnan(_v):
            return False
        return bool(_v)

    def _eval_exit(self, s):
        """Exit condition — returns bool (NaN-safe)."""
        _v = {exit_expr}
        if isinstance(_v, float) and math.isnan(_v):
            return False
        return bool(_v)

    def _eval_size(self, s):
        """Position size multiplier — returns float."""
        return self._safe({size_expr})

    def _eval_delta(self, s):
        """Delta tree — returns float in [0,1] for dynamic leg selection.
        GP's delta_tree outputs a raw normalized value; we sigmoid-clamp to [0,1]
        then map through the template's delta_range to get actual short_delta."""
        _raw = self._safe({delta_expr})
        # Sigmoid clamp: maps any real to (0,1), matching proxy's delta_tree handling
        _clamped = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, _raw))))
        return _clamped

    # -- Multi-leg option execution --

    def OnEndOfAlgorithm(self):
        """Log trade exit reasons to ObjectStore for stop-loss calibration."""
        # P0-5b: guaranteed VIX source-health backstop. Fires for any backtest
        # length (the in-loop day>=3 gate may not reach a short run), so a dead
        # CBOE VIX subscription can never produce a silently-accepted result.
        if not getattr(self, '_vix_ever_populated', False):
            self.Error(
                "P0-5b VIX SOURCE FAILURE (final): real CBOE VIX NEVER populated; "
                f"VIXSpot used the ATM_IV*100 proxy "
                f"({{getattr(self, '_vix_fallback_bars', 0)}} fallback bars) and "
                "VIXTermSlope/VIXChange/VIXMean5d used frozen constants. "
                "Result is INVALID for proxy->QC calibration."
            )
            self.SetRuntimeStatistic("vix_source_failed", "1")
        else:
            self.Debug(
                f"P0-5b VIX source OK: real CBOE VIX populated; "
                f"{{getattr(self, '_vix_fallback_bars', 0)}} warmup fallback bars."
            )
        if hasattr(self, '_trade_log') and self._trade_log:
            import json
            _summary = {{}}
            for t in self._trade_log:
                r = t.get("reason", "unknown")
                _summary[r] = _summary.get(r, 0) + 1
            self.Debug(f"Trade exit reasons: {{_summary}}")
            try:
                self.ObjectStore.Save(
                    "trade_exit_log",
                    json.dumps(self._trade_log)
                )
                self.Debug(f"Saved {{len(self._trade_log)}} trade exit records to ObjectStore")
            except Exception as e:
                self.Debug(f"Failed to save trade log: {{e}}")

    def _close_position(self, reason=""):
        """Close all positions. Logs exit reason for calibration diagnostics."""
        self.Debug(reason)
        # Classify exit reason for stop-loss calibration (task #100)
        _reason_tag = "unknown"
        _r = reason.lower()
        if "stop loss" in _r or "max loss" in _r:
            _reason_tag = "stop_loss"
        elif "eod" in _r or "session boundary" in _r or "force close" in _r:
            _reason_tag = "eod"
        elif "time decay" in _r:
            _reason_tag = "time_decay"
        elif "signal" in _r or "exit" in _r:
            _reason_tag = "signal"
        elif "max hold" in _r or "max_hold" in _r:
            _reason_tag = "max_hold"
        if not hasattr(self, '_trade_log'):
            self._trade_log = []
        self._trade_log.append({{
            "bar": self._bar_count,
            "time": str(self.Time),
            "reason": _reason_tag,
            "bars_held": self._bars_in_trade,
            "entry_credit": getattr(self, '_entry_credit', 0),
        }})
        self.Liquidate()
        self._position_open = False
        self._bars_in_trade = 0

    def _open_position(self, s):
        """Open the {template_name} position."""
        if not self._cached_chain or len(self._cached_chain) < {min_chain_len}:
            self._open_fail_cooldown = 3
            return

        spx_price = self.Securities[self.spx].Price
        # Use size from the SIGNAL bar (stored in _pending_size), not re-evaluated
        # on the fill bar — matches the proxy's pending_size behavior. Do NOT call
        # self._eval_size(s) here: _on_bar already evaluated it up-front this bar,
        # and a second call would double-append its Lag/Delta buffer (parity break).
        size_mult = max(0.0, min(1.0, self._pending_size))
        # Position sizing: faithful transcription of the proxy
        # (evaluator_vectorized.py:1600-1699) — entry_gross-based abs_val, ratio
        # scaler + 5% cap, leverage cap, margin gate with equity-floor de-levering,
        # 50% concentration cap, all in $/share with NOTIONAL=1000. Same formula +
        # equity scaling as the proxy, so n_QC ~= n_proxy (up to the bounded
        # chain-vs-model gross gap) and the equity-floor de-levering transfers. The
        # prior IV-blind wing_pct sizing diverged 5-20x per template and broke Sharpe
        # scale-invariance (audit 2026-05-31). See _size_n_contracts.
        n_contracts = self._size_n_contracts(spx_price, size_mult)
        if n_contracts < 1:
            # proxy `continue`: insufficient margin / sub-unit size -> skip entry
            self._open_fail_cooldown = 1
            return

        self._pre_order_cash = self.Portfolio.Cash
        try:
{open_position_code}
            # P0 fix: set position_open AFTER all orders succeed (not before)
            # If we get here, all MarketOrders were submitted successfully.
            self._position_open = True
            self._bars_in_trade = 0
            # Fresh per-trade UnrealizedProfitPct history (proxy parity:
            # evaluator_vectorized.py:2243 upp_hist.clear() on each new entry — no
            # cross-trade leak). The entry bar appends nothing; the in-position UPP
            # append in _on_bar starts the NEXT bar, so on the first full bar of the
            # trade len(self._upp_hist)==1 and Lag(UPP,k>=1)==0, matching the proxy.
            self._upp_hist.clear()
            self._trades_today += 1
            # Track credit received for stop-loss: snapshot cash change from fills.
            # MarketOrders on liquid SPX 0DTE fill synchronously within the call.
            self._entry_credit = self.Portfolio.Cash - self._pre_order_cash
            # B-3 (campaign sweep): gross |premium| at entry = Sum |HoldingsValue| over
            # the invested option legs (dollars, same basis as unrealised_pnl). The
            # proxy's debit max-loss gate uses max(entry_gross, net) — for a ratio
            # backspread the gross (~$18) >> the net debit (~$2), so a net-only basis
            # stops ~9x too early vs the proxy's 80%-of-gross. Mirror it below.
            self._entry_gross = sum(abs(h.HoldingsValue) for h in self.Portfolio.Values
                                    if getattr(h, "Invested", False))
            # Credit guard (codegen divergence #3 fix): reject entries where
            # actual fill diverges >40% from expected. Prevents positions
            # that the proxy would never have entered.
            _expected_credit = self._compute_expected_credit(spx_price)
            if _expected_credit != 0 and abs(self._entry_credit) > 0.01:
                _fill_ratio = self._entry_credit / _expected_credit
                # Log credit divergence for reconciliation — DO NOT reject.
                # The BS-based expected_credit is too inaccurate for 0DTE:
                # real option prices are dominated by microstructure (bid-ask,
                # vol surface dynamics, early-exercise bounds) not BS model.
                # Rejecting on >40% divergence was liquidating nearly ALL entries,
                # causing 0% win rate. Credit factors will be recalibrated from
                # QC order data once we have successful trades to compare against.
                self.Debug(f"Credit: actual={{self._entry_credit:.0f}} expected={{_expected_credit:.0f}} ratio={{_fill_ratio:.2f}}")
        except Exception as e:
            self.Debug(f"Open position error: {{e}}")
            # P0 fix: cleanup any partial fills from failed multi-leg open
            if self.Portfolio.Invested:
                self.Liquidate()
                self.Debug("Cleaned up partial fill after open failure")
            self._open_fail_cooldown = 3  # wait 3 bars before retrying

    def _bs_delta_to_strike(self, target_delta, spx_price, iv, mtc):
        """BS delta inversion with skew — a faithful transcription of the proxy's
        _delta_to_strike + _skew_iv (evaluator._delta_to_strike,
        evaluator_vectorized._skew_iv), so QC executes the SAME strike the proxy
        modeled and evolved against (P0-7). Newton-Raphson on skew_slope=-0.15 with
        intraday slope dynamics (Todorov 2019; Dim, Eraker & Vilkov 2023) +
        tau-damping (tau_ref=60min). Byte-identical to the proxy across the 0DTE
        grid (incl. mtc<60) — locked by
        tests/test_codegen.py::test_codegen_strike_matches_proxy_delta_to_strike.
        """
        if iv <= 0 or mtc <= 0:
            # Proxy degenerate fallback (evaluator._delta_to_strike): near-ATM.
            return round((spx_price * (1.0 - target_delta * 0.01)) / 5.0) * 5.0

        tau = mtc / (252.0 * 390.0)
        is_call = target_delta > 0
        sigma_sqrt_tau = iv * math.sqrt(max(tau, 1e-12))
        # Initial guess: ATM-IV inversion ignoring skew (clamped like the proxy).
        if is_call:
            _dc = max(min(target_delta, 0.999), 0.001)
        else:
            _dc = max(min(1.0 + target_delta, 0.999), 0.001)
        d1 = _norm_ppf(_dc)
        raw = spx_price * math.exp(-d1 * sigma_sqrt_tau + 0.5 * sigma_sqrt_tau ** 2)

        # Newton-Raphson: refine so BS delta under skew-adjusted IV hits target.
        _SQRT2 = math.sqrt(2.0)
        _INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
        # Loop-invariant skew params — mirror _skew_iv EXACTLY (tau_ref=60min).
        if mtc > 240:
            _eff_slope = -0.15 * 0.7   # morning: flatter
        elif mtc > 120:
            _eff_slope = -0.15         # midday: standard
        else:
            _eff_slope = -0.15 * 1.3   # afternoon: steeper
        _tau_damp = min(1.0, math.sqrt(tau / (60.0 / (252.0 * 390.0))))
        for _ in range(5):
            # Inline _skew_iv(iv, spx_price, raw, mtc, -0.15):
            if iv < 0.001 or raw <= 0:
                # Proxy _skew_iv degenerate guard: max(atm_iv, 0.01) returned
                # BEFORE the iv*3 cap. Unreachable in production (callers floor
                # iv>=0.01), kept so the transcription is faithful in isolation.
                skew_iv = max(iv, 0.01)
            else:
                _d_std = ((raw - spx_price) / (spx_price * sigma_sqrt_tau)) * _tau_damp
                skew_iv = iv * (1.0 + _eff_slope * _d_std)
                if raw > spx_price:                   # call-side discount floor
                    skew_iv = max(skew_iv, iv * 0.92)
                skew_iv = min(max(skew_iv, iv * 0.5, 0.01), iv * 3.0)
            s_sqrt_t = skew_iv * math.sqrt(max(tau, 1e-12))
            if s_sqrt_t < 1e-9:
                break
            d1_skew = (math.log(spx_price / max(raw, 1.0)) + 0.5 * s_sqrt_t ** 2) / s_sqrt_t
            cdf_d1 = 0.5 * math.erfc(-d1_skew / _SQRT2)
            actual_delta = cdf_d1 if is_call else cdf_d1 - 1.0
            err = actual_delta - target_delta
            if abs(err) < 0.001:
                break
            pdf_d1 = _INV_SQRT_2PI * math.exp(-0.5 * d1_skew * d1_skew)
            d_delta_dK = -pdf_d1 / (max(raw, 1.0) * s_sqrt_t)
            if abs(d_delta_dK) < 1e-12:
                break
            raw = raw - err / d_delta_dK
            raw = max(raw, spx_price * 0.7)
            raw = min(raw, spx_price * 1.3)

        return round(raw / 5.0) * 5.0

    def _find_contract(self, option_type, target_delta, spx_price):
        """Select the contract whose STRIKE matches the proxy's delta->strike map.

        P0-7: the proxy (evaluator._delta_to_strike) picks each leg's strike by
        inverting BS delta on the *collector-scale* ATM IV with skew_slope=-0.15
        (Newton-Raphson) — it NEVER consults per-contract market IV. Selecting by
        QC-native c.Greeks.Delta instead keys off QC's IV (~1.9x the collector's,
        see P0-4), so QC would execute a DIFFERENT strike than the proxy modeled
        and evolved against — an execution-axis training-serving skew (Sculley et
        al., 2015). So compute the proxy's target strike with the SAME inversion
        (_bs_delta_to_strike, fed self._last_valid_iv = the P0-4 self-computed ATM
        IV) and snap to the nearest real strike. QC Greeks are never used.
        """
        if not self._cached_chain:
            return None

        right = OptionRight.Call if option_type == "call" else OptionRight.Put
        candidates = [c for c in self._cached_chain if c.Right == right]
        if not candidates:
            return None

        iv_est = self._last_valid_iv if self._last_valid_iv > 0.01 else 0.15
        _market_close = self.Time.replace(hour=16, minute=15, second=0)
        _mtc = max((_market_close - self.Time).total_seconds() / 60.0, 1.0)
        target_strike = self._bs_delta_to_strike(target_delta, spx_price, iv_est, _mtc)
        return min(candidates, key=lambda c: abs(c.Strike - target_strike))
'''


# ---------------------------------------------------------------------------
# Position opening code generation (per template)
# ---------------------------------------------------------------------------

def _gen_dynamic_open_position_code(template_name: str) -> str:
    """Generate _open_position body with DYNAMIC delta from delta_tree.

    Instead of baked-in fixed deltas, the code reads self._sig_delta (the
    delta_tree value _on_bar evaluated up-front this bar) and maps it through the
    template's delta_range to compute actual strike deltas per-trade. This matches
    the proxy evaluator's Level B behavior where delta_tree output selects the
    short delta (delta_signals[i] at the fill bar).
    """
    dr = BASE_TEMPLATE_DELTA_RANGES[template_name]
    min_d, max_d, wing_offset, delta_fixed = dr

    lines = []
    lines.append(f"            # Dynamic delta from GP delta_tree (Level B). Use the up-front")
    lines.append(f"            # value _on_bar computed this bar (continuous buffer); do NOT")
    lines.append(f"            # re-evaluate here -- that would double-append the delta buffer.")
    lines.append(f"            _delta_val = self._sig_delta")
    lines.append(f"            _short_delta = {min_d} + _delta_val * ({max_d} - {min_d})")
    lines.append(f"            self._entry_short_delta = _short_delta  # for delta-dependent stop-loss")
    lines.append(f'            self.Debug(f"Delta tree -> {{{{_delta_val:.3f}}}} -> short_delta={{{{_short_delta:.3f}}}}")')

    if template_name == "iron_condor":
        lines.append(f"            _long_delta = max(_short_delta - {wing_offset}, 0.05)")
        lines.append(f'            c0 = self._find_contract("call", +_short_delta, spx_price)')
        lines.append(f'            c1 = self._find_contract("call", +_long_delta, spx_price)')
        lines.append(f'            c2 = self._find_contract("put", -_short_delta, spx_price)')
        lines.append(f'            c3 = self._find_contract("put", -_long_delta, spx_price)')
        n_legs = 4
        leg_orders = [
            ("c0", "-n_contracts"),
            ("c1", "n_contracts"),
            ("c2", "-n_contracts"),
            ("c3", "n_contracts"),
        ]
    elif template_name == "iron_butterfly":
        lines.append(f"            _wing_delta = _short_delta  # controls wing distance from ATM")
        # IB short legs are ATM (0.50); the delta_tree controls the WING, not the
        # short. The stop-loss base keys on the SHORT delta, so pin it to 0.50 to
        # match the proxy (evaluator_vectorized._compute_dynamic_legs returns the
        # 0.50 short as _dyn[0]; _entry_short_delta = abs(0.50)). Without this the
        # served stop would key on the ~0.20 wing -> ~24% too tight (P0-3 IB fix).
        lines.append(f"            self._entry_short_delta = 0.50  # IB short is ATM; stop keys on 0.50 short (proxy parity)")
        lines.append(f'            c0 = self._find_contract("call", +0.50, spx_price)')
        lines.append(f'            c1 = self._find_contract("call", +_wing_delta, spx_price)')
        lines.append(f'            c2 = self._find_contract("put", -0.50, spx_price)')
        lines.append(f'            c3 = self._find_contract("put", -_wing_delta, spx_price)')
        n_legs = 4
        leg_orders = [
            ("c0", "-n_contracts"),
            ("c1", "n_contracts"),
            ("c2", "-n_contracts"),
            ("c3", "n_contracts"),
        ]
    elif template_name == "bull_put_credit":
        lines.append(f"            _long_delta = max(_short_delta - {wing_offset}, 0.05)")
        lines.append(f'            c0 = self._find_contract("put", -_short_delta, spx_price)')
        lines.append(f'            c1 = self._find_contract("put", -_long_delta, spx_price)')
        n_legs = 2
        leg_orders = [
            ("c0", "-n_contracts"),
            ("c1", "n_contracts"),
        ]
    elif template_name == "bear_call_credit":
        lines.append(f"            _long_delta = max(_short_delta - {wing_offset}, 0.05)")
        lines.append(f'            c0 = self._find_contract("call", +_short_delta, spx_price)')
        lines.append(f'            c1 = self._find_contract("call", +_long_delta, spx_price)')
        n_legs = 2
        leg_orders = [
            ("c0", "-n_contracts"),
            ("c1", "n_contracts"),
        ]
    elif template_name == "ratio_put_backspread":
        lines.append(f"            _long_delta = max(_short_delta - {wing_offset}, 0.05)")
        lines.append(f'            c0 = self._find_contract("put", -_short_delta, spx_price)')
        lines.append(f'            c1 = self._find_contract("put", -_long_delta, spx_price)')
        n_legs = 2
        leg_orders = [
            ("c0", "-n_contracts"),       # sell 1 near-ATM put
            ("c1", "n_contracts * 2"),    # buy 2 OTM puts
        ]
    else:
        raise ValueError(f"Unknown template for dynamic delta: {template_name}")

    lines.append("")
    # Null check
    checks = " or ".join(f"{c} is None" for c, _ in leg_orders)
    lines.append(f"            if {checks}:")
    lines.append(f'                self.Debug("Could not find all contracts for {template_name} (dynamic delta)")')
    lines.append(f"                self._open_fail_cooldown = 3")
    lines.append(f"                return")
    lines.append("")
    # Price check
    price_checks = " or ".join(f"self.Securities[{c}.Symbol].Price == 0" for c, _ in leg_orders)
    lines.append(f"            if {price_checks}:")
    lines.append(f'                self.Debug("Contract(s) have no price data, waiting")')
    lines.append(f"                self._open_fail_cooldown = 3")
    lines.append(f"                return")
    lines.append("")
    # Strike separation for same-side legs
    if template_name in ("bull_put_credit", "bear_call_credit", "ratio_put_backspread"):
        lines.append(f"            if abs(c0.Strike - c1.Strike) < 5:")
        lines.append(f'                self.Debug("Strike separation < $5 between legs, skipping")')
        lines.append(f"                self._open_fail_cooldown = 3")
        lines.append(f"                return")
        lines.append("")
    elif template_name in ("iron_condor", "iron_butterfly"):
        lines.append(f"            if abs(c0.Strike - c1.Strike) < 5 or abs(c2.Strike - c3.Strike) < 5:")
        lines.append(f'                self.Debug("Strike separation < $5 between legs, skipping")')
        lines.append(f"                self._open_fail_cooldown = 3")
        lines.append(f"                return")
        lines.append("")
    # Spread width cap: $40 max. QC v5 data shows $41+ width spreads
    # produce -$4,205 PnL (4 of 9 catastrophic losses). Calibrated from
    # 104 QC trades: $26-40 width = +$2,160; $41+ = -$4,205.
    if template_name in ("bull_put_credit", "bear_call_credit", "ratio_put_backspread"):
        lines.append(f"            _spread_width = abs(c0.Strike - c1.Strike)")
    elif template_name in ("iron_condor", "iron_butterfly"):
        lines.append(f"            _spread_width = max(abs(c0.Strike - c1.Strike), abs(c2.Strike - c3.Strike))")
    lines.append(f"            if _spread_width > 40:")
    lines.append(f'                self.Debug(f"Spread width ${{_spread_width:.0f}} > $40 cap, skipping")')
    lines.append(f"                self._open_fail_cooldown = 3")
    lines.append(f"                return")
    lines.append("")

    # Orders
    for contract_var, qty_expr in leg_orders:
        lines.append(f"            self.MarketOrder({contract_var}.Symbol, {qty_expr})")

    return "\n".join(lines)


def _gen_open_position_code(template_name: str, legs_override: "Optional[List[QCLeg]]" = None) -> str:
    """Generate the body of _open_position for a specific template (STATIC deltas)."""
    legs = legs_override if legs_override is not None else TEMPLATE_LEGS[template_name]
    lines = []
    for i, leg in enumerate(legs):
        lines.append(
            f'            c{i} = self._find_contract('
            f'"{leg.option_type}", {leg.delta_target}, spx_price)'
        )
    lines.append("")
    # Check all contracts found
    checks = " or ".join(f"c{i} is None" for i in range(len(legs)))
    lines.append(f"            if {checks}:")
    lines.append(f'                self.Debug("Could not find all contracts for {template_name}")')
    lines.append(f"                self._open_fail_cooldown = 3")
    lines.append(f"                return")
    lines.append("")
    # Validate all contracts have received price data (avoid "security does not
    # have an accurate price" runtime error — data arrives asynchronously)
    price_checks = " or ".join(
        f"self.Securities[c{i}.Symbol].Price == 0" for i in range(len(legs))
    )
    lines.append(f"            if {price_checks}:")
    lines.append(f'                self.Debug("Contract(s) have no price data, waiting")')
    lines.append(f"                self._open_fail_cooldown = 3")
    lines.append(f"                return")
    lines.append("")
    # Enforce minimum $5 strike separation between same-side legs.
    # The proxy enforces this via _enforce_strike_separation; without it,
    # two legs can land on the same strike, collapsing the spread to zero.
    if len(legs) >= 2:
        for i in range(len(legs) - 1):
            if legs[i].option_type == legs[i + 1].option_type:
                lines.append(f"            if abs(c{i}.Strike - c{i+1}.Strike) < 5:")
                lines.append(f'                self.Debug("Strike separation < $5 between legs {i}/{i+1}, skipping")')
                lines.append(f"                self._open_fail_cooldown = 3")
                lines.append(f"                return")
        lines.append("")
    # Spread width cap: $40 max (calibrated from 104 QC v5 trades).
    if len(legs) >= 2:
        width_pairs = [(f"c{i}", f"c{i+1}") for i in range(len(legs) - 1)]
        width_expr = " or ".join(f"abs({a}.Strike - {b}.Strike) > 40" for a, b in width_pairs)
        lines.append(f"            if {width_expr}:")
        lines.append(f'                self.Debug("Spread width > $40 cap, skipping")')
        lines.append(f"                self._open_fail_cooldown = 3")
        lines.append(f"                return")
        lines.append("")

    # Place orders
    for i, leg in enumerate(legs):
        qty = f"n_contracts * {leg.ratio}" if leg.ratio > 1 else "n_contracts"
        sign = "" if leg.qty_sign > 0 else "-"
        lines.append(
            f"            self.MarketOrder(c{i}.Symbol, {sign}{qty})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_qc_algorithm(
    strategy_id: str,
    template_name: str,
    entry_sexpr: str,
    exit_sexpr: str,
    size_sexpr: str,
    start_date: str = "2025-04-08",
    end_date: str = "",  # empty = today's date
    base_contracts: int = 1,
    delta_value: "Optional[float]" = None,
    delta_sexpr: "Optional[str]" = None,
    regime_guard_iv: "Optional[float]" = None,
    norm_stats: "Optional[Dict]" = None,
    diagnostic_log: bool = False,
    stop_mult: "Optional[float]" = None,
) -> str:
    """Generate a complete QCAlgorithm Python file from GP strategy data.

    Args:
        strategy_id: Unique identifier (used in class name)
        template_name: One of the template names (e.g., "iron_condor_standard")
        entry_sexpr: Entry tree s-expression
        exit_sexpr: Exit tree s-expression
        size_sexpr: Size tree s-expression
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        base_contracts: Base number of contracts before size multiplier
        delta_value: Level B: STATIC GP-discovered delta [0,1], mapped to
            actual short_delta via template's delta_range. None = V1 fixed legs.
            Ignored if delta_sexpr is provided (dynamic takes precedence).
        delta_sexpr: Level B: DYNAMIC delta tree s-expression. Evaluated at
            runtime in QC to compute strike deltas per-trade. Takes precedence
            over delta_value. The tree output is sigmoid-clamped to [0,1] then
            mapped through the template's delta_range.
        stop_mult: The individual's EVOLVED stop-loss base credit-multiple (the
            GP stop gene, gp_engine.Individual.stop_mult). PARITY-CRITICAL: this
            is the base the strategy was evolved under, so the emitted QC stop
            MUST use it (× the execution discount) rather than the proxy default.
            0.0 ⇒ no stop / hold to expiry (the emitted `_stop_loss_multiple` is
            0, and the `if _stop_loss_multiple > 0` guard disables the stop
            entirely — exactly the proxy's behaviour). None ⇒ legacy fallback to
            the proxy's signature default (pre-gene strategies).

    Returns:
        Complete Python source code string for a QCAlgorithm.
    """
    # Default end_date to today
    if not end_date:
        from datetime import date
        end_date = date.today().isoformat()

    # Level B leg resolution:
    # Priority: delta_sexpr (dynamic runtime) > delta_value (static) > template default
    _legs_override = None
    _dynamic_delta = delta_sexpr is not None and template_name in BASE_TEMPLATE_DELTA_RANGES
    if not _dynamic_delta and delta_value is not None and template_name in BASE_TEMPLATE_DELTA_RANGES:
        _legs_override = _build_dynamic_legs(template_name, delta_value)
    elif template_name not in TEMPLATE_LEGS and not _dynamic_delta:
        raise ValueError(
            f"Unknown template {template_name!r}. "
            f"Known: {sorted(TEMPLATE_LEGS.keys())}"
        )

    # Parse trees
    entry_node = from_sexpr(entry_sexpr)
    exit_node = from_sexpr(exit_sexpr)
    size_node = from_sexpr(size_sexpr)

    # Convert to Python expressions
    entry_expr = _node_to_python(entry_node)
    exit_expr = _node_to_python(exit_node)
    size_expr = _node_to_python(size_node)

    # Delta tree: parse and convert if provided, else use constant 0.5
    if delta_sexpr:
        delta_node = from_sexpr(delta_sexpr)
        delta_expr = _node_to_python(delta_node)
    else:
        delta_expr = "0.5"

    # Optional terminal-value logging for proxy<->QC (G11) calibration. Emits a
    # compact `QCDIAG` line once per day carrying the NORMALIZED values of the full
    # terminal dict `s`, so one diagnostic run localises which terminal VALUE
    # diverges live in QC. Compare against layer2.prepare_terminal_data() for the
    # matching minute. ObjectStore export is blocked, so this goes to the log.
    if diagnostic_log:
        # G11 recalibration (2026-06-01): dump the FULL normalized terminal dict `s`
        # (not just this strategy's tree terminals) so ONE diagnostic run yields every
        # _TERMINAL_NOISE_SIGMA terminal for the proxy↔QC MAE/offset re-fit. Logged
        # once per day at the ~10:00 ET bar (MinutesToClose≈375) → ~20 compact lines for
        # a 1-month backtest, easy to copy/paste (ObjectStore export is blocked).
        diag_terminal_block = "\n".join([
            "            # --- DIAGNOSTIC terminal logging (proxy<->QC G11 calibration) ---",
            "            # (round-2 a) SESSION-OPEN GAP TIMING: log SPX vs prev close at early",
            "            # bars on days 2-3, to see WHEN QC's index absorbs the overnight gap",
            "            # (the official open isn't available intraday; this decides the anchor).",
            "            _bod_diag = (self.Time.hour * 60 + self.Time.minute) - (9 * 60 + 30)",
            "            if getattr(self, '_day_count', 0) in (2, 3) and _bod_diag in (1, 2, 4, 6, 8, 10, 30):",
            "                _pc_diag = getattr(self, '_prev_session_close', None) or 0.0",
            "                self.Log(f\"QCGAP {self.Time:%Y-%m-%d} bod={_bod_diag} spx={spx_px:.2f} pc={_pc_diag:.2f}\")",
            "            # (round-2 b) Full NORMALIZED terminal dict once/day at the first bar",
            "            # at/after 10:00 ET (date-latched). Includes _raw_RawSpread (QC's raw",
            "            # ATM spread) for the spread-baseline analysis; the refit skips it.",
            "            if self.Time.hour >= 10 and getattr(self, '_qcdiag_day', None) != self.Time.date():",
            "                self._qcdiag_day = self.Time.date()",
            "                _diag_vals = ' '.join(f'{_k}={float(_v):.5f}' for _k, _v in sorted(s.items()) if (not _k.startswith('_') or _k == '_raw_RawSpread'))",
            "                self.Log(f\"QCDIAG {self.Time:%Y-%m-%d %H:%M} {_diag_vals}\")",
        ])
    else:
        diag_terminal_block = ""

    # Stop-loss base — PARITY with the proxy's credit stop-loss block
    # (evaluator_vectorized.py). Serving a different stop base than the strategy
    # was evolved under is training-serving skew (Sculley et al., 2015). Read the
    # proxy's discount constant as the single source of truth (avoids drift).
    from layer2.evaluator_vectorized import (
        STOP_LOSS_EXECUTION_DISCOUNT as _SL_DISCOUNT,
        vectorized_backtest as _proxy_backtest,
    )
    import inspect as _inspect
    if stop_mult is not None:
        # GP STOP GENE (Phase 1): the evolved stop_mult is the base in BOTH the
        # V1 and the Level-B paths (the proxy's former delta-dependent base was
        # REPLACED by the gene — see evaluator_vectorized.py). Emit the evolved
        # value × discount. 0.0 ⇒ `_stop_loss_multiple = 0.0` ⇒ the `> 0` guard
        # in the emitted body disables the stop = hold-to-expiry, matching the
        # proxy's `stop_loss_credit_multiple > 0` gate exactly.
        stop_base_expr = f"{float(stop_mult)} * {_SL_DISCOUNT}"
    else:
        # Legacy fallback (pre-gene strategies, no stop_mult on the record):
        # mirror the proxy's signature default, branching V1 vs Level-B as before.
        _proxy_stop_mult = _inspect.signature(_proxy_backtest).parameters[
            "stop_loss_credit_multiple"
        ].default
        if _dynamic_delta:
            stop_base_expr = f"(1.5 + self._entry_short_delta * 2.0) * {_SL_DISCOUNT}"
        else:
            stop_base_expr = f"{_proxy_stop_mult} * {_SL_DISCOUNT}"

    # Parse dates
    sy, sm, sd = [int(x) for x in start_date.split("-")]
    ey, em, ed = [int(x) for x in end_date.split("-")]

    # Safe class name (replace non-alphanumeric)
    safe_id = "".join(c if c.isalnum() else "_" for c in strategy_id)

    # Truncate s-expressions for header comment
    entry_short = entry_sexpr[:80] + ("..." if len(entry_sexpr) > 80 else "")
    exit_short = exit_sexpr[:80] + ("..." if len(exit_sexpr) > 80 else "")

    # L2 grammar fix: bake normalization constants as Python dict literal
    # P0-6: normalization must be IDENTICAL at train (proxy) and serve (QC). The
    # proxy evaluates/selects strategies under PER-FOLD MINUTE stats
    # (experiment.py compute_norm_stats_from_data on train_data); embedding the
    # frozen daily-parquet TERMINAL_NORM_STATS here normalizes the SAME raw state
    # differently in QC (e.g. MinutesToClose=200 -> proxy ~-0.01 vs frozen +0.34),
    # firing entry/exit on different bars = training-serving skew (Sculley et al.,
    # 2015). When the caller passes the fold stats the strategy was evolved under,
    # use them; otherwise fall back to the frozen constants. Accepts (center,scale)
    # or (center,scale,method) tuples.
    _stats = {k: (v[0], v[1]) for k, v in TERMINAL_NORM_STATS.items()}
    if norm_stats:
        for name, vals in norm_stats.items():
            _stats[name] = (vals[0], vals[1])
    norm_stats_literal = "{\n"
    for name, (center, scale) in _stats.items():
        norm_stats_literal += f'        "{name}": ({center}, {scale}),\n'
    norm_stats_literal += "    }"

    # Template-specific wing width as fraction of spot for margin computation.
    # For dynamic delta: use the mid-range delta_range to estimate wing width.
    from layer2.templates import template_by_name as _tbn, base_template_by_name as _btbn
    if _dynamic_delta:
        # Dynamic delta: legs are computed at runtime in _open_position.
        # For template format, use the base template's default legs for min_chain_len
        _bt = _btbn(template_name)
        legs = _bt.legs  # need leg count for chain length check
        _is_credit = _bt.is_credit
        # Build legs_tuples from delta_range config (not used for contract finding
        # when dynamic, but needed for template metadata)
        _legs_override_for_meta = _build_dynamic_legs(template_name, 0.5)
        _legs_tuples_repr = repr([(l.option_type, l.delta_target, l.qty_sign, l.ratio)
                                   for l in _legs_override_for_meta])
        _min_chain_len = len(_legs_override_for_meta)
        _open_pos_code = _gen_dynamic_open_position_code(template_name)
    else:
        legs = _legs_override if _legs_override is not None else TEMPLATE_LEGS[template_name]
        if template_name in BASE_TEMPLATE_DELTA_RANGES:
            _is_credit = _btbn(template_name).is_credit
        else:
            _is_credit = _tbn(template_name).is_credit
        _legs_tuples_repr = repr([(l.option_type, l.delta_target, l.qty_sign, l.ratio) for l in legs])
        _min_chain_len = len(legs)
        _open_pos_code = _gen_open_position_code(template_name, legs_override=_legs_override)

    # T3: single-source the credit-correction table from the proxy at gen time.
    from layer2.evaluator_vectorized import _BASE_CREDIT_FACTORS as _BCF
    _credit_factors_literal = repr({k: float(v) for k, v in _BCF.items()})

    # B-2 (campaign discovery sweep): debit time-decay gate must match the proxy —
    # 240 bars for ratio/3-leg structures (e.g. ratio_put_backspread, which needs to
    # hold an underwater position for the gamma-breakout payoff), 40 otherwise. The
    # proxy uses `240 if (n_legs >= 3 or _has_ratio) else 40`
    # (evaluator_vectorized.py); codegen previously hardcoded 40 → force-closed RPB at
    # bar 40 vs the proxy's 240, breaking RPB transfer. `legs` carries the base
    # template's ratio metadata in both the static and dynamic-delta branches.
    _has_ratio_meta = any(getattr(_l, "ratio", 1) > 1 for _l in legs)
    _debit_underwater_bars = 240 if (len(legs) >= 3 or _has_ratio_meta) else 40

    code = _QC_TEMPLATE.format(
        strategy_id=strategy_id,
        template_name=template_name,
        entry_sexpr_short=entry_short,
        exit_sexpr_short=exit_short,
        safe_id=safe_id,
        start_year=sy, start_month=sm, start_day=sd,
        end_year=ey, end_month=em, end_day=ed,
        base_contracts=base_contracts,
        entry_expr=entry_expr,
        exit_expr=exit_expr,
        size_expr=size_expr,
        delta_expr=delta_expr,
        open_position_code=_open_pos_code,
        min_chain_len=_min_chain_len,
        norm_stats_literal=norm_stats_literal,
        is_credit=_is_credit,
        legs_tuples=_legs_tuples_repr,
        credit_factors_literal=_credit_factors_literal,
        regime_guard_iv_val=regime_guard_iv if regime_guard_iv is not None else 0,
        stop_base_expr=stop_base_expr,
        bs_methods=_bs_qc_method_source(indent="    "),
        diag_terminal_block=diag_terminal_block,
        debit_underwater_bars=_debit_underwater_bars,
    )

    return code


def validate_generated_code(code: str) -> Tuple[bool, Optional[str]]:
    """Check that generated code is syntactically valid Python.

    Returns (True, None) if valid, (False, error_message) if not.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
