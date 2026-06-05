"""STVGP strategy templates — 13 defined-risk multi-leg structures.

Templates are seed grammar trees biasing the GP initial population toward
risk-bounded option combinations. Each template defines:
  - Fixed structural metadata (legs, deltas, signs) — NOT GP-evolvable
  - Three evolvable subtrees: entry condition, exit condition, position size

The 13 templates (all scalar-based, condition-neutral):
  Credit (3 widths x 3 structures + IB narrow + IB standard = 11):
    1. iron_condor_standard      — 4-leg neutral, 25d/10d
    2. iron_condor_narrow        — 4-leg neutral, 35d/15d (closer to ATM)
    3. iron_condor_wide          — 4-leg neutral, 15d/10d (farther from ATM)
    4. iron_butterfly_standard   — 4-leg neutral ATM, 50d/10d
    5. iron_butterfly_narrow     — 4-leg neutral ATM, 50d/20d
    6. bull_put_credit_standard  — 2-leg bullish, 25d/10d
    7. bull_put_credit_narrow    — 2-leg bullish, 35d/15d
    8. bull_put_credit_wide      — 2-leg bullish, 15d/10d
    9. bear_call_credit_standard — 2-leg bearish, 25d/10d
   10. bear_call_credit_narrow   — 2-leg bearish, 35d/15d
   11. bear_call_credit_wide     — 2-leg bearish, 15d/10d
  Debit:
   12. bull_call_debit           — 2-leg bullish, debit
   13. bear_put_debit            — 2-leg bearish, debit

Encoder-augmented _enc twins removed (2026-05-11) for ablation fairness:
the 5-condition ablation requires identical template structures across all
conditions. _enc templates baked condition-specific seeds into templates,
creating an unfair comparison. All conditions now run the same 13 template
structures with condition-specific seed substitution (seed_fraction=0.15).

All defined-risk multi-leg per literature (Coval & Shumway 2001; Sinclair
2013, 2020; Bandi-Fusari-Mykland 2024). NO naked positions, NO single
legs — explicitly excluded per D87 risk-management justification.

Source citations per template are in the docstring; full provenance table
lives in `layer2/templates_provenance.md` (Chapter 4 Appendix A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Tuple

from layer2.grammar import (
    GType, Node, FuncDef, FuncNode, TermDef, TermNode,
    Regime, Side,
    GT, LT, AND, OR,
)
from layer2.terminal_stats import normalize_threshold


# ---------------------------------------------------------------------------
# Leg + Template dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leg:
    """One leg of a multi-leg option strategy.

    `delta_target` is signed by convention:
      - call deltas are positive (e.g., +0.15 for OTM call)
      - put deltas are negative (e.g., -0.15 for OTM put)
    `qty_sign` is +1 for long, -1 for short.
    `ratio` >1 supports broken-wing structures (e.g., short 2x at middle strike).
    """
    option_type: Literal["call", "put"]
    delta_target: float
    qty_sign: int           # +1 long, -1 short
    ratio: int = 1          # quantity multiplier (default 1, ratio spreads use 2+)

    def __post_init__(self):
        # Defensive: enforce delta sign matches option type
        if self.option_type == "call" and self.delta_target < 0:
            raise ValueError(
                f"call leg has negative delta {self.delta_target}; "
                f"call deltas must be positive"
            )
        if self.option_type == "put" and self.delta_target > 0:
            raise ValueError(
                f"put leg has positive delta {self.delta_target}; "
                f"put deltas must be negative"
            )
        if self.qty_sign not in (-1, 1):
            raise ValueError(f"qty_sign must be -1 or +1, got {self.qty_sign}")
        if self.ratio < 1:
            raise ValueError(f"ratio must be >=1, got {self.ratio}")


@dataclass
class Template:
    """A trading strategy template — fixed structural legs + evolvable GP subtrees.

    The OptionsBacktester (multi-leg variant) consumes `legs` to compute
    per-bar P&L. The GP evolves `entry_seed`, `exit_seed`, and `size_seed`
    independently across templates (per-template islands per AI Engineer).

    `is_credit` is structural metadata (documentation of the template's
    canonical premium flow per its primary citation). It is NOT consulted
    by the backtester for P&L sign determination — per-trade direction is
    implicit in `_net_position_value = Σ qty_sign × ratio × leg_value`,
    where `qty_sign` already encodes long/short.
    """
    name: str
    description: str
    direction: Literal["neutral", "bullish", "bearish"]
    legs: Tuple[Leg, ...]
    entry_seed: Node                 # GType.BOOL — when to fire
    exit_seed: Node                  # GType.BOOL — when to close
    size_seed: Node                  # GType.REAL in [0,1] — position size
    is_credit: bool = False          # structural metadata — see class docstring
    citations: Tuple[str, ...] = ()  # primary structural source citations
    # Level B: continuous delta exploration (defaults preserve V1 backward compat)
    delta_range: Optional[Tuple[float, float]] = None  # (min_short_delta, max_short_delta)
    wing_offset: Optional[float] = None                # long_delta = short_delta - wing_offset
    delta_fixed: bool = False                          # True for IB (short always ATM 0.50)
    delta_seed: Optional[Node] = None                  # GType.REAL seed for 4th tree
    # GP stop gene (Phase 1, docs/viable_run_plan_2026_06_03.md): the seeded
    # starting value for the per-individual evolved scalar stop_mult. 0.0 = the
    # measured-best hold-to-expiry policy for defined-risk premium sellers (§1.1).
    # None ⇒ seeded individuals fall back to gp_engine.STOP_MULT_SEED (also 0.0).
    # Parallel to delta_seed; base templates set it to 0.0.
    stop_seed: Optional[float] = None

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def __post_init__(self):
        if self.direction not in ("neutral", "bullish", "bearish"):
            raise ValueError(f"direction must be neutral/bullish/bearish, got {self.direction}")
        if not self.legs:
            raise ValueError(f"template {self.name} has zero legs — multi-leg only per D87")
        if len(self.legs) < 2:
            raise ValueError(
                f"template {self.name} has {len(self.legs)} leg(s); "
                f"single-leg structures excluded per D87 (Coval & Shumway 2001)"
            )
        # Validate seed tree types
        if self.entry_seed.ret_type != GType.BOOL:
            raise ValueError(
                f"template {self.name} entry_seed has ret_type "
                f"{self.entry_seed.ret_type}, expected BOOL"
            )
        if self.exit_seed.ret_type != GType.BOOL:
            raise ValueError(
                f"template {self.name} exit_seed has ret_type "
                f"{self.exit_seed.ret_type}, expected BOOL"
            )
        if self.size_seed.ret_type != GType.REAL:
            raise ValueError(
                f"template {self.name} size_seed has ret_type "
                f"{self.size_seed.ret_type}, expected REAL"
            )
        if self.delta_seed is not None and self.delta_seed.ret_type != GType.REAL:
            raise ValueError(
                f"template {self.name} delta_seed has ret_type "
                f"{self.delta_seed.ret_type}, expected REAL"
            )


# ---------------------------------------------------------------------------
# Seed-tree helpers
# ---------------------------------------------------------------------------

def _real_term(name: str, value=None) -> TermNode:
    """Build a REAL-typed terminal node (named lookup or literal)."""
    return TermNode(defn=TermDef(name, GType.REAL), value=value)


def _ephemeral_real(value: float) -> TermNode:
    """Build a fixed REAL constant — GP-mutable starting value."""
    return TermNode(defn=TermDef("EphReal", GType.REAL), value=value)


def _gt(lhs: Node, rhs: Node) -> FuncNode:
    return FuncNode(defn=GT, children=[lhs, rhs])


def _lt(lhs: Node, rhs: Node) -> FuncNode:
    return FuncNode(defn=LT, children=[lhs, rhs])


def _and(a: Node, b: Node) -> FuncNode:
    return FuncNode(defn=AND, children=[a, b])


def _or(a: Node, b: Node) -> FuncNode:
    return FuncNode(defn=OR, children=[a, b])


# Common seed trees — vol-regime entries, time-of-day exits, fixed sizing.
# These are STARTING POINTS that the GP mutates; not the final discovered rules.

def _entry_iv_above(threshold: float) -> FuncNode:
    """Enter when ATM_IV > threshold. Reasonable for credit-receiving structures.

    L2 grammar fix: threshold is converted to normalized space so that
    EphReal[-1,1] compares meaningfully with normalized ATM_IV values.
    e.g., ATM_IV > 0.18 raw becomes ATM_IV_norm > normalize("ATM_IV", 0.18).
    """
    return _gt(_real_term("ATM_IV"),
               _ephemeral_real(normalize_threshold("ATM_IV", threshold)))


def _entry_iv_below(threshold: float) -> FuncNode:
    """Enter when ATM_IV < threshold. Reasonable for debit-paying structures.

    L2 grammar fix: threshold converted to normalized space.
    """
    return _lt(_real_term("ATM_IV"),
               _ephemeral_real(normalize_threshold("ATM_IV", threshold)))


def _exit_close_to_expiry(minutes: float) -> FuncNode:
    """Exit when MinutesToClose < threshold. Time-decay-aware exit for 0DTE.

    L2 grammar fix: threshold converted to normalized space.
    """
    return _lt(_real_term("MinutesToClose"),
               _ephemeral_real(normalize_threshold("MinutesToClose", minutes)))


def _size_constant(value: float) -> TermNode:
    """Constant position size. GP can evolve to a complex size_tree."""
    return _ephemeral_real(value)


# ITEM #5 — VRP-anchored compound entry seed (docs/viable_run_plan_2026_06_03.md).
# Sell premium only when IV is HIGH and genuinely RICH vs realized vol:
# AND(GT(ATM_IV, θ1), GT(IVRVGap5d, θ2))
# This compound shape is the only one that captures the measured edge (+1.16
# frictionless, +1.02 with real fees once the entry double-count is removed; 114
# trades, 75% win on fold-1 train). Single thresholds miss it entirely (≤0).
#
# Both terminals are SCALAR and present in EVERY condition (A scalar-only / B
# probes-only / C emb-only / D real-l1), so the seed is condition-agnostic — a
# fair ablation. The static θ defaults below are FALLBACKS in NORMALIZED space:
# the +1.16 probe used ~1σ ATM_IV (top ~third of IV-days; train ≈ 78th pctile)
# and ~0.5σ IVRVGap5d (premium-rich days; train ≈ 70th pctile). When the run sets
# per_fold_seeds=True these are re-derived per fold on TRAIN data only (the 75th
# percentile of the per-fold robust-normalized terminal; see
# experiment.derive_per_fold_entry_seed) so there is no full-dataset leak.
#
# The thresholds are stated DIRECTLY in normalized space (not via
# normalize_threshold) because they are σ-units, not raw IV/gap levels — the
# probe and the per-fold derivation both operate in normalized space.
_VRP_IV_THRESHOLD_NORM: float = 1.0    # ~1σ ATM_IV (high-IV days), normalized
_VRP_GAP_THRESHOLD_NORM: float = 0.5   # ~0.5σ IVRVGap5d (premium-rich), normalized


def _vrp_compound_entry(
    iv_threshold_norm: float = _VRP_IV_THRESHOLD_NORM,
    gap_threshold_norm: float = _VRP_GAP_THRESHOLD_NORM,
) -> FuncNode:
    """Compound VRP entry: AND(GT(ATM_IV, θ1), GT(IVRVGap5d, θ2)) in norm space.

    BLOCKER fix (2026-06-03 code review): θ1/θ2 are authored DIRECTLY in (fold-)
    normalized space — NOT in frozen TERMINAL_NORM_STATS space like the legacy
    seeds. So they must NOT pass through `recalibrate_seed_thresholds` (frozen→raw→
    fold), which would double-transform them (measured: ATM_IV θ 0.81→2.36, gutting
    selectivity from ~25% to ~7.6%). Tag the EphReal nodes `fold_normalized=True` so
    recalibration skips exactly these (legacy frozen seeds like RPB still recalibrate).
    In-memory instance attr on the non-frozen TermNode; recalibration runs on the
    in-memory tree before serialization, and deepcopy preserves it for seeded copies."""
    _iv_eph = _ephemeral_real(iv_threshold_norm); _iv_eph.fold_normalized = True
    _gap_eph = _ephemeral_real(gap_threshold_norm); _gap_eph.fold_normalized = True
    return _and(
        _gt(_real_term("ATM_IV"), _iv_eph),
        _gt(_real_term("IVRVGap5d"), _gap_eph),
    )


def _exit_hold_to_expiry() -> FuncNode:
    """Hold-to-expiry exit seed (ITEM #5): a comparison that NEVER fires after
    warmup, so the position is held until the evaluator's structural max-hold /
    expiry close. Measured-best for defined-risk premium sellers (§1.1) — the
    intraday 2.5x stop converted recoverable 0DTE drawdowns into locked losses.

    `LT(MinutesToClose, EphReal(-5.0))` is the codebase-canonical never-fire form:
    robust-normalized MinutesToClose ranges [-1.349, +1.349] (MEASURED on fold-1
    train), so it is never < -5.0. The GP can still EVOLVE a real exit from here;
    this is only the seed. Pairs with stop_seed=0.0 (no intraday stop)."""
    return _lt(_real_term("MinutesToClose"), _ephemeral_real(-5.0))


# ---------------------------------------------------------------------------
# Template factory functions — one per template (13 total)
# ---------------------------------------------------------------------------

def iron_condor_standard() -> Template:
    """4-leg neutral defined-risk credit structure (standard width).

    Legs (delta-targeted):
      Short call delta +0.25 / Long call delta +0.10 (call wing)
      Short put delta -0.25 / Long put delta -0.10 (put wing)

    Widened from 25d/5d to 25d/10d (2026-05-11 evaluator V2 spec §2.7):
    5-delta wings degenerate to near-zero value late-day (gamma/vega
    negligible at 5Δ with τ<2h). 10-delta wings retain meaningful
    protective value while keeping risk/reward ratio viable (~60-65%
    break-even win rate).

    Source: Hull (2018) Ch. 12 §12.4; Natenberg (2014) Ch. 11; McMillan (2012) Ch. 13.
    Seed entry: enter when IV > 0.16 (lower threshold — wider wings capture
    more premium at moderate IV).
    Seed exit: close 30 min before expiry.
    """
    return Template(
        name="iron_condor_standard",
        description="4-leg neutral credit spread; profits on range-bound SPX",
        direction="neutral",
        legs=(
            Leg("call", +0.25, qty_sign=-1),  # short OTM call (closer to ATM)
            Leg("call", +0.10, qty_sign=+1),  # long further-OTM call (wing)
            Leg("put",  -0.25, qty_sign=-1),  # short OTM put (closer to ATM)
            Leg("put",  -0.10, qty_sign=+1),  # long further-OTM put (wing)
        ),
        # Range-bound entry: IV high enough for credit + not trending
        entry_seed=_and(
            _entry_iv_above(0.14),
            _and(
                _lt(_real_term("SessionReturn"),
                    _ephemeral_real(normalize_threshold("SessionReturn", 0.005))),
                _gt(_real_term("SessionReturn"),
                    _ephemeral_real(normalize_threshold("SessionReturn", -0.005))),
            ),
        ),
        # Breakout-aware exit: time OR genuine adverse move (>1.5%)
        exit_seed=_or(
            _exit_close_to_expiry(30.0),
            _gt(_real_term("SessionReturn"),
                _ephemeral_real(normalize_threshold("SessionReturn", 0.015))),
        ),
        size_seed=_size_constant(0.5),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12 §12.4",
            "Natenberg (2014) Ch. 11",
            "McMillan (2012) Ch. 13",
        ),
    )


# Backward-compatible alias
iron_condor = iron_condor_standard


def iron_butterfly_standard() -> Template:
    """4-leg neutral defined-risk credit structure, ATM-centered (standard width).

    Legs:
      Short call delta +0.50 (ATM) / Long call delta +0.10 (wing)
      Short put delta -0.50 (ATM) / Long put delta -0.10 (wing)

    Higher credit than iron condor but narrower profit zone — profits when SPX
    pins at ATM at expiry. Pin risk near round strikes is documented for 0DTE
    (Avellaneda & Lipkin 2003).

    Source: Hull (2018) Ch. 12; Natenberg (2014) Ch. 11.
    Seed entry: IV > 0.12 AND RealizedVol30m < 0.015 (high implied + low realized
    = vol premium mispricing, ideal for selling ATM straddle). Widened from 0.20
    to 0.12 (2026-05-09 audit: only 24% of days qualified at 0.20, producing 11
    trades in 507 days — too few for GP to discover patterns).
    Seed exit: 20 min to close OR vol spike (RealizedVol30m crosses above threshold).
    """
    return Template(
        name="iron_butterfly_standard",
        description="4-leg neutral credit; ATM-centered, profits on SPX pin at ATM",
        direction="neutral",
        legs=(
            Leg("call", +0.50, qty_sign=-1),  # short ATM call
            Leg("call", +0.10, qty_sign=+1),  # long OTM call wing
            Leg("put",  -0.50, qty_sign=-1),  # short ATM put
            Leg("put",  -0.10, qty_sign=+1),  # long OTM put wing
        ),
        # High implied + low realized = vol premium mispricing
        entry_seed=_and(
            _entry_iv_above(0.12),
            _lt(_real_term("RealizedVol30m"),
                _ephemeral_real(normalize_threshold("RealizedVol30m", 0.015))),
        ),
        # Time exit OR vol spike (pinning breaks on realized vol jump)
        exit_seed=_or(
            _exit_close_to_expiry(20.0),
            _gt(_real_term("RealizedVol30m"),
                _ephemeral_real(normalize_threshold("RealizedVol30m", 0.025))),
        ),
        size_seed=_size_constant(0.4),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "Natenberg (2014) Ch. 11",
            "Avellaneda & Lipkin (2003) — pinning dynamics",
        ),
    )


# Backward-compatible alias
iron_butterfly = iron_butterfly_standard


def bull_put_credit_standard() -> Template:
    """2-leg bullish defined-risk credit spread (standard width).

    Legs:
      Short put delta -0.25 (OTM put — receives credit)
      Long put delta -0.10 (further-OTM put — defines max loss)

    Widened from 25d/5d to 25d/10d (2026-05-11 evaluator V2 spec §2.7):
    5-delta wings degenerate late-day. 10-delta retains protective value.

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    Seed entry: IV > 0.12 AND not in CRISIS regime.
    Seed exit: 30 min to close.
    """
    return Template(
        name="bull_put_credit_standard",
        description="2-leg bullish credit spread; profits when SPX > short put strike",
        direction="bullish",
        legs=(
            Leg("put", -0.25, qty_sign=-1),  # short OTM put (closer to ATM)
            Leg("put", -0.10, qty_sign=+1),  # long further-OTM put (wing)
        ),
        entry_seed=_and(
            _entry_iv_above(0.12),
            _lt(_real_term("PredRV15"),
                _ephemeral_real(normalize_threshold("PredRV15", 0.020))),
        ),
        exit_seed=_exit_close_to_expiry(30.0),
        size_seed=_size_constant(0.5),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


# Backward-compatible alias
bull_put_credit = bull_put_credit_standard


def bear_call_credit_standard() -> Template:
    """2-leg bearish defined-risk credit spread (standard width).

    Legs:
      Short call delta +0.25
      Long call delta +0.10

    Widened to 25d/10d matching bull_put_credit (2026-05-11 evaluator V2 spec §2.7).
    Directional seed: also requires SessionReturn > 0 (selling calls after
    a rally, when call premium is elevated and mean-reversion is more likely).

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    """
    return Template(
        name="bear_call_credit_standard",
        description="2-leg bearish credit spread; profits when SPX < short call strike",
        direction="bearish",
        legs=(
            Leg("call", +0.25, qty_sign=-1),
            Leg("call", +0.10, qty_sign=+1),  # wing raised 5d→10d
        ),
        entry_seed=_and(
            _and(
                _entry_iv_above(0.12),
                _lt(_real_term("PredRV15"),
                    _ephemeral_real(normalize_threshold("PredRV15", 0.020))),
            ),
            # M1 fix: directional hint — sell calls after a rally
            _gt(_real_term("SessionReturn"),
                _ephemeral_real(normalize_threshold("SessionReturn", 0.001))),
        ),
        exit_seed=_exit_close_to_expiry(30.0),
        size_seed=_size_constant(0.5),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


# Backward-compatible alias
bear_call_credit = bear_call_credit_standard


def bull_call_debit() -> Template:
    """2-leg bullish defined-risk debit spread.

    Legs:
      Long call delta +0.50 (ATM, paying for vega)
      Short call delta +0.20 (OTM, capping max profit + reducing cost)

    Profits when SPX rises past long_strike + debit. Debit spreads are
    directional bets — enter on momentum in the direction of the bet.

    Source: Hull (2018) Ch. 12; Natenberg (2014) Ch. 11.
    Seed entry: SessionReturn > 0.1% (bullish momentum).
    """
    return Template(
        name="bull_call_debit",
        description="2-leg bullish debit spread; profits on directional up-move",
        direction="bullish",
        legs=(
            Leg("call", +0.50, qty_sign=+1),  # long ATM call
            Leg("call", +0.20, qty_sign=-1),  # short OTM call (cap)
        ),
        entry_seed=_gt(_real_term("SessionReturn"),
                       _ephemeral_real(normalize_threshold("SessionReturn", 0.001))),
        exit_seed=_exit_close_to_expiry(45.0),
        size_seed=_size_constant(0.4),
        is_credit=False,
        citations=(
            "Hull (2018) Ch. 12",
            "Natenberg (2014) Ch. 11",
        ),
    )


def bear_put_debit() -> Template:
    """2-leg bearish defined-risk debit spread (mirror of bull_call_debit).

    Seed entry: SessionReturn < -0.1% (bearish momentum).
    """
    return Template(
        name="bear_put_debit",
        description="2-leg bearish debit spread; profits on directional down-move",
        direction="bearish",
        legs=(
            Leg("put", -0.50, qty_sign=+1),
            Leg("put", -0.20, qty_sign=-1),
        ),
        entry_seed=_lt(_real_term("SessionReturn"),
                       _ephemeral_real(normalize_threshold("SessionReturn", -0.001))),
        exit_seed=_exit_close_to_expiry(45.0),
        size_seed=_size_constant(0.4),
        is_credit=False,
        citations=(
            "Hull (2018) Ch. 12",
            "Natenberg (2014) Ch. 11",
        ),
    )


# ---------------------------------------------------------------------------
# Narrow variants — closer to ATM = higher credit, higher gamma risk
# ---------------------------------------------------------------------------

def iron_condor_narrow() -> Template:
    """4-leg neutral IC with narrow wings (35d/15d).

    Closer-to-ATM shorts collect more premium but have higher gamma risk.
    Best in moderate-to-high IV, range-bound sessions.

    Source: Hull (2018) Ch. 12 §12.4; Natenberg (2014) Ch. 11; McMillan (2012) Ch. 13.
    """
    return Template(
        name="iron_condor_narrow",
        description="4-leg neutral credit; narrow wings (35d/15d), higher premium",
        direction="neutral",
        legs=(
            Leg("call", +0.35, qty_sign=-1),  # short closer-to-ATM call
            Leg("call", +0.15, qty_sign=+1),  # long wing call
            Leg("put",  -0.35, qty_sign=-1),  # short closer-to-ATM put
            Leg("put",  -0.15, qty_sign=+1),  # long wing put
        ),
        # Tighter range requirement for narrow wings (higher gamma)
        entry_seed=_and(
            _entry_iv_above(0.18),
            _and(
                _lt(_real_term("SessionReturn"),
                    _ephemeral_real(normalize_threshold("SessionReturn", 0.003))),
                _gt(_real_term("SessionReturn"),
                    _ephemeral_real(normalize_threshold("SessionReturn", -0.003))),
            ),
        ),
        # Hold to EOD for max theta
        exit_seed=_exit_close_to_expiry(5.0),
        size_seed=_size_constant(0.4),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12 §12.4",
            "Natenberg (2014) Ch. 11",
            "McMillan (2012) Ch. 13",
        ),
    )


def iron_butterfly_narrow() -> Template:
    """4-leg neutral IB with narrow wings (50d/20d).

    Tighter protective wings than standard IB (50d/10d). Still ATM-centered
    but wing protection starts closer, reducing tail risk.

    Source: Hull (2018) Ch. 12; Natenberg (2014) Ch. 11.
    """
    return Template(
        name="iron_butterfly_narrow",
        description="4-leg neutral credit; ATM shorts, narrow wings (50d/20d)",
        direction="neutral",
        legs=(
            Leg("call", +0.50, qty_sign=-1),  # short ATM call
            Leg("call", +0.20, qty_sign=+1),  # long wing call (closer than standard)
            Leg("put",  -0.50, qty_sign=-1),  # short ATM put
            Leg("put",  -0.20, qty_sign=+1),  # long wing put (closer than standard)
        ),
        # Calm market entry: IV elevated + realized vol low
        entry_seed=_and(
            _entry_iv_above(0.14),
            _lt(_real_term("RealizedVol30m"),
                _ephemeral_real(normalize_threshold("RealizedVol30m", 0.012))),
        ),
        # Hold to EOD for max theta
        exit_seed=_exit_close_to_expiry(5.0),
        size_seed=_size_constant(0.4),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "Natenberg (2014) Ch. 11",
            "Avellaneda & Lipkin (2003) — pinning dynamics",
        ),
    )


def bull_put_credit_narrow() -> Template:
    """2-leg bullish credit spread with narrow wings (35d/15d).

    Closer-to-ATM short put collects more premium. Higher gamma risk
    requires stronger vol conviction.

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    """
    return Template(
        name="bull_put_credit_narrow",
        description="2-leg bullish credit; narrow wings (35d/15d), higher premium",
        direction="bullish",
        legs=(
            Leg("put", -0.35, qty_sign=-1),  # short closer-to-ATM put
            Leg("put", -0.15, qty_sign=+1),  # long wing put
        ),
        entry_seed=_and(
            _entry_iv_above(0.20),
            _lt(_real_term("PredRV15"),
                _ephemeral_real(normalize_threshold("PredRV15", 0.018))),
        ),
        # Hold to EOD for max theta
        exit_seed=_exit_close_to_expiry(5.0),
        size_seed=_size_constant(0.4),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


def bear_call_credit_narrow() -> Template:
    """2-leg bearish credit spread with narrow wings (35d/15d).

    Mirror of bull_put_credit_narrow for bearish directional bias.

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    """
    return Template(
        name="bear_call_credit_narrow",
        description="2-leg bearish credit; narrow wings (35d/15d), higher premium",
        direction="bearish",
        legs=(
            Leg("call", +0.35, qty_sign=-1),  # short closer-to-ATM call
            Leg("call", +0.15, qty_sign=+1),  # long wing call
        ),
        entry_seed=_and(
            _entry_iv_above(0.20),
            _gt(_real_term("SessionReturn"),
                _ephemeral_real(normalize_threshold("SessionReturn", 0.003))),
        ),
        # Hold to EOD for max theta
        exit_seed=_exit_close_to_expiry(5.0),
        size_seed=_size_constant(0.4),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


# ---------------------------------------------------------------------------
# Wide variants — farther from ATM = lower credit, lower gamma risk
# ---------------------------------------------------------------------------

def iron_condor_wide() -> Template:
    """4-leg neutral IC with wide wings (15d/10d).

    Farther-from-ATM shorts collect less premium but have lower gamma risk.
    Viable in calm, low-vol markets where 25d shorts would be too aggressive.

    Source: Hull (2018) Ch. 12 §12.4; Natenberg (2014) Ch. 11; McMillan (2012) Ch. 13.
    """
    return Template(
        name="iron_condor_wide",
        description="4-leg neutral credit; wide wings (15d/10d), lower gamma risk",
        direction="neutral",
        legs=(
            Leg("call", +0.15, qty_sign=-1),  # short farther-OTM call
            Leg("call", +0.10, qty_sign=+1),  # long wing call
            Leg("put",  -0.15, qty_sign=-1),  # short farther-OTM put
            Leg("put",  -0.10, qty_sign=+1),  # long wing put
        ),
        # Calm market entry — low IV threshold, low realized vol
        entry_seed=_and(
            _entry_iv_above(0.10),
            _lt(_real_term("RealizedVol30m"),
                _ephemeral_real(normalize_threshold("RealizedVol30m", 0.012))),
        ),
        # Standard timing for wide wings
        exit_seed=_exit_close_to_expiry(20.0),
        size_seed=_size_constant(0.6),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12 §12.4",
            "Natenberg (2014) Ch. 11",
            "McMillan (2012) Ch. 13",
        ),
    )


def bull_put_credit_wide() -> Template:
    """2-leg bullish credit spread with wide wings (15d/10d).

    Farther-from-ATM short put collects less premium but has very low gamma
    risk. Afternoon-only entry for pure theta harvest.

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    """
    return Template(
        name="bull_put_credit_wide",
        description="2-leg bullish credit; wide wings (15d/10d), afternoon theta harvest",
        direction="bullish",
        legs=(
            Leg("put", -0.15, qty_sign=-1),  # short farther-OTM put
            Leg("put", -0.10, qty_sign=+1),  # long wing put
        ),
        # Afternoon-only entry: MTC < 120 (after ~2pm)
        entry_seed=_and(
            _entry_iv_above(0.10),
            _lt(_real_term("MinutesToClose"),
                _ephemeral_real(normalize_threshold("MinutesToClose", 120))),
        ),
        exit_seed=_exit_close_to_expiry(20.0),
        size_seed=_size_constant(0.6),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


def bear_call_credit_wide() -> Template:
    """2-leg bearish credit spread with wide wings (15d/10d).

    Mirror of bull_put_credit_wide for bearish directional bias.

    Source: Hull (2018) Ch. 12; McMillan (2012) Ch. 9.
    """
    return Template(
        name="bear_call_credit_wide",
        description="2-leg bearish credit; wide wings (15d/10d), lower gamma risk",
        direction="bearish",
        legs=(
            Leg("call", +0.15, qty_sign=-1),  # short farther-OTM call
            Leg("call", +0.10, qty_sign=+1),  # long wing call
        ),
        entry_seed=_and(
            _entry_iv_above(0.10),
            _gt(_real_term("SessionReturn"),
                _ephemeral_real(normalize_threshold("SessionReturn", 0.001))),
        ),
        exit_seed=_exit_close_to_expiry(20.0),
        size_seed=_size_constant(0.6),
        is_credit=True,
        citations=(
            "Hull (2018) Ch. 12",
            "McMillan (2012) Ch. 9",
        ),
    )


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# All templates as a registry. Use all_templates() to instantiate.
# 13 condition-neutral scalar templates (no _enc twins — removed 2026-05-11
# for ablation fairness; see module docstring).
TEMPLATE_FACTORIES: Tuple[Callable[[], Template], ...] = (
    # Credit — standard width
    iron_condor_standard,
    iron_butterfly_standard,
    bull_put_credit_standard,
    bear_call_credit_standard,
    # Credit — narrow width
    iron_condor_narrow,
    iron_butterfly_narrow,
    bull_put_credit_narrow,
    bear_call_credit_narrow,
    # Credit — wide width
    iron_condor_wide,
    bull_put_credit_wide,
    bear_call_credit_wide,
    # Debit
    bull_call_debit,
    bear_put_debit,
)


def all_templates() -> List[Template]:
    """Instantiate the template library.

    Returns a fresh list each call — templates carry seed trees that GP will
    mutate, so callers should not share template instances across populations.
    """
    return [factory() for factory in TEMPLATE_FACTORIES]


# ---------------------------------------------------------------------------
# V1 backup: preserved 13 fixed-delta templates (pre-Level-B)
# ---------------------------------------------------------------------------
V1_BACKUP_TEMPLATE_FACTORIES: Tuple[Callable[[], Template], ...] = (
    iron_condor_standard, iron_butterfly_standard,
    bull_put_credit_standard, bear_call_credit_standard,
    iron_condor_narrow, iron_butterfly_narrow,
    bull_put_credit_narrow, bear_call_credit_narrow,
    iron_condor_wide, bull_put_credit_wide, bear_call_credit_wide,
    bull_call_debit, bear_put_debit,
)


# ---------------------------------------------------------------------------
# Level B: 5 base templates with continuous delta exploration
# ---------------------------------------------------------------------------
# The GP evolves a 4th tree (delta_tree) whose output [0,1] maps to the
# short-leg delta via: short_delta = min_d + delta_output * (max_d - min_d).
# Wing delta = short_delta - wing_offset (floored at 0.05).

def iron_condor_base() -> Template:
    """IC base: delta_tree controls short_delta in [0.15, 0.40].
    Wing offset = 0.15 (long wing at short_delta - 0.15)."""
    return Template(
        name="iron_condor",
        description="4-leg neutral credit; GP-evolved strike width",
        direction="neutral",
        legs=(  # Placeholder deltas — overridden by delta_tree at backtest time
            Leg("call", +0.25, qty_sign=-1),
            Leg("call", +0.10, qty_sign=+1),
            Leg("put",  -0.25, qty_sign=-1),
            Leg("put",  -0.10, qty_sign=+1),
        ),
        # ITEM #5 (docs/viable_run_plan_2026_06_03.md): compound VRP entry —
        # replaces the leaked, all-negative grid-search seed. Sell premium only
        # when IV is high AND genuinely rich vs realized vol. ATM_IV + IVRVGap5d
        # are scalar terminals present in ALL conditions (A/B/C/D) -> condition-
        # agnostic (fair ablation). Static thresholds are the +1.16-frictionless
        # probe's ~1σ IV / ~0.5σ gap (NORMALIZED space) — a FALLBACK for when
        # per_fold_seeds is off; per_fold_seeds re-derives θ1/θ2 per fold on TRAIN.
        entry_seed=_vrp_compound_entry(),
        # ITEM #5: exit = hold-to-expiry (measured-best; the intraday stop realized
        # recoverable 0DTE drawdowns into locked losses). GP can evolve a real exit.
        exit_seed=_exit_hold_to_expiry(),
        size_seed=_size_constant(0.5),
        is_credit=True,
        delta_range=(0.15, 0.40),
        wing_offset=0.15,
        delta_seed=_ephemeral_real(0.4),  # maps to short_delta=0.25 (standard)
        stop_seed=0.0,  # GP stop gene: seed at hold-to-expiry (measured-best, §1.1)
    )


def iron_butterfly_base() -> Template:
    """IB base: short FIXED at ATM (0.50). delta_tree controls wing delta [0.10, 0.30]."""
    return Template(
        name="iron_butterfly",
        description="4-leg neutral credit; ATM short, GP-evolved wing width",
        direction="neutral",
        legs=(
            Leg("call", +0.50, qty_sign=-1),
            Leg("call", +0.10, qty_sign=+1),
            Leg("put",  -0.50, qty_sign=-1),
            Leg("put",  -0.10, qty_sign=+1),
        ),
        # ITEM #5: compound VRP entry (replaces the leaked, all-negative seed).
        # Sell ATM straddle premium only when IV is high AND rich vs realized vol.
        entry_seed=_vrp_compound_entry(),
        # ITEM #5: exit = hold-to-expiry (measured-best). GP can evolve a real exit.
        exit_seed=_exit_hold_to_expiry(),
        size_seed=_size_constant(0.4),
        is_credit=True,
        delta_range=(0.10, 0.30),  # controls WING delta (short is fixed ATM)
        wing_offset=None,          # not applicable — wing delta IS the delta_tree output
        delta_fixed=True,          # short legs fixed at 0.50
        delta_seed=_ephemeral_real(0.5),  # midpoint: wing_delta=0.20 (was 0.0=boundary)
        stop_seed=0.0,  # GP stop gene: seed at hold-to-expiry (measured-best, §1.1)
    )


def bull_put_credit_base() -> Template:
    """BPC base: delta_tree controls short_delta in [0.15, 0.40]."""
    return Template(
        name="bull_put_credit",
        description="2-leg bullish credit; GP-evolved strike width",
        direction="bullish",
        legs=(
            Leg("put", -0.25, qty_sign=-1),
            Leg("put", -0.10, qty_sign=+1),
        ),
        # ITEM #5: compound VRP entry (replaces the leaked, all-negative seed).
        # Sell put-spread premium only when IV is high AND rich vs realized vol —
        # the measured +1.16-frictionless / +1.02-with-fees edge on this template.
        entry_seed=_vrp_compound_entry(),
        # ITEM #5: exit = hold-to-expiry (measured-best). GP can evolve a real exit.
        exit_seed=_exit_hold_to_expiry(),
        size_seed=_size_constant(0.5),
        is_credit=True,
        delta_range=(0.15, 0.40),
        wing_offset=0.15,
        delta_seed=_ephemeral_real(0.4),
        stop_seed=0.0,  # GP stop gene: seed at hold-to-expiry (measured-best, §1.1)
    )


def bear_call_credit_base() -> Template:
    """BCC base: delta_tree controls short_delta in [0.15, 0.40]."""
    return Template(
        name="bear_call_credit",
        description="2-leg bearish credit; GP-evolved strike width",
        direction="bearish",
        legs=(
            Leg("call", +0.25, qty_sign=-1),
            Leg("call", +0.10, qty_sign=+1),
        ),
        # ITEM #5: compound VRP entry (replaces the leaked, all-negative seed).
        # Sell call-spread premium only when IV is high AND rich vs realized vol.
        # Condition-agnostic: ATM_IV + IVRVGap5d are in every condition's terminal set.
        entry_seed=_vrp_compound_entry(),
        # ITEM #5: exit = hold-to-expiry (measured-best). GP can evolve a real exit.
        exit_seed=_exit_hold_to_expiry(),
        size_seed=_size_constant(0.5),
        is_credit=True,
        delta_range=(0.15, 0.40),
        wing_offset=0.15,
        delta_seed=_ephemeral_real(0.4),
        stop_seed=0.0,  # GP stop gene: seed at hold-to-expiry (measured-best, §1.1)
    )


def ratio_put_backspread_base() -> Template:
    """Ratio put backspread: sell 1 near-ATM put, buy 2 OTM puts.

    Debit structure that profits from large downside moves via gamma
    acceleration. The structure profits from breakout moves rather than
    pinning (53% WR on random entry). The 2:1 long/short ratio means net
    long 1 put below the long strike — unlimited downside profit.

    Empirical baseline (random entry, full dataset):
      Ratio put backspread: Sharpe=-0.93, WR=53%, expectancy=-$13

    Max loss = net debit + (short_strike - long_strike). Occurs when
    underlying pins at the long strike at expiry.

    Source: Natenberg (2014) Ch. 11; McMillan (2012) Ch. 14 (ratio spreads);
            Hull (2018) Ch. 12 §12.5.
    """
    return Template(
        name="ratio_put_backspread",
        description="2-leg bearish debit; sell 1 put, buy 2 OTM puts — profits on large down moves",
        direction="bearish",
        legs=(
            Leg("put", -0.40, qty_sign=-1, ratio=1),  # sell 1 near-ATM put
            Leg("put", -0.20, qty_sign=+1, ratio=2),  # buy 2 OTM puts
        ),
        # ITEM #5 NOTE (deliberate divergence): the compound VRP entry —
        # AND(GT(ATM_IV,θ1), GT(IVRVGap5d,θ2)) — is a premium-SELLING signal (enter
        # when options are RICH). RPB is a premium-BUYING DEBIT structure that
        # profits from large down-moves, so a "sell-when-rich" gate INVERTS its
        # thesis (measured: VRP-seeded RPB = -0.52 Sharpe on fold-1 train, the worst
        # of the 5). The VRP seed is therefore applied to the 4 CREDIT templates
        # only; RPB keeps a debit-appropriate entry (high IV + a down-move, where a
        # downside continuation is more likely). The per-fold candidate pool
        # (candidate_entry_seeds) INCLUDES the IV×SR AND-combos, so when
        # per_fold_seeds=True the derivation re-selects RPB's entry from train data
        # — data-driven, no full-dataset leak. See the report for the rationale.
        entry_seed=_and(
            _gt(_real_term("ATM_IV"), _ephemeral_real(0.0)),
            _lt(_real_term("SessionReturn"), _ephemeral_real(0.0)),
        ),
        # Exit: time OR reversal (session recovers > 1%). The structure
        # profits from continued down moves; exit on reversal to lock gains.
        exit_seed=_or(
            _exit_close_to_expiry(30.0),
            _gt(_real_term("SessionReturn"),
                _ephemeral_real(normalize_threshold("SessionReturn", 0.010))),
        ),
        size_seed=_size_constant(0.5),
        is_credit=False,
        delta_range=(0.25, 0.50),  # GP controls short put delta
        wing_offset=0.20,          # long puts 20 delta-points below short
        delta_seed=_ephemeral_real(0.5),
        # GP stop gene: 0.0 = no stop. NB this is a DEBIT structure — the evolved
        # stop_mult only gates the credit branch of vectorized_backtest; the debit
        # max-loss gate (80% of risk basis) governs here regardless. Seeded for
        # consistency so the gene's range/serialization is uniform across templates.
        stop_seed=0.0,
    )


BASE_TEMPLATE_FACTORIES: Tuple[Callable[[], Template], ...] = (
    iron_condor_base,
    iron_butterfly_base,
    bull_put_credit_base,
    bear_call_credit_base,
    ratio_put_backspread_base,
)


# Backward-compatible name aliases for template_by_name lookups.
# Maps old names (without _standard) to canonical names.
_TEMPLATE_NAME_ALIASES: Dict[str, str] = {
    "iron_condor": "iron_condor_standard",
    "iron_butterfly": "iron_butterfly_standard",
    "bull_put_credit": "bull_put_credit_standard",
    "bear_call_credit": "bear_call_credit_standard",
}


def template_by_name(name: str) -> Template:
    """Look up a V1 template by name (e.g., 'iron_condor_standard').

    Supports backward-compatible aliases: old names without '_standard'
    suffix are transparently mapped to their canonical equivalents.
    Raises KeyError if the name (or its alias) is not found.

    For Level B base templates, use base_template_by_name().
    """
    # Resolve backward-compatible aliases
    canonical = _TEMPLATE_NAME_ALIASES.get(name, name)
    for factory in TEMPLATE_FACTORIES:
        t = factory()
        if t.name == canonical:
            return t
    available = ", ".join(f().name for f in TEMPLATE_FACTORIES)
    raise KeyError(f"unknown template {name!r}; available: {available}")


def base_template_by_name(name: str) -> Template:
    """Look up a Level B base template by name (e.g., 'iron_condor').

    Raises KeyError if not found in BASE_TEMPLATE_FACTORIES.
    """
    for factory in BASE_TEMPLATE_FACTORIES:
        t = factory()
        if t.name == name:
            return t
    available = ", ".join(f().name for f in BASE_TEMPLATE_FACTORIES)
    raise KeyError(f"unknown base template {name!r}; available: {available}")
