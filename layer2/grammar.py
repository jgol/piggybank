"""STVGP Grammar: type system, function/terminal definitions, and tree operations.

Design decisions: L2.6 (Multiply, SafeDivide as analytic quotient, Sqrt),
L2.7 (~45 terminals), Montana 1995 type-safe crossover, Koza 1992 ramped
half-and-half.  No external GP library dependency.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Type system
# ---------------------------------------------------------------------------

class GType(Enum):
    REAL = auto()    # continuous scalar
    BOOL = auto()    # boolean signal
    REGIME = auto()  # categorical (4 regimes)
    SIDE = auto()    # trade direction
    INT = auto()     # lag parameter (fixed set, not freely evolved)
    # typed vector types — replaces retired PCA-compressed embedding scalars.
    # Each is an opaque 128-d (or multi-layer D*L-d) vector from per-group mean-pooling
    # of the iTransformer output. The GP never sees individual coordinates; it
    # composes vectors via typed operators (EmbNorm, EmbCos, EmbSub, EmbLag).
    EMB_SHARED   = auto()  # global mean over all variates
    EMB_GRID     = auto()  # options moneyness grid (raw variates 0-87)
    EMB_STRIKE   = auto()  # strike-level aggregates (99-104)
    EMB_SPX      = auto()  # SPX-derived features (105-118)
    EMB_VIX      = auto()  # VIX term structure (119-130)
    EMB_FLOW_RAW = auto()  # raw order flow (131-139)
    EMB_FLOW_AGG = auto()  # multi-scale flow aggregates (141-158, v3 corpus only)
    # R2 per-variate embeddings — bypass mean-pool bottleneck for 3 key variates.
    # Each is the RAW encoder output for a single variate position (not group-pooled).
    EMB_VAR_ATM_IV     = auto()  # v41: ATM call IV (dominant options pricing input)
    EMB_VAR_ATM_SPREAD = auto()  # v46: ATM bid-ask spread (transaction cost signal)
    EMB_VAR_SPX_RET    = auto()  # v105: SPX session log return (directional signal)

# List of typed-vector GTypes (useful for iterating in operator construction
# and for the typed-vector terminal set)
EMB_TYPES: Tuple["GType", ...] = (
    GType.EMB_SHARED, GType.EMB_GRID, GType.EMB_STRIKE, GType.EMB_SPX,
    GType.EMB_VIX, GType.EMB_FLOW_RAW, GType.EMB_FLOW_AGG,
    # R2 per-variate types (bypass mean-pool bottleneck)
    GType.EMB_VAR_ATM_IV, GType.EMB_VAR_ATM_SPREAD, GType.EMB_VAR_SPX_RET,
)

# Map typed-vector GType → raw variate range (for batch_forecast pool construction)
EMB_TYPE_TO_RAW_RANGE: Dict["GType", range] = {
    GType.EMB_GRID:     range(0, 88),
    GType.EMB_STRIKE:   range(99, 105),
    GType.EMB_SPX:      range(105, 119),
    GType.EMB_VIX:      range(119, 131),
    GType.EMB_FLOW_RAW: range(131, 140),
    GType.EMB_FLOW_AGG: range(141, 159),
    # EMB_SHARED uses all variates — not a range, handled specially
}

class Regime(IntEnum):
    """4-class regime from L1 probe (PredRegime). Verified against data:
    R0: IV=0.079, credit favorable 79%. R1: IV=0.120, credit favorable 77%.
    R2: IV=0.135, IV≈RV, credit favorable only 46% (danger zone).
    R3: IV=0.185, rich premium, credit favorable 67%.
    """
    LOW_VOL = 0            # low IV, calm market
    MID_VOL = 1            # moderate IV
    HIGH_VOL_STRESSED = 2  # IV≈RV, no edge for credit
    HIGH_VOL_PREMIUM = 3   # richest premium, manageable risk

class Side(Enum):
    CALL = 0
    PUT = 1
    NEUTRAL = 2

# ---------------------------------------------------------------------------
# Node definitions (declarative signatures)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuncDef:
    """Signature for an internal (function) node."""
    name: str
    arg_types: Tuple[GType, ...]
    ret_type: GType
    @property
    def arity(self) -> int:
        return len(self.arg_types)

@dataclass(frozen=True)
class TermDef:
    """Signature for a terminal (leaf) node.
    *sampler* produces a fresh value each instantiation (ephemeral constants).
    """
    name: str
    ret_type: GType
    value: Any = None
    sampler: Optional[Callable[[], Any]] = field(default=None, hash=False, compare=False)
    def sample_value(self) -> Any:
        return self.sampler() if self.sampler is not None else self.value

# ---------------------------------------------------------------------------
# Expression tree nodes
# ---------------------------------------------------------------------------

@dataclass
class FuncNode:
    defn: FuncDef
    children: List[Node] = field(default_factory=list)
    @property
    def ret_type(self) -> GType:
        return self.defn.ret_type
    @property
    def name(self) -> str:
        return self.defn.name

@dataclass
class TermNode:
    defn: TermDef
    value: Any = None
    def __post_init__(self):
        if self.value is None:
            self.value = self.defn.sample_value()
    @property
    def ret_type(self) -> GType:
        return self.defn.ret_type
    @property
    def name(self) -> str:
        return self.defn.name

Node = FuncNode | TermNode

# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def depth(node: Node) -> int:
    """Tree depth (max distance from root to a TermNode leaf).

    Fix #2 (2026-04-25): EmbProj_X_k(EMB_X) is treated as depth-1.
    Even though the syntax tree has a function node containing a typed-
    vector terminal, semantically it is a single operation: "project
    typed vector onto k-th PC of group X's PCA basis". The k is hard-
    coded in the operator NAME (not an evolvable INT argument), and the
    operator's only argument MUST be the typed-vector terminal — there
    is no internal structure to grow into. Counting it as depth-2 made
    EmbProj-using trees burn `max_depth` budget twice as fast as
    semantically-equivalent unary operations like EmbNorm (which is
    also depth-1 in syntax-tree convention but represents a single op).
    Treating EmbProj as depth-1 brings depth accounting in line with
    semantic complexity.

    Children of EmbProj that are NOT typed-vector terminals (which
    can't happen by the grammar's typing rules, but we defend) fall
    back to the ordinary depth recursion.
    """
    if isinstance(node, TermNode):
        return 0
    # Fix #2: collapse EmbProj_X_k(EMB_X_term) → depth-1.
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbProj_"):
        if (len(node.children) == 1
                and isinstance(node.children[0], TermNode)):
            return 1
    return 1 + max((depth(c) for c in node.children), default=0)

def node_count(node: Node) -> int:
    """Count of nodes in the tree.

    Fix #2 (2026-04-25): EmbProj_X_k(EMB_X) counts as 1 node (not 2).
    Same rationale as `depth`: EmbProj is semantically a single
    operation, the typed-vector child is an obligatory implicit input
    rather than a separately-evolved subtree. Counting it as 1 node
    aligns `total_nodes` budget accounting with semantic complexity
    so EmbProj-using trees aren't disadvantaged by `max_nodes` caps.
    """
    if isinstance(node, TermNode):
        return 1
    if isinstance(node, FuncNode) and node.defn.name.startswith("EmbProj_"):
        if (len(node.children) == 1
                and isinstance(node.children[0], TermNode)):
            return 1
    return 1 + sum(node_count(c) for c in node.children)

def all_nodes_indexed(
    node: Node, _d: int = 0, _p: Tuple[int, ...] = (),
) -> List[Tuple[Node, int, Tuple[int, ...]]]:
    """Flatten tree into (node, depth, path-from-root) triples."""
    out: List[Tuple[Node, int, Tuple[int, ...]]] = [(node, _d, _p)]
    if isinstance(node, FuncNode):
        for i, ch in enumerate(node.children):
            out.extend(all_nodes_indexed(ch, _d + 1, _p + (i,)))
    return out

def to_str(node: Node) -> str:
    """Human-readable S-expression."""
    if isinstance(node, TermNode):
        v = node.value
        # Check Enum BEFORE int/float — IntEnum subclasses int, so
        # isinstance(Regime.LOW_VOL, int) is True. Without this order,
        # Regime values serialize as "LOW_VOL(0)" instead of "LOW_VOL".
        if isinstance(v, Enum):
            return v.name
        if isinstance(v, (int, float)):
            return f"{node.name}({v})"
        return node.name
    args = ", ".join(to_str(c) for c in node.children)
    return f"{node.name}({args})"


# ---------------------------------------------------------------------------
# Commutative-canonical structural key (diversity-preserving dedup)
# ---------------------------------------------------------------------------
#
# Plain `to_str` is order-sensitive, so semantically identical trees hash to
# distinct strings: AND(a, b) != AND(b, a), GT(x, y) != LT(y, x). Dedup keyed
# on `to_str` therefore lets commutative twins survive as "distinct" Pareto
# front members — a 24-member front collapses to ~8 real skeletons.
#
# `canonical_key` returns a structural signature invariant to:
# (a) argument order of genuinely commutative operators, and
# (b) the GT(x, y) <-> LT(y, x) comparison mirror.
# It canonicalizes children first (bottom-up), then sorts the children of
# commutative ops by their own canonical key.

# Genuinely commutative binary operators — f(a, b) == f(b, a) for all a, b.
# AND/OR : boolean conjunction/disjunction (order-free).
# Add/Mul: real addition/multiplication (order-free).
# RegimeIs(a, b): evaluator computes `a == b` (symmetric equality).
# EmbCos_* (cosine similarity, dot(a,b)/(|a||b|)) is also symmetric; handled
# by name-prefix below since it is a per-group operator family.
# DELIBERATELY EXCLUDED (order matters): Sub, Div (analytic quotient),
# EmbSub_* (vector difference), Lag/Delta/EmbLag_* (2nd arg is an INT
# parameter), CrossAbove/CrossBelow (directional crossing), IfThenElse/IfSide
# (positional branches), NOT/Sqrt (unary). GT/LT are handled separately via
# the comparison-mirror rule, NOT treated as commutative (GT(x,y) != GT(y,x)).
_COMMUTATIVE_OP_NAMES = frozenset({"AND", "OR", "Add", "Mul", "RegimeIs"})


def _is_commutative_func(node: "FuncNode") -> bool:
    """True iff `node` is a genuinely commutative binary operator.

    Covers the fixed set in `_COMMUTATIVE_OP_NAMES` plus the EmbCos_* family
    (cosine similarity is symmetric in its two vector arguments).
    """
    if len(node.children) != 2:
        return False
    name = node.defn.name
    return name in _COMMUTATIVE_OP_NAMES or name.startswith("EmbCos_")


def canonical_key(node: Node, _eph_round: Optional[int] = None) -> str:
    """Canonical structural signature, invariant to commutative arg order.

    Two trees that differ only by (a) the argument order of a commutative
    operator (AND/OR/Add/Mul/RegimeIs/EmbCos_*) or (b) the GT<->LT mirror
    [GT(x, y) == LT(y, x)] produce the *same* key. Non-commutative operators
    (Sub, Div, CrossAbove/Below, Lag, IfThenElse, ...) keep their argument
    order, so e.g. Sub(a, b) and Sub(b, a) stay distinct.

    Args:
        node: the tree (or subtree) to canonicalize.
        _eph_round: if not None, round EphReal terminal values to this many
            decimals before keying (mirrors the phenotypic-dedup tolerance in
            gp_engine so threshold jitter doesn't defeat dedup). EphInt is
            never rounded.

    Returns:
        A deterministic string key. Equal keys <=> commutative-equivalent trees.
    """
    if isinstance(node, TermNode):
        v = node.value
        if isinstance(v, Enum):
            return v.name
        if isinstance(v, float) and _eph_round is not None and node.name == "EphReal":
            return f"{node.name}({round(v, _eph_round)})"
        if isinstance(v, (int, float)):
            return f"{node.name}({v})"
        return node.name

    # FuncNode — canonicalize children first (bottom-up).
    child_keys = [canonical_key(c, _eph_round) for c in node.children]
    name = node.defn.name

    # Comparison mirror: normalize LT(a, b) -> GT(b, a). After this, GT(x, y)
    # and LT(y, x) both render as GT over the same child-key pair, while
    # GT(x, y) and GT(y, x) stay distinct (GT is NOT commutative). Do not sort.
    if name == "LT" and len(child_keys) == 2:
        return f"GT({child_keys[1]}, {child_keys[0]})"

    # Commutative ops: sort child keys so arg order can't create a false twin.
    if _is_commutative_func(node):
        child_keys = sorted(child_keys)

    return f"{name}({', '.join(child_keys)})"


_TIME_ONLY_TERMINALS = frozenset({"MinutesToClose", "BarOfDay", "ThetaUrgency"})
_COMPARISON_OPS = frozenset({"GT", "LT", "CrossAbove", "CrossBelow"})

# ---------------------------------------------------------------------------
# QC-NON-TRANSFERABLE SCALARS (G11 calibration, 2026-06-01) — DOCUMENTED + REVERSIBLE
# ---------------------------------------------------------------------------
# These raw-scalar terminals are EXCLUDED FROM THE GP SAMPLING POOL (Grammar.__init__)
# because their proxy value does NOT predict their QuantConnect-served value — a GP
# champion that gates on them would behave randomly differently live (training-serving
# skew). The parser (_TERM_LOOKUP, from the unfiltered build_terminal_set()) still
# knows them, so historical strategies/tests that reference them parse normally; only
# new random trees avoid them.
#
# Evidence (G11 diagnostic, 20 Jan-2025 days at 10:00 ET, normalized minute space,
# AFTER rounds 1-2 fixed every deterministic bug — RV double-sqrt, RV5d 1-day lag,
# spread spikes, open-price stale source, IVRVGap sampling):
# terminal proxy<->QC corr why it can't transfer
# SessionReturn +0.03 open anchor = official-open vs intraday-index; ~independent
# OvernightGap +0.29 log(open/prev_close): ~3-pt official-vs-intraday capture
# noise on each endpoint ≈ the gap signal itself
# RawSpread +0.34 live-chain ATM spread != History()-collected spread
# RawSpread_5m (~0.34) 5-bar mean of the above
# DeltaSpread1 -0.43 bar-to-bar spread CHANGE = difference of two noisy
# spreads = pure noise (negative corr)
# DeltaSpread5 (~) 5-bar spread change, same
# PutCallSkew +0.87* *misleading: gap is TWO-SIGNED scatter (per-day std
# 1.98x the mean), localized to the illiquid 25Δ wings
# (live quotes vs History()-collected). PROVEN not a
# convention bug — the ATM_IV control (same spline at
# delta 0.50) transfers at corr 0.997 / ~0 offset, so no
# global r/delta/T/spline bug exists. Irreducible
# live-chain-vs-collected-surface provenance at the wings.
# SessionPosition +0.61 MAE 0.59 — running (price-low)/(high-low); the early
# morning reconstruction-vs-index error pollutes the
# day's high/low. Removed for a fully-clean <0.1 grammar
# (the PutCallSkew lesson: corr is an unreliable transfer
# signal, so the 0.59 MAE governs).
#
# TO RE-ENABLE a terminal: remove it from this set. Do so when QC can serve it
# faithfully — e.g. an official-open data feed would make OvernightGap/SessionReturn
# computable, or a serving-side spread fix would restore RawSpread correlation.
# Removal affects conditions that sample raw scalars (A=scalar-only, D=real-l1, and
# the SessionReturn/OvernightGap that B=probes-only also lists); C=emb-only is
# unaffected (no raw scalars). See project_terminal_parity_calibration memory.
_QC_NONTRANSFERABLE_SCALARS = frozenset({
    "SessionReturn", "OvernightGap", "RawSpread", "RawSpread_5m",
    "DeltaSpread1", "DeltaSpread5", "PutCallSkew", "SessionPosition",
})


def _leaf_terminal_names(node: Node) -> set:
    """Collect data terminal names from all leaves (excluding EphReal/EphInt)."""
    if isinstance(node, TermNode):
        return {node.name} if node.name not in ("EphReal", "EphInt") else set()
    return set().union(*(_leaf_terminal_names(c) for c in node.children))


def _is_time_vs_market(node: FuncNode) -> bool:
    """Detect comparisons where one side is time-dominated.

    Catches:
    - Pure: GT(BarOfDay, EphReal(X)) — time vs ephemeral
    - Mixed: GT(BarOfDay, ATM_IV) — time vs market
    - Dominant: GT(Add(BarOfDay, ATM_IV), EphReal(X)) — time is >50%
      of leaf terminals. BarOfDay variance dominates the sum, making
      this a disguised clock. (Red team Exploit 1/5)
    """
    if node.name not in _COMPARISON_OPS or len(node.children) < 2:
        return False
    left = _leaf_terminal_names(node.children[0])
    right = _leaf_terminal_names(node.children[1])
    # Time-only vs ephemeral-only = pure clock
    left_all_time = left.issubset(_TIME_ONLY_TERMINALS) if left else False
    right_all_time = right.issubset(_TIME_ONLY_TERMINALS) if right else False
    left_all_eph = not left
    right_all_eph = not right
    if (left_all_time and right_all_eph) or (right_all_time and left_all_eph):
        return True
    # Time-DOMINANT vs EPHEMERAL: one side has >=50% time terminals AND
    # the other side is ALL ephemeral. This catches disguised clocks like
    # GT(Add(BarOfDay, ATM_IV), EphReal(X)) where BarOfDay dominates.
    # Only flags when the non-time side is ephemeral — if both sides have
    # real market terminals, it's a legitimate mixed comparison.
    for terms, other in [(left, right), (right, left)]:
        if not terms:
            continue
        other_all_eph = not other  # empty set = all ephemerals
        if not other_all_eph:
            continue  # other side has market terminals — legitimate
        time_frac = len(terms & _TIME_ONLY_TERMINALS) / len(terms)
        if time_frac >= 0.5 and len(terms & _TIME_ONLY_TERMINALS) >= 1:
            return True
    if not left or not right:
        return False  # one side all ephemeral, other is market-only — ok
    if left_all_time == right_all_time:
        return False  # both time or both market = ok
    return True  # different categories = degenerate


def _count_ite_nodes(node: Node) -> int:
    """Count IfThenElse nodes in a tree."""
    if not isinstance(node, FuncNode):
        return 0
    count = 1 if node.name == "IfThenElse" else 0
    return count + sum(_count_ite_nodes(c) for c in node.children)


def _is_ephemeral_leaf(node: Node) -> bool:
    """True if node is an ephemeral constant (EphReal or EphInt)."""
    return isinstance(node, TermNode) and node.name in ("EphReal", "EphInt")


def _evaluates_to_constant(node: Node) -> bool:
    """True if the subtree evaluates to a constant (all leaves are ephemerals).

    Catches Sqrt(EphReal(x)), Add(EphReal(x), EphReal(y)), etc. — constant
    expressions wrapped in functions that bypass the ephemeral-leaf check.
    A constant child in CrossAbove/CrossBelow produces an inert crossing
    condition (constants never cross anything).
    """
    if isinstance(node, TermNode):
        return node.name in ("EphReal", "EphInt")
    if isinstance(node, FuncNode):
        return all(_evaluates_to_constant(c) for c in node.children)
    return False


def simplify_tree(node: Node) -> Node:
    """Collapse identity patterns in GP trees (post-crossover/mutation).

    Patterns simplified:
    - NOT(NOT(x)) → x  (identity, wastes 2 nodes)
    - NOT(LT(x, y)) → GT(y, x)  (canonicalize, reduces NOT usage)
    - NOT(GT(x, y)) → LT(y, x)  (canonicalize)

    Applied after every crossover and mutation to prevent NOT accumulation,
    which is the primary structural pathway to tautological entries.
    """
    if isinstance(node, TermNode):
        return node

    # Recurse first (bottom-up simplification)
    simplified_children = [simplify_tree(c) for c in node.children]
    node = FuncNode(defn=node.defn, children=simplified_children)

    if node.name == "NOT" and len(simplified_children) == 1:
        child = simplified_children[0]
        # NOT(NOT(x)) → x
        if isinstance(child, FuncNode) and child.name == "NOT" and len(child.children) == 1:
            return child.children[0]
        # NOT(LT(x, y)) → GT(y, x)
        if isinstance(child, FuncNode) and child.name == "LT" and len(child.children) == 2:
            return FuncNode(defn=GT, children=[child.children[1], child.children[0]])
        # NOT(GT(x, y)) → LT(y, x)
        if isinstance(child, FuncNode) and child.name == "GT" and len(child.children) == 2:
            return FuncNode(defn=LT, children=[child.children[1], child.children[0]])
        # NOT(CrossAbove/Below) is NOT simplified here — it's a semantic
        # inversion (99% fire rate → 1%), not an identity. Caught by
        # is_structurally_degenerate as a root-level check instead.

    return node


def is_structurally_degenerate(node: Node, max_ite: int = 2, _root: bool = True) -> bool:
    """Detect trees that are structurally trivial or economically nonsensical.

    Catches:
    - GT(X, X), LT(X, X), CrossAbove(X, X), CrossBelow(X, X) where both
      children are identical subtrees → always false.
    - GT/LT(EphReal, EphReal) — comparison of two constants is always-true or
      always-false (~50% of random instances). Wastes tree budget.
    - CrossAbove/CrossBelow(EphReal, EphReal) — constants never cross, always false.
    - Time-vs-market comparisons (BarOfDay vs ATM_IV etc.) — no economic meaning.
    - More than max_ite IfThenElse nodes (opacity cap, default 2).
      Raised from 1 to 2 (2026-05-19): with max_nodes=15, 2 ITE nodes
      allow richer conditional logic without excessive opacity.
    """
    if not isinstance(node, FuncNode):
        return False
    if _count_ite_nodes(node) > max_ite:
        return True
    if node.name in _COMPARISON_OPS and len(node.children) >= 2:
        if to_str(node.children[0]) == to_str(node.children[1]):
            return True
        # Time-vs-market: only check at ROOT-level comparisons.
        # Recursive checking (pre-2026-05-19) killed 18% of random trees
        # by poisoning entire trees from any child comparison like
        # GT(BarOfDay, ATM_IV). In 0DTE context, time comparisons inside
        # AND/OR with market conditions are economically valid (e.g.,
        # "enter between 2-3PM AND when IV is elevated"). Also fixes
        # false positives on exit seeds like OR(LT(MinutesToClose, X),
        # GT(SessionReturn, Y)) which are valid time+market fallbacks.
        if _root and _is_time_vs_market(node):
            return True
        # Constant-vs-constant: GT(const, const) is always-true/false,
        # CrossAbove/CrossBelow(const, const) is always-false.
        # Also catches f(const) like Sqrt(EphReal(x)) — same degeneracy.
        if _evaluates_to_constant(node.children[0]) and _evaluates_to_constant(node.children[1]):
            return True
        # Constant-vs-terminal in CrossAbove/CrossBelow: a constant never
        # "crosses" anything — the crossing condition requires the constant
        # to change between bars, which it can't.
        if node.name in ("CrossAbove", "CrossBelow"):
            if _evaluates_to_constant(node.children[0]) or _evaluates_to_constant(node.children[1]):
                return True
    # NOT(CrossAbove/Below) at the ROOT of a tree fires 98-99% of bars.
    # Only check at root — nested NOT(Cross*) inside AND/OR is fine
    # because the outer operator reduces the fire rate.
    if _root and (node.name == "NOT" and len(node.children) == 1
            and isinstance(node.children[0], FuncNode)
            and node.children[0].name in ("CrossAbove", "CrossBelow")):
        return True
    # AND(x, NOT(x)) → always false; OR(x, NOT(x)) → always true
    if node.name in ("AND", "OR") and len(node.children) == 2:
        c0_str, c1_str = to_str(node.children[0]), to_str(node.children[1])
        # Check if one child is NOT(other)
        for a, b in [(node.children[0], node.children[1]),
                      (node.children[1], node.children[0])]:
            if (isinstance(a, FuncNode) and a.name == "NOT"
                    and len(a.children) == 1 and to_str(a.children[0]) == to_str(b)):
                return True
    # Whole-tree time-dominance: if >=80% of all leaf terminals are time-only,
    # the tree is a disguised clock regardless of structure. Catches
    # AND(GT(BarOfDay, EphReal), GT(BarOfDay, EphReal)) which the root-only
    # time_vs_market check misses. (2026-05-19: complement to root-only scoping)
    if _root:
        all_terms = _leaf_terminal_names(node)
        if all_terms:
            time_frac = len(all_terms & _TIME_ONLY_TERMINALS) / len(all_terms)
            if time_frac >= 0.80:
                return True
    # Recurse: any child degenerate makes the whole tree suspect
    # (but time_vs_market is NOT checked recursively — only at root)
    return any(is_structurally_degenerate(c, max_ite=max_ite, _root=False)
               for c in node.children if isinstance(c, FuncNode))


# ---------------------------------------------------------------------------
# S-expression deserialization (JSONL → Node tree)
# ---------------------------------------------------------------------------

def _build_from_sexpr_lookups() -> Tuple[Dict[str, "FuncDef"], Dict[str, "TermDef"]]:
    """Build name → definition lookup tables for from_sexpr().

    Called once at module load. Covers ALL function and terminal definitions
    including dynamically generated EMB_OPERATORS and EmbProj_* operators.
    """
    func_lookup: Dict[str, FuncDef] = {}
    # Standard functions
    for fd in [GT, LT, AND, OR, NOT, ADD, SUB, MUL, DIV, SQRT,
               LAG, DELTA, CROSS_ABOVE, CROSS_BELOW,
               IN_REGIME, REGIME_IS, IF_REAL, IF_SIDE]:
        func_lookup[fd.name] = fd
    # EMB_OPERATORS (EmbNorm_*, EmbCos_*, EmbSub_*, EmbLag_*, EmbProj_*)
    for fd in EMB_OPERATORS:
        func_lookup[fd.name] = fd

    term_lookup: Dict[str, TermDef] = {}
    # Build ALL terminal sets and merge — from_sexpr must handle any condition
    for td in build_terminal_set():
        term_lookup[td.name] = td
    # Regime enum names (bare terminals without parens in to_str)
    for r in Regime:
        term_lookup[r.name] = TermDef(r.name, GType.REGIME, value=r)
    # Side enum names
    for s in Side:
        term_lookup[s.name] = TermDef(s.name, GType.SIDE, value=s)
    # L2 grammar fix: AbstainFlag removed from terminal sets (always 0.0,
    # wasted search budget), but kept in the from_sexpr lookup for backward
    # compatibility — existing GP results may contain trees with AbstainFlag.
    if "AbstainFlag" not in term_lookup:
        term_lookup["AbstainFlag"] = TermDef("AbstainFlag", GType.REAL)
    return func_lookup, term_lookup

_FUNC_LOOKUP: Dict[str, "FuncDef"] = {}
_TERM_LOOKUP: Dict[str, "TermDef"] = {}


def _ensure_lookups() -> None:
    """Lazy-init the lookup tables (avoids circular-import issues at module load)."""
    global _FUNC_LOOKUP, _TERM_LOOKUP
    if not _FUNC_LOOKUP:
        _FUNC_LOOKUP, _TERM_LOOKUP = _build_from_sexpr_lookups()


def _tokenize_sexpr(s: str) -> List[str]:
    """Split s-expression into tokens: names, numbers, '(', ')', ','."""
    tokens: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t\n\r":
            i += 1
        elif c in "(,)":
            tokens.append(c)
            i += 1
        elif c == "-" and i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] == "."):
            # Negative number
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == "." or s[j] == "e" or s[j] == "E"
                                  or (s[j] == "-" and j > 0 and s[j - 1] in "eE")):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "." or s[j] == "e" or s[j] == "E"
                                  or (s[j] == "-" and j > 0 and s[j - 1] in "eE")):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"Unexpected character {c!r} at position {i} in: {s!r}")
    return tokens


def from_sexpr(s: str) -> Node:
    """Parse an s-expression string (as produced by to_str) back into a Node tree.

    Round-trip property: to_str(from_sexpr(s)) == s for all valid s-expressions.

    Examples:
        from_sexpr("ATM_IV") → TermNode(defn=T("ATM_IV", REAL))
        from_sexpr("GT(ATM_IV, EphReal(0.18))") → FuncNode(GT, [TermNode(ATM_IV), TermNode(EphReal, 0.18)])
        from_sexpr("LOW_VOL") → TermNode(defn=T("LOW_VOL", REGIME), value=Regime.LOW_VOL)
    """
    _ensure_lookups()
    tokens = _tokenize_sexpr(s)
    node, pos = _parse_sexpr(tokens, 0)
    if pos != len(tokens):
        raise ValueError(
            f"Unexpected tokens after position {pos}: "
            f"{' '.join(tokens[pos:pos+5])}... in: {s!r}"
        )
    return node


def _parse_sexpr(tokens: List[str], pos: int) -> Tuple[Node, int]:
    """Recursive descent parser. Returns (node, next_position)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    name = tokens[pos]

    # Check if next token is '(' — if so, this is either a function call or
    # a terminal with a value (EphReal(0.18), EphInt(5))
    if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        # Check if this is a known function
        if name in _FUNC_LOOKUP:
            fd = _FUNC_LOOKUP[name]
            pos += 2  # skip name and '('
            children: List[Node] = []
            while pos < len(tokens) and tokens[pos] != ")":
                if tokens[pos] == ",":
                    pos += 1
                    continue
                child, pos = _parse_sexpr(tokens, pos)
                children.append(child)
            if pos >= len(tokens):
                raise ValueError(f"Unbalanced parentheses in s-expression for '{name}'")
            pos += 1  # skip ')'
            return FuncNode(defn=fd, children=children), pos

        # Terminal with value: EphReal(0.18), EphInt(5)
        if name in _TERM_LOOKUP:
            td = _TERM_LOOKUP[name]
            pos += 2  # skip name and '('
            val_str = tokens[pos]
            pos += 1  # skip value
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1  # skip ')'
            # Parse the value
            if td.ret_type == GType.INT:
                value = int(val_str)
            else:
                value = float(val_str)
            return TermNode(defn=td, value=value), pos

        raise ValueError(f"Unknown function/terminal name: {name!r}")

    # Bare terminal (no parens): ATM_IV, LOW_VOL, CALL, etc.
    if name in _TERM_LOOKUP:
        td = _TERM_LOOKUP[name]
        return TermNode(defn=td, value=td.value), pos + 1

    # Try as a number literal (shouldn't happen in to_str output, but defensive)
    try:
        if "." in name:
            return TermNode(defn=_TERM_LOOKUP["EphReal"], value=float(name)), pos + 1
        return TermNode(defn=_TERM_LOOKUP["EphInt"], value=int(name)), pos + 1
    except (ValueError, KeyError):
        pass

    raise ValueError(f"Unknown token: {name!r} at position {pos}")


# ---------------------------------------------------------------------------
# Natural-language serialization (L2 → L3 handoff)
# ---------------------------------------------------------------------------

# Friendly terminal names — make the English readable without losing
# disambiguation. Unlisted terminals fall back to their raw name.
_NL_TERM_NAMES: Dict[str, str] = {
    "PredRV15":         "predicted 15-minute realized volatility",
    "PredRV30":         "predicted 30-minute realized volatility",
    "RegimeAboveLow":   "binary: 1 if regime >= MID_VOL (not low-vol)",
    "RegimeIsHigh":     "binary: 1 if regime >= HIGH_VOL_STRESSED",
    "RegimeIsPremium":  "binary: 1 if regime == HIGH_VOL_PREMIUM",
    "PredSpread":       "predicted ATM spread change",
    "ATM_IV":           "ATM implied volatility",
    "VIXSpot":          "VIX spot",
    "VIXTermSlope":     "VIX term-structure slope",
    "SessionReturn":    "cumulative SPX return since session open",
    "RealizedVol30m":   "trailing 30-minute Parkinson realized volatility",
    "UnrealizedProfitPct": "fraction of the position's max profit currently captured (0 at entry, +1 at max profit, negative when losing; 0 when flat)",
    "RawSpread":        "raw ATM bid-ask spread",
    "DeltaSpread1":     "1-bar ATM spread change",
    "DeltaSpread5":     "5-bar ATM spread change",
    "MinutesToClose":   "minutes until market close",
    "SPXClose":         "SPX price",
    "EMB_SHARED":       "shared L1 embedding",
    "EMB_GRID":         "options-grid L1 embedding",
    "EMB_STRIKE":       "strike-aggregate L1 embedding",
    "EMB_SPX":          "SPX-derived L1 embedding",
    "EMB_VIX":          "VIX-term L1 embedding",
    "EMB_FLOW_RAW":     "raw order-flow L1 embedding",
    "EMB_FLOW_AGG":     "aggregated order-flow L1 embedding",
    "EMB_VAR_ATM_IV":     "ATM-IV per-variate L1 embedding",
    "EMB_VAR_ATM_SPREAD": "ATM-spread per-variate L1 embedding",
    "EMB_VAR_SPX_RET":    "SPX-return per-variate L1 embedding",
}


def _nl_term(node: "TermNode") -> str:
    """Render a terminal node in English."""
    v = node.value
    if isinstance(v, Enum):
        return v.name.lower().replace("_", " ")
    if isinstance(v, (int, float)):
        # Ephemeral constants — show the value.
        if node.name in ("ConstReal", "ConstInt"):
            return f"{v:g}"
        return f"{_NL_TERM_NAMES.get(node.name, node.name)} ({v:g})"
    return _NL_TERM_NAMES.get(node.name, node.name)


_NL_INFIX = {
    "Add": " + ", "Sub": " − ", "Mul": " × ", "Div": " ÷ ",
    "GT":  " > ",  "LT":  " < ",
    "AND": " AND ", "OR":  " OR ",
}
_NL_PREFIX_UNARY = {
    "NOT":  "NOT ",  "Sqrt": "√",
}


def to_nl(node: Node) -> str:
    """Tree → human-readable English for L3 agent consumption.

    Produces a single-line phrase suitable for feeding into the L3 Spec
    agent. For boolean entry trees: "entry fires when X > Y AND Z > 0.5".
    For real expressions: "X + Y × 2". For temporal / typed-vector ops:
    descriptive phrases ("5-bar lag of SPX price", "cosine similarity
    between EMB_GRID and EMB_VIX").

    Infix rendering with outer parens matches the visual precedence a
    reader expects; deterministic, so two identical trees always produce
    the same string (invariant the L3 agent can hash against).
    """
    if isinstance(node, TermNode):
        return _nl_term(node)
    name = node.name
    kids = [to_nl(c) for c in node.children]

    # Infix binary operators
    if name in _NL_INFIX and len(kids) == 2:
        return f"({kids[0]}{_NL_INFIX[name]}{kids[1]})"

    # Unary prefix
    if name in _NL_PREFIX_UNARY and len(kids) == 1:
        return f"({_NL_PREFIX_UNARY[name]}{kids[0]})"

    # Temporal
    if name == "Lag" and len(kids) == 2:
        return f"{kids[1]}-bar lag of {kids[0]}"
    if name == "Delta" and len(kids) == 2:
        return f"{kids[0]} minus its {kids[1]}-bar lag"
    if name == "CrossAbove" and len(kids) == 2:
        return f"({kids[0]} crosses above {kids[1]})"
    if name == "CrossBelow" and len(kids) == 2:
        return f"({kids[0]} crosses below {kids[1]})"

    # Regime
    if name == "InRegime" and len(kids) == 1:
        return f"(current regime is {kids[0]})"
    if name == "RegimeIs" and len(kids) == 2:
        return f"({kids[0]} equals {kids[1]})"

    # Conditional
    if name == "IfThenElse" and len(kids) == 3:
        return f"(if {kids[0]} then {kids[1]} else {kids[2]})"
    if name == "IfSide" and len(kids) == 3:
        return f"(if {kids[0]} choose side {kids[1]} else {kids[2]})"

    # Typed-vector operators (EmbNorm_EMB_GRID, EmbCos_EMB_GRID_EMB_VIX, …)
    if name.startswith("EmbNorm_"):
        return f"L2 norm of {kids[0]}"
    if name.startswith("EmbCos_"):
        return f"cosine similarity between {kids[0]} and {kids[1]}"
    if name.startswith("EmbSub_"):
        return f"({kids[0]} minus {kids[1]})"
    if name.startswith("EmbLag_"):
        return f"{kids[1]}-bar lag of {kids[0]}"
    # v8: EmbProj_EMB_X_k — render as "k-th principal component of {X}".
    # The ordinal helps readers understand the basis index without
    # reference to the pca_bases.npz file.
    if name.startswith("EmbProj_"):
        # Parse suffix "EMB_X_k" → group, k
        suffix = name[len("EmbProj_"):]
        _dot = suffix.rfind("_")
        try:
            k = int(suffix[_dot + 1:])
            group = suffix[:_dot]
            ordinals = ["0th", "1st", "2nd", "3rd", "4th", "5th"]
            k_str = ordinals[k] if 0 <= k < len(ordinals) else f"{k}th"
            return f"{k_str} principal component of {_NL_TERM_NAMES.get(group, group)}"
        except (ValueError, IndexError):
            return f"{name}({kids[0] if kids else ''})"

    # Generic fallback — functional notation
    return f"{name}({', '.join(kids)})"


def to_nl_strategy(entry_tree: Node, exit_tree: Node, size_tree: Node,
                   template_name: str) -> str:
    """Full-strategy NL description combining entry, exit, size + template.

    Returns a multi-line paragraph suitable for the L3 Spec agent prompt.
    """
    return (
        f"Strategy template: {template_name.replace('_', ' ')}.\n"
        f"  ENTRY  when: {to_nl(entry_tree)}\n"
        f"  EXIT   when: {to_nl(exit_tree)}\n"
        f"  SIZE  scalar: {to_nl(size_tree)} "
        f"(multiplier applied to the template's default position size)."
    )

def deep_copy(node: Node) -> Node:
    if isinstance(node, TermNode):
        return TermNode(defn=node.defn, value=node.value)
    return FuncNode(defn=node.defn, children=[deep_copy(c) for c in node.children])

def validate(node: Node) -> bool:
    """Recursively check that every node's children match declared arg_types.

    Also enforces the EmbLag first-arg-is-terminal constraint — trees whose
    EmbLag_* node has a non-TermNode first child are rejected here so the
    evaluator's defensive no-lag fallback (layer2/evaluator.py EmbLag_*)
    can never be silently exercised by a valid tree.
    """
    if isinstance(node, TermNode):
        return True
    if len(node.children) != node.defn.arity:
        return False
    # EmbLag_* constraint: first child must be a TermNode (typed-vector leaf).
    # See _EMBLAG_FUNCDEFS / is_emblag_funcdef.
    if is_emblag_funcdef(node.defn) and not isinstance(node.children[0], TermNode):
        return False
    return all(
        ch.ret_type == exp and validate(ch)
        for ch, exp in zip(node.children, node.defn.arg_types)
    )

def _replace_at_path(root: Node, path: Tuple[int, ...], repl: Node) -> Node:
    """New tree with the subtree at *path* replaced by *repl*."""
    if not path:
        return repl
    assert isinstance(root, FuncNode)
    idx, rest = path[0], path[1:]
    kids = list(root.children)
    kids[idx] = _replace_at_path(root.children[idx], rest, repl)
    return FuncNode(defn=root.defn, children=kids)


def _node_at_path(root: Node, path: Tuple[int, ...]) -> Node:
    """Walk *path* from *root*; return the node at that location."""
    n = root
    for i in path:
        assert isinstance(n, FuncNode)
        n = n.children[i]
    return n


def _path_is_emblag_first_child(root: Node, path: Tuple[int, ...]) -> bool:
    """True iff *path* points to the first child (index 0) of an EmbLag_*
    node inside *root*. Used by subtree_mutate and crossover to detect
    the EmbLag first-arg slot before modifying the tree."""
    if not path or path[-1] != 0:
        return False
    parent = _node_at_path(root, path[:-1])
    return isinstance(parent, FuncNode) and is_emblag_funcdef(parent.defn)


def _crossover_would_violate_emblag(
    root: Node, path: Tuple[int, ...], repl: Node,
) -> bool:
    """Detect the two crossover violations of the EmbLag first-arg rule:

      (a) *path* lands on the first child of an EmbLag_* node in *root*
          AND *repl* is a FuncNode — swap would install a non-terminal
          as the lag operand.
      (b) *repl* is itself an EmbLag_* FuncNode whose first child is
          a FuncNode — swapping it in anywhere would drag an already-
          invalid subtree into the parent. (The current grammar never
          constructs such subtrees, but crossover partners from an
          externally-provided tree could; be conservative.)

    Returns True if EITHER violation would occur.
    """
    # (a)
    if _path_is_emblag_first_child(root, path) and not isinstance(repl, TermNode):
        return True
    # (b) — scan *repl* for any internal EmbLag_* with non-terminal first child.
    def _bad(n: Node) -> bool:
        if isinstance(n, TermNode):
            return False
        if is_emblag_funcdef(n.defn) and not isinstance(n.children[0], TermNode):
            return True
        return any(_bad(c) for c in n.children)
    return _bad(repl)

# ---------------------------------------------------------------------------
# Standard function set (18 operators)
# ---------------------------------------------------------------------------

GT  = FuncDef("GT",  (GType.REAL, GType.REAL), GType.BOOL)
LT  = FuncDef("LT",  (GType.REAL, GType.REAL), GType.BOOL)
AND = FuncDef("AND", (GType.BOOL, GType.BOOL), GType.BOOL)
OR  = FuncDef("OR",  (GType.BOOL, GType.BOOL), GType.BOOL)
NOT = FuncDef("NOT", (GType.BOOL,),            GType.BOOL)

ADD  = FuncDef("Add",  (GType.REAL, GType.REAL), GType.REAL)
SUB  = FuncDef("Sub",  (GType.REAL, GType.REAL), GType.REAL)
MUL  = FuncDef("Mul",  (GType.REAL, GType.REAL), GType.REAL)
DIV  = FuncDef("Div",  (GType.REAL, GType.REAL), GType.REAL)  # analytic quotient
SQRT = FuncDef("Sqrt", (GType.REAL,),            GType.REAL)

LAG         = FuncDef("Lag",        (GType.REAL, GType.INT),  GType.REAL)
DELTA       = FuncDef("Delta",      (GType.REAL, GType.INT),  GType.REAL)
CROSS_ABOVE = FuncDef("CrossAbove", (GType.REAL, GType.REAL), GType.BOOL)
CROSS_BELOW = FuncDef("CrossBelow", (GType.REAL, GType.REAL), GType.BOOL)

IN_REGIME = FuncDef("InRegime", (GType.REGIME,),               GType.BOOL)
REGIME_IS = FuncDef("RegimeIs", (GType.REGIME, GType.REGIME),  GType.BOOL)

IF_REAL = FuncDef("IfThenElse", (GType.BOOL, GType.REAL, GType.REAL), GType.REAL)
IF_SIDE = FuncDef("IfSide",     (GType.BOOL, GType.SIDE, GType.SIDE), GType.SIDE)

# typed-vector operators — one FuncDef per (operator, embedding-type)
# combination so the grammar's type system enforces same-type matching on
# binary ops (EmbCos and EmbSub only accept same-type vectors).
#
# Per type t ∈ EMB_TYPES:
# EmbNorm_t: (t,) → REAL (L2 norm, scale signal)
# EmbCos_t: (t, t) → REAL (cosine similarity, shape signal)
# EmbSub_t: (t, t) → t (vector difference — cross-time or cross-state)
# EmbLag_t: (t, INT) → t (vector from `int` bars ago via rolling buffer)
#
# This is 4 operators × 10 typed-vector groups = 40 FuncDefs.
EMB_OPERATORS: List[FuncDef] = []
# EmbLag FuncDef registry — these operators have a hard grammar-level
# constraint that their first argument MUST be a TermNode (typed-vector
# terminal). See the evaluator's EmbLag_* branch: for a non-leaf first
# child, the evaluator silently falls back to no-lag evaluation, which
# would let GP generate trees whose semantics DON'T match their printed
# shape. We reject such trees at generation / crossover / mutation time
# via `_is_valid_emblag_tree` below, rather than changing the evaluator's
# defensive fallback.
_EMBLAG_FUNCDEFS: set[FuncDef] = set()
for _t in EMB_TYPES:
    _n = _t.name  # e.g. "EMB_GRID"
    EMB_OPERATORS.append(FuncDef(f"EmbNorm_{_n}", (_t,),           GType.REAL))
    EMB_OPERATORS.append(FuncDef(f"EmbCos_{_n}",  (_t, _t),        GType.REAL))
    EMB_OPERATORS.append(FuncDef(f"EmbSub_{_n}",  (_t, _t),        _t))
    _emblag = FuncDef(f"EmbLag_{_n}",  (_t, GType.INT), _t)
    EMB_OPERATORS.append(_emblag)
    _EMBLAG_FUNCDEFS.add(_emblag)

# v8 (RF / grammar extension): per-typed-vector PCA projection operators.
# Each EmbProj_EMB_X_k is a unary function (EMB_X,) → REAL that projects
# the typed vector onto the k-th principal component of a pre-fit PCA
# basis (fit on the training window only, persisted to layer2/pca_bases.npz
# per `layer2.inference.pca_bases`). The PC index is baked into the
# operator NAME (not an evolvable INT argument) — mirrors the existing
# EmbNorm / EmbCos / EmbSub family's shape, so scalar-only grammar strip
# auto-filters via the existing `f not in EMB_OPERATORS` membership
# check, and existing validation / crossover / mutation machinery
# requires zero new code.
#
# v9 amendment: K is per-group (capped at 15), replacing the v8 flat K=5.
# Per-group K is set to min(15, ceil(eff_rank)) on the train slice and
# locked by the user (2026-04-25). Total operators: 99 (84 group-pooled
# + 15 R2 per-variate, was 35 at v8).
#
# L8-12 SINGLE SOURCE-OF-TRUTH: PER_GROUP_K_V9 is defined ONCE in
# `layer2/inference/pca_bases.py` and imported here. A divergence between
# grammar and PCA basis K would silently make the
# `if k >= components.shape[0]` guard in evaluator's EmbProj dispatch
# return 0.0 for a subset of operators. Importing from a single source
# eliminates that drift. The unit test
# `test_v9_grammar_pca_per_group_k_single_source_of_truth` verifies they
# stay in sync.
from layer2.inference.pca_bases import PER_GROUP_K_V9 as _PER_GROUP_K_V9
PER_GROUP_K_GRAMMAR_V9: Dict[str, int] = dict(_PER_GROUP_K_V9)
# Sum across groups should be 99 (84 group-pooled + 15 per-variate R2).
_TOTAL_EMBPROJ_OPERATORS_V9 = sum(PER_GROUP_K_GRAMMAR_V9.values())
assert _TOTAL_EMBPROJ_OPERATORS_V9 == 99, (
    f"PER_GROUP_K_GRAMMAR_V9 sums to {_TOTAL_EMBPROJ_OPERATORS_V9}, "
    f"expected 99 (v9 design spec + R2 per-variate extension)"
)
for _t in EMB_TYPES:
    _n = _t.name
    if _n not in PER_GROUP_K_GRAMMAR_V9:
        raise RuntimeError(
            f"L8-12 single-source-of-truth violation: PER_GROUP_K_V9 (from "
            f"layer2.inference.pca_bases) is missing entry for {_n!r}. "
            f"Got keys: {sorted(PER_GROUP_K_GRAMMAR_V9)}. Update "
            f"layer2/inference/pca_bases.py::PER_GROUP_K_V9."
        )
    _k_for_group = PER_GROUP_K_GRAMMAR_V9[_n]
    for _k in range(_k_for_group):
        EMB_OPERATORS.append(
            FuncDef(f"EmbProj_{_n}_{_k}", (_t,), GType.REAL)
        )


def is_emblag_funcdef(fd: FuncDef) -> bool:
    """True iff `fd` is an EmbLag_* typed-vector lag operator.

    EmbLag requires a TermNode first child (see the evaluator; nested
    FuncNode first child silently falls back to no-lag evaluation). The
    tree validator and constructors use this predicate to enforce the
    constraint at generation time.
    """
    return fd in _EMBLAG_FUNCDEFS

ALL_FUNCTIONS: List[FuncDef] = [
    GT, LT, AND, OR, NOT, ADD, SUB, MUL, DIV, SQRT,
    LAG, DELTA, CROSS_ABOVE, CROSS_BELOW,
    IN_REGIME, REGIME_IS, IF_REAL, IF_SIDE,
] + EMB_OPERATORS


# F2 fix (2026-04-25, HIGH) — operator-class–stratified sampling.
#
# Without stratification, `_feasible_func` picks uniformly across the per-
# return-type pool. real-l1's REAL pool has ~106 candidates (incl. 84
# EmbProj_*); scalar-only's REAL pool has 8. So per random REAL slot fill,
# scalar-only's basic arithmetic gets ~13× the selection density of real-l1.
# This compounds the search-space size asymmetry into a per-mutation
# pressure asymmetry — at mutation_rate=0.25, scalar-only's effective
# arithmetic-perturbation rate is ~12× real-l1's.
#
# F2 fix: stratified two-stage sampling. Stage 1 picks a *class* uniformly
# (e.g. arithmetic, comparison, EmbProj, regime). Stage 2 picks a function
# within that class uniformly. This decouples per-class selection density
# from per-class cardinality. EmbProj's 84 ops get the same class-level
# weight as the 5 arithmetic ops, so real-l1's arithmetic backbone is no
# longer drowned by the EmbProj expansion.
#
# Class assignments (groupings preserve operator semantics):
# arithmetic: ADD, SUB, MUL, DIV, SQRT
# comparison: GT, LT
# logical: AND, OR, NOT
# temporal: LAG, DELTA, CROSS_ABOVE, CROSS_BELOW
# regime: IN_REGIME, REGIME_IS
# conditional: IF_REAL, IF_SIDE
# emb_norm: EmbNorm_* (per-group)
# emb_cos: EmbCos_* (per-group)
# emb_sub: EmbSub_* (per-group)
# emb_lag: EmbLag_* (per-group)
# emb_proj: EmbProj_* (per-group, all PCs)
#
# Note: emb_norm/cos/sub/lag/proj are kept as SEPARATE classes (one each).
# They could be merged into one "emb" super-class — that's a design choice.
# Keeping them separate gives EmbProj the same class-level weight as
# EmbNorm/EmbCos/EmbSub/EmbLag, mirroring the v8 grammar shape where each
# of those was a single-operator family.
OPERATOR_CLASSES: Dict[str, Tuple[FuncDef, ...]] = {
    "arithmetic":  (ADD, SUB, MUL, DIV, SQRT),
    "comparison":  (GT, LT),
    "logical":     (AND, OR, NOT),
    "temporal":    (LAG, DELTA, CROSS_ABOVE, CROSS_BELOW),
    "regime":      (IN_REGIME, REGIME_IS),
    "conditional": (IF_REAL, IF_SIDE),
    "emb_norm":    tuple(f for f in EMB_OPERATORS if f.name.startswith("EmbNorm_")),
    "emb_cos":     tuple(f for f in EMB_OPERATORS if f.name.startswith("EmbCos_")),
    "emb_sub":     tuple(f for f in EMB_OPERATORS if f.name.startswith("EmbSub_")),
    "emb_lag":     tuple(f for f in EMB_OPERATORS if f.name.startswith("EmbLag_")),
    "emb_proj":    tuple(f for f in EMB_OPERATORS if f.name.startswith("EmbProj_")),
}

# Reverse map: FuncDef -> class name. Built once at module load. Used by
# `_feasible_func` to bucket the candidate pool.
_FUNC_TO_CLASS: Dict[FuncDef, str] = {}
for _cls_name, _cls_funcs in OPERATOR_CLASSES.items():
    for _f in _cls_funcs:
        _FUNC_TO_CLASS[_f] = _cls_name
# Sanity: every ALL_FUNCTIONS entry must be classified — otherwise the
# stratified sampler would silently exclude it from the pool.
_unclassified = [f.name for f in ALL_FUNCTIONS if f not in _FUNC_TO_CLASS]
assert not _unclassified, (
    f"F2 violation: {len(_unclassified)} ALL_FUNCTIONS not in any "
    f"OPERATOR_CLASSES bucket: {_unclassified[:5]}... "
    f"Add them to OPERATOR_CLASSES."
)

# ---------------------------------------------------------------------------
# Standard terminal set (: ~45 terminals)
# ---------------------------------------------------------------------------

_LAG_CHOICES = (1, 3, 5, 10, 15, 30)

def build_terminal_set() -> List[TermDef]:
    """Assemble the full terminal set (Condition D) per D84 typed-vector contract.

    Channel 1 (typed vectors):       10 opaque vector terminals (7 base + 3 R2 per-variate)
    Channel 2 (probe forecasts):     4 scalar terminals
    Channel 3 (raw bypass scalars):  7 scalar terminals + 3 smoothed (5m)
    Channel 4 (0DTE structural):     MinutesToClose, BarOfDay, SessionReturn
    Other:                           EphReal, EphInt, Regime/Side literals
    """
    T, R = TermDef, GType.REAL
    terms: List[TermDef] = []
    # Channel 1: typed vector terminals ( typed-vector contract, replaces
    # the retired EmbPC{k} PCA-compressed scalars). Each terminal is an
    # opaque vector that GP composes only via typed operators (EmbNorm /
    # EmbCos / EmbSub / EmbLag). The terminal NAME is the GType name by
    # convention; the evaluator resolves the name back to the vector at
    # evaluation time via EvaluationContext.get_vec().
    # Includes R2 per-variate types (EMB_VAR_ATM_IV, EMB_VAR_ATM_SPREAD,
    # EMB_VAR_SPX_RET) which bypass the mean-pool bottleneck.
    for emb_t in EMB_TYPES:
        terms.append(T(emb_t.name, emb_t))   # e.g. T("EMB_GRID", GType.EMB_GRID)
    # Channel 2: probe forecasts
    # PredRegime replaced with binary regime indicators (fix: ordinal GT/LT on
    # categorical labels wastes search budget on redundant thresholds).
    # RegimeAboveLow=1 when regime≥1, RegimeIsHigh=1 when regime≥2,
    # RegimeIsPremium=1 when regime=3. Derived from PredRegime in bar_data.
    terms += [T("PredRV15", R), T("PredRV30", R), T("PredSpread", R)]
    # : 4 new probe targets (see (internal doc) for justification)
    terms += [T("PredGammaAccel", R), T("PredSmileConvexity", R)]
    terms += [T("PredJump", R), T("PredFlowToxicity", R)]
    terms += [T("RegimeAboveLow", R), T("RegimeIsHigh", R), T("RegimeIsPremium", R)]
    # Channel 3: raw bypass scalars (VIX routed here per /SSL-004-FE finding
    # that vix_term reconstruction is structurally negative under MVR)
    terms += [
        T("DeltaSpread1", R), T("DeltaSpread5", R), T("RawSpread", R), T("ATM_IV", R),
        T("VIXSpot", R), T("VIXTermSlope", R), T("RealizedVol30m", R),
    ]
    # Channel 4: 0DTE primitives (available now)
    terms += [T("MinutesToClose", R)]
    # BarOfDay: 0-indexed position within session (0 = first bar, ~59 = last
    # for 60-bar 5-min sessions). Computed from MinutesToClose in the evaluator
    # if not present in the Parquet. Gives GP direct access to time-of-day
    # without needing to invert MinutesToClose semantics.
    terms += [T("BarOfDay", R)]
    # SessionReturn: cumulative SPX return since session open (×100 for scale).
    # Vilkov (2023): conditional 10:00 ET entry rules deliver economically
    # meaningful OOS performance. Gao et al. (2018, JFE): first-30-min
    # return predicts rest-of-day. Gives GP intraday momentum/mean-reversion.
    terms += [T("SessionReturn", R)]
    # 5-bar smoothed terminals: reduce 1-minute noise overfitting while
    # preserving signal (Lopez de Prado 2018, Ch. 5). GP can choose between
    # raw (minute-level) and smoothed (5-min) representations.
    terms += [T("ATM_IV_5m", R), T("RealizedVol30m_5m", R), T("RawSpread_5m", R)]
    terms += [T("VIXChange", R)]
    # Put-call skew + session position (Fixes #3, #5 from financial correctness audit)
    terms += [T("PutCallSkew", R), T("SessionPosition", R)]
    # OvernightGap: log(open/prev_close). Captures overnight sentiment shifts.
    terms += [T("OvernightGap", R)]
    # GridReliability: fraction of IV grid points with valid data [0,1].
    # GP can gate entries on high reliability to avoid extrapolated pricing.
    terms += [T("GridReliability", R)]
    # ThetaUrgency: 1/sqrt(max(MTC, 1)). Captures nonlinear 0DTE theta decay
    # (BS ATM theta scales as 1/sqrt(T)). GP cannot construct this from
    # existing terminals: Div is analytic quotient, normalization makes MTC
    # negative in PM, 4-node cost is 27% of budget. Sufficiency restoration
    # per Koza (1992). Kommenda et al. (2020), Virgolin et al. (2021).
    terms += [T("ThetaUrgency", R)]
    # Inter-day memory terminals (constant within day, updated at session open).
    # Andersen et al. (2003): multi-day vol clustering is the primary driver
    # of realized-vol dynamics. These terminals give GP the ability to condition
    # on sustained regime states that single-day scalars cannot detect.
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R)]
    terms += [T("RV5d", R), T("IVRVGap5d", R)]
    # UnrealizedProfitPct: position-aware unrealized-profit FRACTION (Phase-2
    # of docs/viable_run_plan_2026_06_03.md). The exit-tree's only POSITION-STATE
    # terminal — lets the GP express the canonical premium-selling exit
    # ("close at 50% of max profit", GT(UnrealizedProfitPct, EphReal(0.5))) and
    # loss-cuts, which no MARKET terminal can. RAW (un-normalized): the fraction
    # is already a natural bounded scale (≈0 at entry, →+1 at max profit,
    # negative against you), so EphReal in [-3,3] reaches the whole [0,1] profit
    # band literally. 0.0 when FLAT (entry-tree references are benign no-ops).
    # Position-dependent → CANNOT be precomputed; the evaluator injects the live
    # per-bar value into exit-tree evaluation (see evaluator_vectorized.py).
    # Condition-agnostic (it is position state, not an encoder feature): present
    # in ALL conditions so the A/B/C/D ablation stays fair.
    terms += [T("UnrealizedProfitPct", R)]
    # Ephemeral constants
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    # Regime literals
    for r in Regime:
        terms.append(T(r.name, GType.REGIME, value=r))
    # Side literals
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


# ---------------------------------------------------------------------------
# H1 scalar-only subset (H2 third condition — see Commit 4)
# ---------------------------------------------------------------------------

def build_scalar_only_terminal_set() -> List[TermDef]:
    """Terminal set for the scalar-only H2 condition.

    Strips:
      - 7 typed-vector terminals (Channel 1)
      - 4 L1 probe scalars (Channel 2: PredRV15/PredRV30/PredRegime/PredSpread)
      - 4 RegimeProb columns (encoder-derived softmax outputs)
      - All 4 Regime literal terminals (these only function in the presence
        of PredRegime / RegimeProb input — without them, IN_REGIME and
        REGIME_IS would compare to the evaluator's `current_regime`, which
        is itself populated from L1's PredRegime → leakage)

    Keeps raw bypass scalars (Channel 3), 0DTE structural terminals
    (Channel 4: MinutesToClose, BarOfDay, SessionReturn), 5m smoothed
    terminals, ephemeral constants, and Side literals.

    F1 fix (2026-04-25, BLOCKER): regime literal terminals removed because
    leaving them in — combined with IN_REGIME/REGIME_IS in the function
    set — let scalar-only trees consume `ctx.current_regime`, which is set
    from PredRegime by EvaluationContext.update(). That is the L1 4-class
    regime probe, an iTransformer head output, NOT a raw signal. So
    "scalar-only" was silently L1-aware via the regime axis. F1 removes
    BOTH the operators (in SCALAR_ONLY_FUNCTIONS) AND the Regime literal
    terminals (here), AND filters PredRegime/RegimeProb out of bar_data
    in the backtester (see fitness.py / experiment.py condition routing).

    Rationale: the dissertation H1 claim is that TYPED L1 EMBEDDINGS carry
    discriminative signal beyond what raw scalars expose. Without this
    subset as a third H2 arm, the two-condition test (real-l1 vs shuffled-
    l1) only shows "L1 has signal at all" — not "typed embeddings
    specifically beat raw scalars." Per AI Engineer review.

    Probe scalars are stripped because they are L1-derived (PredRV15/30/
    Regime/Spread come from the iTransformer probes); leaving them in
    would make scalar-only a "probe-only" condition, which is an
    ABLATION of L1, not an EXCLUSION of it.
    """
    T, R = TermDef, GType.REAL
    terms: List[TermDef] = []
    # Channel 3: raw bypass scalars (identical to full set)
    terms += [
        T("DeltaSpread1", R), T("DeltaSpread5", R), T("RawSpread", R), T("ATM_IV", R),
        T("VIXSpot", R), T("VIXTermSlope", R), T("RealizedVol30m", R),
    ]
    # Channel 4: 0DTE primitives
    terms += [T("MinutesToClose", R)]
    terms += [T("BarOfDay", R)]
    terms += [T("SessionReturn", R)]
    terms += [T("ATM_IV_5m", R), T("RealizedVol30m_5m", R), T("RawSpread_5m", R)]
    # VIXChange: day-over-day VIX change. VIXSpot is daily-constant so
    # Delta(VIXSpot, k) always returns 0 intraday. VIXChange gives GP
    # the overnight VIX dynamics (spike/collapse detection) directly.
    terms += [T("VIXChange", R)]
    # PutCallSkew: IV(25Δput) - IV(25Δcall). Primary 0DTE edge signal —
    # skew dynamics drive iron condor/butterfly profitability and signal
    # crash-protection demand shifts. Condition A MUST have this.
    terms += [T("PutCallSkew", R)]
    # SessionPosition: where price sits within today's range [0=low, 1=high].
    # Mean-reversion signal — extremes tend to revert intra-session (0DTE).
    terms += [T("SessionPosition", R)]
    # OvernightGap: log(open/prev_close). Same as full condition set.
    terms += [T("OvernightGap", R)]
    # GridReliability: fraction of valid grid IVs. Same as full set.
    terms += [T("GridReliability", R)]
    # ThetaUrgency: 1/sqrt(MTC). Same as full set — sufficiency restoration.
    terms += [T("ThetaUrgency", R)]
    # Inter-day memory terminals (same as full set for scalar-only).
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R)]
    terms += [T("RV5d", R), T("IVRVGap5d", R)]
    # UnrealizedProfitPct: position-aware unrealized-profit FRACTION. Position
    # STATE, not an encoder feature, so it is present in EVERY condition equally
    # (the A/B/C/D ablation must compare encoder representations, not exit
    # vocabulary). RAW + 0.0-when-flat; the evaluator injects the live per-bar
    # value into exit-tree evaluation. See build_terminal_set() for the full note.
    terms += [T("UnrealizedProfitPct", R)]
    # Ephemeral constants (evolution still needs ephemeral real + int lags)
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    # F1 fix: Regime literals INTENTIONALLY ABSENT. With IN_REGIME/REGIME_IS
    # also stripped from SCALAR_ONLY_FUNCTIONS, no scalar-only tree can
    # reference regime semantics — the grammar is genuinely L1-free.
    # Side literals retained: SIDE is a valid scalar-only GType (only used
    # by IF_SIDE which needs SIDE inputs from Side literals — neutral to
    # L1).
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


def build_probes_only_terminal_set() -> List[TermDef]:
    """Terminal set for Condition B (probes-pure): probes + structural, NO bypass scalars.

    Clean ablation design (L2.60): encoder-input scalars (ATM_IV, VIXSpot,
    RealizedVol30m, RawSpread, DeltaSpread1/5, VIXTermSlope) are EXCLUDED
    because they are already encoded in the encoder's 147-variate input.
    Including them would let the GP shortcut around the probes.

    Includes:
      - Probe scalars: PredRV15, PredRV30, PredSpread, PredRegime
      - Structural terminals: MinutesToClose, BarOfDay, SessionReturn
        (NOT in encoder input — encoder has no absolute clock)
      - Regime/Side literals, ephemerals

    Excludes:
      - ALL bypass scalars that are in the encoder's input
      - ALL typed-vector terminals (EMB_*)
    """
    T, R = TermDef, GType.REAL
    terms: List[TermDef] = []
    # Encoder's supervised scalar outputs (PredRegime replaced with binary indicators)
    terms += [T("PredRV15", R), T("PredRV30", R), T("PredSpread", R)]
    # expansion: 4 new probe targets (peer-reviewed, see (internal doc))
    terms += [T("PredGammaAccel", R)]      # Bakshi & Kapadia (2003): gamma/theta trajectory
    terms += [T("PredSmileConvexity", R)]   # Carr & Wu (2009): IV butterfly spread change
    terms += [T("PredJump", R)]             # Bollerslev & Todorov (2011): tail risk probability
    terms += [T("PredFlowToxicity", R)]     # Easley et al. (2012): order flow toxicity
    terms += [T("RegimeAboveLow", R), T("RegimeIsHigh", R), T("RegimeIsPremium", R)]
    # Structural terminals NOT in encoder input (absolute time, cumulative return)
    terms += [T("MinutesToClose", R), T("BarOfDay", R), T("SessionReturn", R)]
    # Inter-day terminals: multi-day context the encoder cannot see
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R), T("RV5d", R), T("IVRVGap5d", R)]
    # Non-encoder-derived environmental context — same as Condition A to avoid
    # confounding A-vs-B comparison (ablation audit 2026-05-14).
    terms += [T("PutCallSkew", R), T("OvernightGap", R)]
    terms += [T("GridReliability", R), T("SessionPosition", R)]
    terms += [T("ThetaUrgency", R)]  # 1/sqrt(MTC) — sufficiency restoration
    # UnrealizedProfitPct: position-aware exit terminal. Position STATE (not an
    # encoder/probe output), so it is in EVERY condition equally to keep the
    # ablation fair. See build_terminal_set() for the full note.
    terms += [T("UnrealizedProfitPct", R)]
    # EXCLUDED: ATM_IV_5m, RealizedVol30m_5m, RawSpread_5m, VIXChange —
    # all derived from encoder-input scalars. Including them lets GP shortcut
    # around probes. Stricter ablation ( principled design).
    # Ephemerals
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    # Regime literals (functional with PredRegime as input source)
    for r in Regime:
        terms.append(T(r.name, GType.REGIME, value=r))
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


def build_emb_only_terminal_set() -> List[TermDef]:
    """Terminal set for Condition C (emb-pure): embeddings + market-structure, NO bypass scalars.

    Clean ablation design (L2.60, updated L2.63): encoder-input scalars
    (ATM_IV, VIXSpot, RealizedVol30m, RawSpread, DeltaSpread1/5,
    VIXTermSlope) are EXCLUDED because they are already encoded in the
    encoder's 147-variate input. Including them would let the GP shortcut
    around the embedding operators.

    Includes:
      - Typed-vector terminals: 10 EMB_* groups (384-d each)
      - EmbProj/EmbNorm/EmbCos/EmbSub/EmbLag operators
      - Structural terminals: MinutesToClose, BarOfDay, SessionReturn
        (NOT in encoder input — encoder has no absolute clock)
      - Market-structure terminals: PutCallSkew, OvernightGap,
        GridReliability, SessionPosition — derived from minute-bar
        price/IV data, NOT from the encoder's 147-variate input, NOT
        bypass scalars, NOT probe outputs. These provide GP with enough
        REAL terminals for meaningful BOOL construction (L2.63 fix for
        terminal starvation) and enable semantically useful seed
        substitutions (L2.63 fix for seed degeneration).
      - Side literals, ephemerals

    Excludes:
      - ALL bypass scalars that are in the encoder's input
      - ALL probe scalars (PredRV15/30/Spread/Regime/GammaAccel/etc.)
      - Regime literals (regime comes from probes which are excluded)
      - Smoothed scalars (ATM_IV_5m, etc.) — derived from encoder inputs
      - VIXChange — derived from VIXSpot (encoder input)
    """
    T, R = TermDef, GType.REAL
    terms: List[TermDef] = []
    # Typed-vector terminals (for EmbNorm/EmbCos/EmbSub/EmbLag/EmbProj)
    for grp in ("EMB_SHARED", "EMB_GRID", "EMB_STRIKE", "EMB_SPX", "EMB_VIX",
                "EMB_FLOW_RAW", "EMB_FLOW_AGG",
                "EMB_VAR_ATM_IV", "EMB_VAR_ATM_SPREAD", "EMB_VAR_SPX_RET"):
        terms.append(T(grp, GType[grp]))
    # Structural terminals NOT in encoder input (absolute time, cumulative return)
    terms += [T("MinutesToClose", R), T("BarOfDay", R), T("SessionReturn", R)]
    terms += [T("ThetaUrgency", R)]  # 1/sqrt(MTC) — structural time terminal
    # Inter-day terminals: capture multi-day context that the encoder does NOT
    # see (encoder processes intraday 60-bar windows only). These mitigate
    # terminal starvation without undermining the ablation — they provide
    # COMPLEMENTARY cross-day context available to both A and C conditions.
    terms += [T("SPXReturn3d", R), T("VIXMean5d", R), T("RV5d", R), T("IVRVGap5d", R)]
    # UnrealizedProfitPct: position-aware exit terminal. Position STATE, not a
    # market-derived scalar and not an encoder feature — including it does NOT
    # let GP shortcut around EmbProj (it carries no encoder information). Present
    # in EVERY condition equally so the ablation compares representations, not
    # exit vocabulary. See build_terminal_set() for the full note.
    terms += [T("UnrealizedProfitPct", R)]
    # REMOVED (clean ablation, #29): PutCallSkew, OvernightGap,
    # GridReliability, SessionPosition were market-derived scalars that let
    # GP shortcut around EmbProj operators. The encoder utility audit showed
    # GP used these INSTEAD of EmbProj in the prior Condition C run. For a
    # clean H2 ablation, Condition C must only have time primitives + EmbProj.
    # If GP cannot produce viable strategies without these, that IS the answer
    # to whether the encoder is useful.
    # EXCLUDED: ATM_IV_5m, RealizedVol30m_5m, RawSpread_5m, VIXChange —
    # all derived from encoder-input scalars.
    # Ephemerals
    terms.append(T("EphReal", R, sampler=lambda: round(random.uniform(-3, 3), 4)))
    terms.append(T("EphInt", GType.INT, sampler=lambda: random.choice(_LAG_CHOICES)))
    # No Regime literals (regime is probe-derived, probes excluded in this condition)
    for s in Side:
        terms.append(T(s.name, GType.SIDE, value=s))
    return terms


# Functions available under the scalar-only condition. F1 fix
# (BLOCKER, 2026-04-25): strips EMB_OPERATORS (was already done) PLUS
# IN_REGIME and REGIME_IS — these compare against ctx.current_regime,
# which EvaluationContext.update() populates from PredRegime (an L1
# iTransformer probe output). Leaving IN_REGIME/REGIME_IS in scalar-only
# silently let scalar-only trees consume L1's 4-class regime label even
# though `_SCALAR_ONLY_TERMS_FORBIDDEN` blocked PredRegime as a terminal.
# The grammar must be operator-coherent with the terminal set: no
# IN_REGIME / REGIME_IS without Regime input source.
_SCALAR_ONLY_FORBIDDEN_FUNCS: Tuple[FuncDef, ...] = (IN_REGIME, REGIME_IS)
SCALAR_ONLY_FUNCTIONS: List[FuncDef] = [
    f for f in ALL_FUNCTIONS
    if f not in EMB_OPERATORS and f not in _SCALAR_ONLY_FORBIDDEN_FUNCS
]

# Condition B (probes-pure): scalar functions + regime operators, NO embedding operators.
# GP gets probes (PredRV15/30/Spread/Regime) + structural (MTC/BoD/SR) + regime ops.
# NO bypass scalars (ATM_IV, VIX, etc.) — those are in the encoder's input ().
PROBES_ONLY_FUNCTIONS: List[FuncDef] = [
    f for f in ALL_FUNCTIONS
    if f not in EMB_OPERATORS
]

# Condition C (emb-pure): full embedding operators, NO probe scalars or bypass scalars.
# GP gets typed vectors + EmbX operators + structural (MTC/BoD/SR) only ().
# InRegime/RegimeIs stripped: REGIME type has zero terminals in C, making these
# unreachable. Removing them is defense-in-depth (ablation audit 2026-05-14).
EMB_ONLY_FUNCTIONS: List[FuncDef] = [
    f for f in ALL_FUNCTIONS if f not in (IN_REGIME, REGIME_IS)
]

# : Screened Condition C function set.
# Removes unsupervised embedding operators (EmbNorm, EmbCos, EmbSub, EmbLag)
# and filters EmbProj to only pre-screened components (top K=3 per group,
# BH q<0.10). This reduces the search space from 157 to ~48 functions,
# bringing it within ~3x of Condition A's 18 — removing the search space
# confound from the ablation.
#
# Pre-screening manifest: layer2/embproj_screen.json (generated by
# scripts/screen_embproj.py using training-split Spearman correlations).
_UNSUPERVISED_EMB_OPS: frozenset = frozenset(
    f for f in EMB_OPERATORS
    if f.name.startswith(("EmbNorm_", "EmbCos_", "EmbSub_", "EmbLag_"))
)

def _load_screened_embproj() -> frozenset:
    """Load pre-screened EmbProj operator names from manifest.

    Lazy-loaded: only called when build_emb_only_screened_functions() is
    invoked (Condition C runs). Not called at module import time — avoids
    filesystem I/O and warnings for Condition A/B/D/test imports.

    Raises FileNotFoundError if manifest missing — silent fallback would
    invalidate the ablation without any indication.
    """
    import json
    screen_path = Path(__file__).parent / "embproj_screen.json"
    if not screen_path.exists():
        raise FileNotFoundError(
            f"EmbProj screen manifest not found at {screen_path}. "
            f"Run 'python -m scripts.screen_embproj' to generate it "
            f"before running Condition C."
        )
    data = json.loads(screen_path.read_text())
    screened = data.get("screened_components", {})
    allowed_names = set()
    for group, ks in screened.items():
        for k in ks:
            allowed_names.add(f"EmbProj_{group}_{k}")
    return frozenset(allowed_names)

# Lazy cache — populated on first call to build_emb_only_screened_functions()
_SCREENED_EMBPROJ_NAMES: Optional[frozenset] = None

def build_emb_only_screened_functions() -> List[FuncDef]:
    """Screened function set for Condition C (L2.64).

    Includes:
      - 18 standard scalar functions (minus IN_REGIME/REGIME_IS)
      - ~30 screened EmbProj operators (top 3 per group, BH q<0.10)
    Excludes:
      - EmbNorm, EmbCos, EmbSub, EmbLag (unsupervised — no target-aligned signal)
      - Unscreened EmbProj components (noise that dilutes search)
      - IN_REGIME, REGIME_IS (no regime terminals in C)

    Total: ~46 functions vs 157 in the unscreened grammar.
    """
    global _SCREENED_EMBPROJ_NAMES
    if _SCREENED_EMBPROJ_NAMES is None:
        _SCREENED_EMBPROJ_NAMES = _load_screened_embproj()

    result = []
    for f in ALL_FUNCTIONS:
        # Skip regime operators (no regime terminals in C)
        if f in (IN_REGIME, REGIME_IS):
            continue
        # Skip all unsupervised embedding operators
        if f in _UNSUPERVISED_EMB_OPS:
            continue
        # For EmbProj, keep only screened components
        if f.name.startswith("EmbProj_"):
            if f.name not in _SCREENED_EMBPROJ_NAMES:
                continue
        result.append(f)
    return result


# Raw-scalar analogs for L1 probe scalars (used when rewriting seed trees
# for the scalar-only condition). Each probe scalar maps to the closest
# raw-scalar signal of the same SEMANTIC CLASS:
# PredRV15/30 → RealizedVol30m (both RV measures; predictive → reactive)
# PredSpread → RawSpread (direct raw analog)
# PredRegime is INTENTIONALLY ABSENT: regime is a 4-class L1 inference
# output ({LOW_VOL, MID_VOL, HIGH_VOL_STRESSED, HIGH_VOL_PREMIUM}), which has
# no raw-scalar analog. Silently mapping to AbstainFlag would produce
# nonsense geometry (binary {0,1} vs. 4-class categorical) while
# appearing to succeed — a methodological trap flagged by the L2
# Commit 4 audit (AI Engineer + Model QA). We reject rather than coerce:
# if a future seed tree adds PredRegime, the substitution helper raises
# ValueError, forcing an explicit design decision per template.
_SCALAR_ONLY_TERM_SUBSTITUTIONS: Dict[str, str] = {
    "PredRV15":   "RealizedVol30m",
    "PredRV30":   "RealizedVol30m",
    # BLOCKER-14 (2026-06-01 audit): PredSpread MUST NOT map to RawSpread —
    # RawSpread is itself QC-NON-TRANSFERABLE (the History()-built proxy spread
    # diverges from QC's live option-chain spread). Map to RealizedVol30m, a
    # TRANSFERABLE vol proxy correlated with spread (wider spreads track higher
    # vol), so a probe→scalar seed rewrite never re-introduces a non-transferable
    # leaf. (RealizedVol30m is in the scalar-only sampling pool.)
    "PredSpread": "RealizedVol30m",
}

# BLOCKER-14 (2026-06-01 data audit): mechanical, TYPE-SAFE (REAL→REAL)
# substitution of the 8 QC-NON-TRANSFERABLE scalars when rewriting a template
# seed for the scalar-only condition. WHY: the scalar-only SEED path
# (substitute_stripped_terminals_for_scalar_only) previously passed plain
# scalars through UNCHANGED, so every Level-B template seed (all 5 gate on
# SessionReturn) injected a non-transferable terminal into ~15% of the initial
# population (seed_fraction). Random tree GENERATION already excludes these
# (Grammar.__init__ filters _QC_NONTRANSFERABLE_SCALARS from the sampling pool),
# but seeds bypassed that filter. Each target is the semantically-closest
# TRANSFERABLE REAL terminal that IS in build_scalar_only_terminal_set():
# SessionReturn / OvernightGap → SPXReturn3d (directional return)
# RawSpread / RawSpread_5m / SessionPosition → RealizedVol30m_5m (vol/liquidity ctx)
# DeltaSpread1 / DeltaSpread5 → VIXChange (a Δ / change measure)
# PutCallSkew → VIXTermSlope (term-structure / skew-like)
# These are SEED INITIALIZERS only (the GP evolves away from them); the point is
# purely to keep non-transferable leaves out of the population, NOT to claim
# semantic identity. All targets are verified present in the scalar-only pool.
_SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS: Dict[str, str] = {
    "SessionReturn":   "SPXReturn3d",
    "OvernightGap":    "SPXReturn3d",
    "RawSpread":       "RealizedVol30m_5m",
    "RawSpread_5m":    "RealizedVol30m_5m",
    "SessionPosition": "RealizedVol30m_5m",
    "DeltaSpread1":    "VIXChange",
    "DeltaSpread5":    "VIXChange",
    "PutCallSkew":     "VIXTermSlope",
}

# Probe scalars + RegimeProb columns that have no acceptable scalar-only
# analog. The substitution helper raises ValueError if these appear in a
# seed tree — forcing the template author to design an explicit scalar-
# only seed rather than silently accepting a semantically-wrong rewrite.
#
# F1 fix expansion (2026-04-25): besides PredRegime, also forbid all probe
# scalars (PredRV15/30/Spread are in _SCALAR_ONLY_TERM_SUBSTITUTIONS so
# they're rewritten, not forbidden — kept that way for backwards-compat).
# The RegimeProb* columns are softmax outputs of the L1 regime head —
# they don't appear in the grammar terminal set under either arm
# (they're not in build_terminal_set()) so this list is defensive.
_SCALAR_ONLY_TERMS_FORBIDDEN: Tuple[str, ...] = (
    "PredRegime", "RegimeAboveLow", "RegimeIsHigh", "RegimeIsPremium",
    # new probes — forbidden in scalar-only (Condition A)
    "PredGammaAccel", "PredSmileConvexity", "PredJump", "PredFlowToxicity",
    # Defensive — these are not in build_terminal_set today, but if a
    # future template author adds them as a seed terminal, the substitution
    # path must reject them rather than silently mapping to a raw scalar.
    "RegimeProb0", "RegimeProb1", "RegimeProb2", "RegimeProb3",
)


def _build_scalar_only_terms_by_name() -> Dict[str, TermDef]:
    """Precomputed name→TermDef map for scalar-only terminals (used by seed
    substitution to construct replacement TermNodes)."""
    out: Dict[str, TermDef] = {}
    for t in build_scalar_only_terminal_set():
        out[t.name] = t
    return out


# Bypass scalars that are in the encoder's 147-variate input.
# These must be substituted out of seed trees for conditions B and C ().
# NOTE: ATM_IV_5m, RealizedVol30m_5m, RawSpread_5m are 5-bar rolling means
# of encoder-input scalars. Available in A (scalar baseline) but EXCLUDED from
# B/C (stricter ablation — including them lets GP shortcut around probes/embeddings).
_ENCODER_INPUT_SCALARS: frozenset = frozenset({
    "ATM_IV", "VIXSpot", "VIXTermSlope", "RealizedVol30m", "RawSpread",
    "DeltaSpread1", "DeltaSpread5",
    # Inter-day memory terminals (SPXReturn3d, VIXMean5d, RV5d, IVRVGap5d)
    # are NOT encoder inputs — the encoder processes only intraday 60-bar
    # windows. These capture multi-day context the encoder cannot see.
    # They are available to ALL conditions as complementary context.
})

# For probes-pure (B): replace encoder-input scalars with probe analogs
_PROBES_PURE_SUBSTITUTIONS: Dict[str, str] = {
    "ATM_IV": "PredRV15",           # IV → predicted RV (correlated)
    "RealizedVol30m": "PredRV30",   # RV → predicted RV
    "RawSpread": "PredSpread",      # spread → predicted spread
    "VIXSpot": "PredRV15",          # VIX → predicted RV (no VIX probe)
    "VIXTermSlope": "PredSpread",   # term slope → predicted spread (loose)
    "DeltaSpread1": "SessionReturn",# delta spread → session return (structural)
    "DeltaSpread5": "SessionReturn",
    # Inter-day terminals (SPXReturn3d, VIXMean5d, RV5d, IVRVGap5d) are NOT
    # substituted — they capture multi-day context the encoder cannot see
    # (encoder processes intraday 60-bar windows only). Available to ALL
    # conditions as complementary context. Removed from substitution map
    # 2026-05-19 to prevent ablation contamination.
}

# For emb-pure (C): replace encoder-input scalars with structural terminals
_EMB_PURE_SUBSTITUTIONS: Dict[str, str] = {
    # Clean ablation (#29): structural terminals removed from Condition C.
    # Only MinutesToClose, BarOfDay, SessionReturn remain. All encoder-input
    # scalars map to these 3 time/session primitives. This IS restrictive —
    # seed trees will be less diverse. But the point of Condition C is to
    # test whether EmbProj operators can replace market scalars, not to give
    # GP alternative market scalars to use instead.
    "ATM_IV": "SessionReturn",
    "RealizedVol30m": "SessionReturn",
    "RawSpread": "MinutesToClose",
    "VIXSpot": "BarOfDay",
    "VIXTermSlope": "SessionReturn",
    "DeltaSpread1": "SessionReturn",
    "DeltaSpread5": "MinutesToClose",
    # Inter-day terminals NOT substituted — available directly in Condition C
    # as complementary multi-day context (not encoder inputs).
}


def _preserve_term_tags(src: "TermNode", dst: "TermNode") -> "TermNode":
    """Carry dynamic semantic tags across a TermNode reconstruction in a tree
    walker, then return ``dst``.

    BLOCKER fix (2026-06-03 review): the seed-substitution walkers rebuild every
    terminal via ``TermNode(defn=, value=)``, which DROPS dynamic instance
    attributes. The VRP compound seed tags its threshold EphReals
    ``fold_normalized=True`` (authored directly in fold-normalized space) so
    ``recalibrate_seed_thresholds`` skips them. Substitution runs BEFORE
    recalibration in experiment._run_one_template, so without re-attaching the tag
    here the scalar-only / probes-only / emb-only conditions silently lose it and
    the threshold gets double-transformed (the exact corruption the tag prevents;
    real-l1 is unaffected because it never substitutes). Only nodes that pass
    THROUGH a walker unchanged (same defn/value) carry the tag — remapped/
    substituted nodes are different terminals and must not inherit it.
    """
    if getattr(src, "fold_normalized", False):
        dst.fold_normalized = True
    return dst


def _substitute_terminals(tree: Node, subs: Dict[str, str],
                          available: Dict[str, TermDef]) -> Node:
    """Generic seed tree substitution: replace terminal names per subs map.

    Three-tier resolution for each terminal:
    1. If in `subs` map and target exists in `available` → substitute.
    2. If in `_ENCODER_INPUT_SCALARS` but not in subs → structural fallback.
    3. If NOT in `available` at all (e.g., probe scalars in emb-pure) →
       structural fallback. This prevents terminals from conditions the
       seed was written for from leaking into a different condition.
    """
    _structural_fallback = available.get(
        "SessionReturn", available.get("MinutesToClose"))

    def _walk(n: Node) -> Node:
        if isinstance(n, TermNode):
            # Tier 1: explicit substitution
            if n.defn.name in subs:
                target = subs[n.defn.name]
                if target in available:
                    return TermNode(defn=available[target], value=None)
            # Tier 2: encoder-input scalar fallback
            if n.defn.name in _ENCODER_INPUT_SCALARS:
                if _structural_fallback is not None:
                    return TermNode(defn=_structural_fallback, value=None)
            # Tier 3: terminal not available in this condition's set
            # (catches probe scalars leaking into emb-pure, etc.)
            if n.defn.name not in available and n.defn.name not in (
                "EphReal", "EphInt",
                "CALL", "PUT", "LONG", "SHORT", "NEUTRAL",
                "LOW_VOL", "MID_VOL", "HIGH_VOL_STRESSED", "HIGH_VOL_PREMIUM",
            ):
                if _structural_fallback is not None:
                    return TermNode(defn=_structural_fallback, value=None)
            # Pass-through (unchanged terminal): preserve dynamic tags (fold_normalized).
            return _preserve_term_tags(n, TermNode(defn=n.defn, value=n.value))
        return FuncNode(defn=n.defn, children=[_walk(c) for c in n.children])
    return _walk(tree)


def substitute_for_probes_pure(tree: Node) -> Node:
    """Replace encoder-input scalars in seed trees for Condition B (probes-pure)."""
    available = {t.name: t for t in build_probes_only_terminal_set()}
    return _substitute_terminals(tree, _PROBES_PURE_SUBSTITUTIONS, available)


def substitute_for_emb_pure(tree: Node) -> Node:
    """Replace encoder-input scalars in seed trees for Condition C (emb-pure)."""
    available = {t.name: t for t in build_emb_only_terminal_set()}
    return _substitute_terminals(tree, _EMB_PURE_SUBSTITUTIONS, available)


def substitute_stripped_terminals_for_scalar_only(tree: Node) -> Node:
    """Walk a tree and replace any probe-scalar or typed-vector terminal
    with its scalar-only analog, so a template seed built for the full
    grammar can be used as an initial individual under the scalar-only
    condition without leaking L1-derived features into the grammar contract.

    Used by `experiment.run_experiment` when `condition == "scalar-only"`.
    A copy is returned; the input tree is not mutated.
    """
    scalar_terms = _build_scalar_only_terms_by_name()

    def _resolve_transferable(name: str) -> str:
        """Map a terminal name to a TRANSFERABLE scalar-only terminal name.

        BLOCKER-14: if `name` is one of the 8 QC-non-transferable scalars,
        substitute its type-safe transferable analog; otherwise return it
        unchanged. Applied to BOTH raw seed leaves AND the output of a probe
        substitution (belt-and-suspenders, so no rewrite re-introduces a
        non-transferable leaf).
        """
        return _SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS.get(name, name)

    def _walk(n: Node) -> Node:
        if isinstance(n, TermNode):
            # Probe-scalar substitution (semantically-defensible analogs)
            if n.defn.name in _SCALAR_ONLY_TERM_SUBSTITUTIONS:
                target = _resolve_transferable(
                    _SCALAR_ONLY_TERM_SUBSTITUTIONS[n.defn.name])
                new_defn = scalar_terms[target]
                return TermNode(defn=new_defn, value=None)
            # BLOCKER-14 (2026-06-01 data audit): QC-NON-TRANSFERABLE scalar leaf →
            # type-safe transferable REAL analog. Without this, Level-B seeds (all
            # gate on SessionReturn) leaked a non-transferable terminal into the
            # seeded fraction of the population. Mechanical REAL→REAL replacement
            # from the scalar-only pool; see _SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS.
            if n.defn.name in _SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS:
                _tgt = _SCALAR_ONLY_NONTRANSFERABLE_SUBSTITUTIONS[n.defn.name]
                return TermNode(defn=scalar_terms[_tgt], value=None)
            # Forbidden probes (no semantic analog — would be a silent
            # collapse if mapped). Raise so the template author is forced
            # to design an explicit scalar-only seed.
            if n.defn.name in _SCALAR_ONLY_TERMS_FORBIDDEN:
                raise ValueError(
                    f"seed tree references {n.defn.name!r}, which has no "
                    f"acceptable scalar-only analog (PredRegime is a 4-class "
                    f"L1 inference with no raw-scalar equivalent). Define "
                    f"an explicit scalar-only seed for this template or "
                    f"exclude it from the scalar-only condition."
                )
            # Typed-vector terminal → has no scalar analog. Seed trees for
            # the 8 templates don't use typed-vector terminals directly
            # (they're only reachable via operators), but guard against
            # future seed drift by raising explicitly.
            if n.defn.ret_type in EMB_TYPES:
                raise ValueError(
                    f"cannot substitute typed-vector terminal {n.defn.name!r} "
                    f"in scalar-only condition — no scalar analog exists"
                )
            # Everything else (scalars, regime/side literals, ephemerals)
            # is unchanged — they already pass the scalar-only grammar.
            # Pass-through: preserve dynamic tags (fold_normalized VRP thresholds).
            return _preserve_term_tags(n, TermNode(defn=n.defn, value=n.value))
        # FuncNode — recurse into children. A FuncNode whose defn is in
        # EMB_OPERATORS would not appear in a seed tree; same guard.
        if n.defn in EMB_OPERATORS:
            raise ValueError(
                f"cannot substitute typed-vector operator {n.defn.name!r} "
                f"in scalar-only condition"
            )
        return FuncNode(defn=n.defn, children=[_walk(c) for c in n.children])

    return _walk(tree)

# ---------------------------------------------------------------------------
# Grammar class
# ---------------------------------------------------------------------------

class Grammar:
    """Typed function/terminal registry with tree generation and genetic ops."""

    def __init__(
        self,
        *,  # keyword-only to prevent positional arg order swap (TermDef/FuncDef)
        functions: Sequence[FuncDef] | None = None,
        terminals: Sequence[TermDef] | None = None,
        max_depth: int = 6,
        max_nodes: int = 25,
    ):
        self.functions = list(functions or ALL_FUNCTIONS)
        # Exclude terminals that do NOT transfer proxy->QC from the SAMPLING pool
        # (see _QC_NONTRANSFERABLE_SCALARS). The parser (_TERM_LOOKUP, built from the
        # unfiltered build_terminal_set()) still knows them, so existing strategies
        # and tests that reference them parse fine — only random tree generation
        # avoids them. Re-enable by emptying _QC_NONTRANSFERABLE_SCALARS.
        self.terminals = [t for t in (terminals or build_terminal_set())
                          if t.name not in _QC_NONTRANSFERABLE_SCALARS]
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        self._funcs_by_ret: Dict[GType, List[FuncDef]] = {}
        for f in self.functions:
            self._funcs_by_ret.setdefault(f.ret_type, []).append(f)
        self._terms_by_ret: Dict[GType, List[TermDef]] = {}
        for t in self.terminals:
            self._terms_by_ret.setdefault(t.ret_type, []).append(t)

        # Minimum depth to produce each type (Bool=1 because no Bool terminal)
        self._min_depth = self._compute_min_depths()

    def _compute_min_depths(self) -> Dict[GType, int]:
        known: Dict[GType, int] = {t: 0 for t in self._terms_by_ret}
        changed = True
        while changed:
            changed = False
            for f in self.functions:
                if all(at in known for at in f.arg_types):
                    cost = 1 + max((known[at] for at in f.arg_types), default=0)
                    if f.ret_type not in known or cost < known[f.ret_type]:
                        known[f.ret_type] = cost
                        changed = True
        return known

    # -- helpers -----------------------------------------------------------------

    def _has_term(self, t: GType) -> bool:
        return bool(self._terms_by_ret.get(t))

    def _random_term(self, t: GType) -> TermNode:
        return TermNode(defn=random.choice(self._terms_by_ret[t]))

    def _feasible_func(self, ret: GType, budget: int) -> Optional[FuncDef]:
        """Random function for *ret* whose args fit within *budget - 1* depth.

        F2 fix (2026-04-25, HIGH): operator-class-stratified sampling.
        Stage 1: pick a class uniformly from the classes that have ≥1
        feasible function. Stage 2: pick a function within that class
        uniformly. Equalizes per-class selection density across grammar
        sizes so the 84 EmbProj operators do not drown out the 5 arithmetic
        operators. Per-class membership is defined in OPERATOR_CLASSES.

        For EmbLag_* operators, the first arg must be a TermNode — so we
        only admit an EmbLag candidate if a typed-vector terminal of the
        required EMB type exists in the terminal set. Without this guard,
        `_feasible_func` would happily return an EmbLag operator even when
        the grammar has no valid first-arg leaf, and tree construction
        would fall back to producing a non-terminal first child, silently
        violating the constraint enforced by `validate()`.
        """
        # Build per-class feasible bucket. Each class collects only the
        # functions in `self.functions` (so a Grammar restricted to
        # SCALAR_ONLY_FUNCTIONS will have empty emb_*/regime classes).
        by_class: Dict[str, List[FuncDef]] = {}
        for f in self._funcs_by_ret.get(ret, []):
            if not all(self._min_depth.get(a, 99) < budget for a in f.arg_types):
                continue
            if is_emblag_funcdef(f) and not self._has_term(f.arg_types[0]):
                continue  # no typed-vector leaf available → EmbLag unviable
            cls = _FUNC_TO_CLASS.get(f)
            if cls is None:
                # Defensive: a future operator added without OPERATOR_CLASSES
                # entry would silently fall out of the stratified sampler.
                # Use a synthetic class name so it remains reachable but is
                # surfaceable to debugging.
                cls = "_unclassified"
            by_class.setdefault(cls, []).append(f)
        if not by_class:
            return None
        chosen_class = random.choice(sorted(by_class.keys()))
        return random.choice(by_class[chosen_class])

    def _make_children_for(self, fd: FuncDef, md: int,
                            gen_fn: Callable[[GType, int], Node]) -> List[Node]:
        """Build the children list for a FuncNode, respecting operator-level
        argument constraints. Currently only EmbLag_* is constrained: its
        first argument must be a TermNode (typed-vector leaf). Every other
        operator delegates to `gen_fn` (grow/full) uniformly.

        Degenerate-composition guard: for binary same-type embedding operators
        (EmbSub_*, EmbCos_*), if both children are the same terminal, regenerate
        the second child. EmbSub(T,T) = zero-vector and EmbCos(T,T) = 1.0 are
        constant — GP search budget wasted on a structural no-op.
        """
        if is_emblag_funcdef(fd):
            # child[0]: forced TermNode of the typed-vector type. The
            # caller guarantees `_has_term(fd.arg_types[0])` via
            # `_feasible_func`'s filter, so this is always safe.
            first = self._random_term(fd.arg_types[0])
            rest = [gen_fn(a, md - 1) for a in fd.arg_types[1:]]
            return [first] + rest
        children = [gen_fn(a, md - 1) for a in fd.arg_types]
        # Degenerate-composition guard for EmbSub_* and EmbCos_*:
        # If both children are the SAME terminal, the result is a constant
        # (EmbSub=zero, EmbCos=1.0). When only one terminal of that type
        # exists (which is the case for all EMB types — each has exactly one
        # terminal), wrap the second child in EmbLag to create a meaningful
        # "current vs lagged" comparison. This is the most common useful
        # pattern for same-type binary embedding operators.
        if (fd.name.startswith("EmbSub_") or fd.name.startswith("EmbCos_")):
            if (len(children) == 2
                    and isinstance(children[0], TermNode)
                    and isinstance(children[1], TermNode)
                    and children[0].name == children[1].name):
                # Try to get a different terminal first
                emb_type = fd.arg_types[1]
                found_different = False
                for _ in range(5):
                    alt = self._random_term(emb_type)
                    if alt.name != children[0].name:
                        children[1] = alt
                        found_different = True
                        break
                if not found_different:
                    # Only one terminal of this type exists — wrap child[1]
                    # in EmbLag to create "current vs N-bars-ago" comparison.
                    emb_lag_fd = next(
                        (f for f in self.functions
                         if f.name == f"EmbLag_{children[0].name}"),
                        None,
                    )
                    if emb_lag_fd is not None:
                        lag_term = TermNode(
                            defn=TermDef("EphInt", GType.INT),
                            value=random.choice(_LAG_CHOICES),
                        )
                        children[1] = FuncNode(
                            defn=emb_lag_fd,
                            children=[self._random_term(emb_type), lag_term],
                        )
        return children

    # -- generation --------------------------------------------------------------

    def grow(self, ret: GType, max_d: int | None = None) -> Node:
        """Grow: randomly pick terminal or function at each level."""
        md = max_d if max_d is not None else self.max_depth
        can_term = self._has_term(ret)
        fd = self._feasible_func(ret, md) if md > 0 else None
        if fd is None:
            return self._random_term(ret) if can_term else self._force_func(ret, md)
        if can_term and random.random() < 0.5:
            return self._random_term(ret)
        return FuncNode(defn=fd, children=self._make_children_for(fd, md, self.grow))

    def full(self, ret: GType, max_d: int | None = None) -> Node:
        """Full: always expand with functions until depth exhausted."""
        md = max_d if max_d is not None else self.max_depth
        fd = self._feasible_func(ret, md) if md > 0 else None
        if fd is None:
            return self._random_term(ret) if self._has_term(ret) else self._force_func(ret, md)
        return FuncNode(defn=fd, children=self._make_children_for(fd, md, self.full))

    def _force_func(self, ret: GType, md: int) -> Node:
        """Produce a minimal-depth subtree for types without terminals (e.g. Bool)."""
        extra = self._min_depth.get(ret, 1)
        effective_md = max(md, extra) + 1
        fd = self._feasible_func(ret, effective_md)
        if fd is None:
            raise ValueError(f"Cannot produce type {ret}")
        return FuncNode(defn=fd,
                         children=self._make_children_for(fd, effective_md, self.grow))

    def ramped_half_and_half(self, ret: GType, pop_size: int, min_depth: int = 2) -> List[Node]:
        lo = max(min_depth, self._min_depth.get(ret, 0))
        depths = list(range(lo, self.max_depth + 1))
        per = max(1, pop_size // len(depths))
        trees = [self._gen_capped(ret, d) for d in depths for _ in range(per)]
        while len(trees) < pop_size:
            trees.append(self._gen_capped(ret, random.choice(depths)))
        return trees[:pop_size]

    def _gen_capped(self, ret: GType, md: int) -> Node:
        for _ in range(10):
            t = (self.full if random.random() < 0.5 else self.grow)(ret, md)
            if node_count(t) <= self.max_nodes and depth(t) <= self.max_depth:
                return t
        # S3 fix: fallback with retry (depth-2 grow can rarely exceed max_nodes
        # with arity-3 functions). Use max_d=1 as last resort (max 4 nodes).
        for _fb_attempt in range(5):
            _fb_depth = max(self._min_depth.get(ret, 0), 2) if _fb_attempt < 3 else 1
            fallback = self.grow(ret, max_d=_fb_depth)
            if depth(fallback) <= self.max_depth and node_count(fallback) <= self.max_nodes:
                return fallback
        # Absolute last resort: single terminal
        return self._random_term(ret) if self._has_term(ret) else fallback

    # -- validation --------------------------------------------------------------

    def validate(self, node: Node) -> bool:
        return validate(node)

    # -- crossover ---------------------------------------------------------------

    def crossover(self, pa: Node, pb: Node) -> Tuple[Node, Node]:
        """Type-safe subtree crossover.  Non-destructive, cap-respecting.

        Also respects the EmbLag first-arg-is-terminal constraint: a
        candidate pairing that would produce an EmbLag_* node with a
        non-TermNode first child is rejected and another pairing is tried.
        """
        na = all_nodes_indexed(pa)
        nb = all_nodes_indexed(pb)
        b_idx: Dict[GType, List[Tuple[Node, Tuple[int, ...]]]] = {}
        for n, _, p in nb:
            b_idx.setdefault(n.ret_type, []).append((n, p))
        random.shuffle(na)
        for node_a, _, pa_path in na:
            matches = b_idx.get(node_a.ret_type)
            if not matches:
                continue
            # Try multiple B matches per A node (up to 5) instead of just 1.
            # With 15-node trees, most single-attempt swaps violate the cap
            # because only equal-size subtrees can swap without growing past
            # max_nodes. Trying multiple partners dramatically improves the
            # success rate. (Langdon & Poli 2002: crossover effectiveness)
            _shuffled_matches = list(matches)
            random.shuffle(_shuffled_matches)
            for node_b, pb_path in _shuffled_matches[:5]:
                if _crossover_would_violate_emblag(pa, pa_path, node_b) or \
                   _crossover_would_violate_emblag(pb, pb_path, node_a):
                    continue
                c1 = _replace_at_path(deep_copy(pa), pa_path, deep_copy(node_b))
                c2 = _replace_at_path(deep_copy(pb), pb_path, deep_copy(node_a))
                if (depth(c1) <= self.max_depth and depth(c2) <= self.max_depth
                        and node_count(c1) <= self.max_nodes
                        and node_count(c2) <= self.max_nodes):
                    return c1, c2
        return deep_copy(pa), deep_copy(pb)

    # -- mutation ----------------------------------------------------------------

    def point_mutate(self, tree: Node, p: float = 0.1) -> Node:
        """Replace each node with prob *p* by another of same return type."""
        return self._pm(deep_copy(tree), p, parent_fd=None, arg_idx=None)

    def _pm(self, n: Node, p: float,
             parent_fd: Optional[FuncDef], arg_idx: Optional[int]) -> Node:
        # EmbLag first-arg slot: must remain a TermNode of the same EMB
        # type. Any mutation here is restricted to picking another terminal.
        in_emblag_first_slot = (
            parent_fd is not None and arg_idx == 0 and is_emblag_funcdef(parent_fd)
        )
        if isinstance(n, TermNode):
            if random.random() >= p or not self._has_term(n.ret_type):
                return n
            # EphReal perturbation: Gaussian noise (sigma=0.3) instead of
            # uniform resample. Preserves learned thresholds while allowing
            # gradual refinement. 30% chance of swapping to a different
            # terminal entirely (exploration), 70% perturbation (exploitation).
            if n.name == "EphReal" and random.random() < 0.70:
                new_val = n.value + random.gauss(0, 0.3)
                new_val = max(-3.0, min(3.0, new_val))  # clamp to EphReal range
                return TermNode(defn=n.defn, value=round(new_val, 4))
            return self._random_term(n.ret_type)
        if random.random() < p and not in_emblag_first_slot:
            fd = self._feasible_func(n.ret_type, 3)
            if fd:
                return FuncNode(defn=fd,
                                 children=self._make_children_for(fd, 3, self.grow))
        n.children = [self._pm(c, p, parent_fd=n.defn, arg_idx=i)
                      for i, c in enumerate(n.children)]
        return n

    def subtree_mutate(self, tree: Node, max_d: int = 3) -> Node:
        """Replace a random non-root subtree with a fresh random tree.

        Respects EmbLag first-arg-is-terminal: if the randomly-chosen
        target is the first child of an EmbLag_* node, replacement is
        restricted to a fresh TermNode of the required EMB type rather
        than an arbitrary subtree.
        """
        ns = all_nodes_indexed(tree)
        cands = [(n, p) for n, _, p in ns if p]
        if not cands:
            return self.grow(tree.ret_type, max_d=max_d)
        tgt, path = random.choice(cands)
        # If the target is the first child of an EmbLag_* parent, force the
        # replacement to be a TermNode of the same EMB type.
        if _path_is_emblag_first_child(tree, path):
            repl = self._random_term(tgt.ret_type)
        else:
            repl = self.grow(tgt.ret_type, max_d)
        new = _replace_at_path(deep_copy(tree), path, repl)
        return new if node_count(new) <= self.max_nodes else deep_copy(tree)

    # -- serialization -----------------------------------------------------------

    @staticmethod
    def to_str(node: Node) -> str:
        return to_str(node)


# ---------------------------------------------------------------------------
# Pre-registration drift guard
# ---------------------------------------------------------------------------

# Path to the H2 pre-registration YAML. Kept module-level so callers
# (experiment.py, unit tests) can point the assertion at the same file.
_PREREG_YAML_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "methodology" / "(internal doc)"
)

# Regex to extract "(N functions, M terminals)" from the real-l1 grammar
# description string. The v1 YAML commits to "full typed-vector grammar
# (46 functions, 29 terminals)" and the H2 freeze tag binds those counts.
# Matching lexically (not structurally) because the YAML uses a freeform
# string field, not a structured count; keeps this function robust to
# whitespace drift in the description.
_PREREG_GRAMMAR_COUNTS_RE = re.compile(
    r"\(\s*(\d+)\s+functions?\s*,\s*(\d+)\s+terminals?\s*\)",
    re.IGNORECASE,
)


def _read_preregistration_grammar_counts(
    yaml_path: Optional[Path] = None,
) -> Tuple[int, int]:
    """Parse the pre-registered (n_functions, n_terminals) from the H2 YAML.

    Does a tiny hand-rolled YAML probe (avoids importing PyYAML at module
    load time since grammar.py is imported widely). We only need the
    `conditions.real-l1.grammar` string — a single-line value in the
    committed YAML — which we extract by line scan + regex.

    Raises ValueError if the expected field is missing or unparseable.
    """
    path = yaml_path or _PREREG_YAML_PATH
    text = path.read_text()
    # The YAML has a top-level `conditions:` block with a `real-l1:` entry
    # and a single-line `grammar: full ... (46 functions, 29 terminals)`.
    # Scan for the first occurrence; the amendments block uses the same
    # regex-matchable format but downstream so the first hit is the live
    # committed count.
    in_real_l1 = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("real-l1:"):
            in_real_l1 = True
            continue
        if in_real_l1 and stripped.startswith("grammar:"):
            m = _PREREG_GRAMMAR_COUNTS_RE.search(stripped)
            if m is None:
                raise ValueError(
                    f"found conditions.real-l1.grammar in {path} but could "
                    f"not parse '(N functions, M terminals)': {stripped!r}"
                )
            return int(m.group(1)), int(m.group(2))
        # A sibling key at the same indent breaks out of real-l1
        if in_real_l1 and stripped and not line.startswith((" ", "\t")):
            in_real_l1 = False
    raise ValueError(
        f"did not find conditions.real-l1.grammar in {path}; "
        f"cannot verify grammar-size invariant"
    )


def assert_grammar_matches_preregistration(
    yaml_path: Optional[Path] = None,
) -> None:
    """Fail fast if the live grammar has drifted from the pre-registered
    H2 counts. Called at experiment init so drift surfaces as a clear
    runtime error *before* any GP evolution begins.

    The pre-registration (``(internal doc)``) commits to
    46 functions and 29 terminals in the live grammar. Either changing
    requires an explicit amendment entry in the YAML AND bumping the
    grammar-signature section of results.json.

    Args:
        yaml_path: optional override for the YAML location (tests use this
            to point at a fixture). Defaults to ``(internal doc)``.

    Raises:
        AssertionError: if either count mismatches.
        ValueError: if the YAML is missing the expected field (bug surface
            we want to fail loud on).
    """
    expected_funcs, expected_terms = _read_preregistration_grammar_counts(yaml_path)
    actual_funcs = len(ALL_FUNCTIONS)
    actual_terms = len(build_terminal_set())
    if actual_funcs != expected_funcs or actual_terms != expected_terms:
        raise AssertionError(
            "grammar drifted from H2 pre-registration:\n"
            f"  pre-registered: {expected_funcs} functions, {expected_terms} terminals\n"
            f"  live grammar:   {actual_funcs} functions, {actual_terms} terminals\n"
            f"Either revert the grammar change or file an amendment in "
            f"(internal doc) (and re-tag h2-preregistration-v{{N}})."
        )
