"""Unit tests for commutative-canonical dedup of GP individuals.

`canonical_key` (layer2.grammar) returns a structural signature invariant to
the argument order of genuinely commutative operators and the GT<->LT mirror.
This stops the Pareto front from filling with semantically-identical clones
(a known diversity defect: dedup on raw `to_str` treats AND(a,b) and AND(b,a)
as distinct, so a 24-member front is really ~8 skeletons).

These tests build real grammar nodes via `from_sexpr` and assert:
  * commutative twins collapse (AND/OR/Add/Mul/RegimeIs/EmbCos_*),
  * the comparison mirror collapses (GT(x,y) == LT(y,x)),
  * non-commutative ops are preserved (Sub/Div/CrossAbove/EmbSub/GT(y,x)),
  * a population dedup pass over commutative-equivalent twins collapses them,
  * genuinely distinct trees stay distinct.
"""
import numpy as np
import pandas as pd
import pytest

from layer2.grammar import canonical_key, from_sexpr, to_str
from layer2.gp_engine import Individual


def _k(sexpr: str, **kw) -> str:
    """canonical_key of a tree parsed from its s-expression."""
    return canonical_key(from_sexpr(sexpr), **kw)


# ---------------------------------------------------------------------------
# Commutative operators collapse
# ---------------------------------------------------------------------------

def test_and_commutes():
    assert _k("AND(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0)))") == \
           _k("AND(LT(VIXSpot, EphReal(20.0)), GT(ATM_IV, EphReal(0.18)))")


def test_or_commutes():
    assert _k("OR(GT(ATM_IV, EphReal(0.18)), GT(VIXSpot, EphReal(20.0)))") == \
           _k("OR(GT(VIXSpot, EphReal(20.0)), GT(ATM_IV, EphReal(0.18)))")


def test_add_commutes():
    assert _k("Add(ATM_IV, VIXSpot)") == _k("Add(VIXSpot, ATM_IV)")


def test_mul_commutes():
    assert _k("Mul(ATM_IV, VIXSpot)") == _k("Mul(VIXSpot, ATM_IV)")


def test_regime_is_commutes():
    # RegimeIs evaluates `a == b` — symmetric equality.
    assert _k("RegimeIs(LOW_VOL, HIGH_VOL_PREMIUM)") == \
           _k("RegimeIs(HIGH_VOL_PREMIUM, LOW_VOL)")


def test_embcos_commutes():
    # Cosine similarity dot(a,b)/(|a||b|) is symmetric in its two vectors.
    a = "EmbCos_EMB_GRID(EMB_GRID, EmbSub_EMB_GRID(EMB_GRID, EMB_GRID))"
    b = "EmbCos_EMB_GRID(EmbSub_EMB_GRID(EMB_GRID, EMB_GRID), EMB_GRID)"
    assert _k(a) == _k(b)


# ---------------------------------------------------------------------------
# Comparison mirror: GT(x, y) == LT(y, x)
# ---------------------------------------------------------------------------

def test_gt_lt_mirror():
    assert _k("GT(ATM_IV, VIXSpot)") == _k("LT(VIXSpot, ATM_IV)")


def test_gt_lt_mirror_with_eph():
    assert _k("GT(ATM_IV, EphReal(0.18))") == _k("LT(EphReal(0.18), ATM_IV)")


def test_gt_not_commutative():
    # The mirror is GT(x,y)==LT(y,x), NOT GT(x,y)==GT(y,x).
    assert _k("GT(ATM_IV, VIXSpot)") != _k("GT(VIXSpot, ATM_IV)")


def test_lt_pair_distinct_when_args_swapped():
    assert _k("LT(ATM_IV, VIXSpot)") != _k("LT(VIXSpot, ATM_IV)")


# ---------------------------------------------------------------------------
# Non-commutative operators are preserved
# ---------------------------------------------------------------------------

def test_sub_not_commutative():
    assert _k("Sub(ATM_IV, VIXSpot)") != _k("Sub(VIXSpot, ATM_IV)")


def test_div_not_commutative():
    assert _k("Div(ATM_IV, VIXSpot)") != _k("Div(VIXSpot, ATM_IV)")


def test_crossabove_not_commutative():
    assert _k("CrossAbove(ATM_IV, VIXSpot)") != _k("CrossAbove(VIXSpot, ATM_IV)")


def test_embsub_not_commutative():
    # Vector difference a - b is direction-sensitive.
    a = "EmbSub_EMB_SPX(EMB_SPX, EmbLag_EMB_SPX(EMB_SPX, EphInt(5)))"
    b = "EmbSub_EMB_SPX(EmbLag_EMB_SPX(EMB_SPX, EphInt(5)), EMB_SPX)"
    assert _k(a) != _k(b)


def test_ifthenelse_branches_preserved():
    # IfThenElse(cond, a, b) != IfThenElse(cond, b, a) — branches are positional.
    a = "IfThenElse(GT(ATM_IV, EphReal(0.2)), VIXSpot, ATM_IV)"
    b = "IfThenElse(GT(ATM_IV, EphReal(0.2)), ATM_IV, VIXSpot)"
    assert _k(a) != _k(b)


# ---------------------------------------------------------------------------
# Recursion + nesting
# ---------------------------------------------------------------------------

def test_nested_commutative_recurses():
    # Reorder at BOTH levels: outer AND swapped, and an inner OR swapped.
    a = ("AND(OR(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0))), "
         "GT(SessionReturn, EphReal(0.0)))")
    b = ("AND(GT(SessionReturn, EphReal(0.0)), "
         "OR(LT(VIXSpot, EphReal(20.0)), GT(ATM_IV, EphReal(0.18))))")
    assert _k(a) == _k(b)


def test_distinct_trees_stay_distinct():
    # Different terminals / thresholds -> different keys.
    assert _k("GT(ATM_IV, EphReal(0.18))") != _k("GT(ATM_IV, EphReal(0.30))")
    assert _k("GT(ATM_IV, VIXSpot)") != _k("GT(SessionReturn, VIXSpot)")
    assert _k("AND(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0)))") != \
           _k("OR(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0)))")


# ---------------------------------------------------------------------------
# Ephemeral rounding tolerance (mirrors gp_engine phenotypic dedup)
# ---------------------------------------------------------------------------

def test_eph_round_collapses_jitter():
    assert _k("GT(ATM_IV, EphReal(0.183))", _eph_round=1) == \
           _k("GT(ATM_IV, EphReal(0.178))", _eph_round=1)


def test_no_eph_round_keeps_distinct():
    assert _k("GT(ATM_IV, EphReal(0.183))") != _k("GT(ATM_IV, EphReal(0.178))")


# ---------------------------------------------------------------------------
# Population-level dedup pass collapses commutative twins
# ---------------------------------------------------------------------------

def _make_ind(entry_sexpr: str, exit_sexpr: str = "GT(MinutesToClose, EphReal(5.0))") \
        -> Individual:
    return Individual(
        entry_tree=from_sexpr(entry_sexpr),
        exit_tree=from_sexpr(exit_sexpr),
        size_tree=from_sexpr("EphReal(0.5)"),
        template_name="iron_condor",
    )


def test_population_dedup_collapses_commutative_twins():
    """A seen-set keyed on canonical_key collapses commutative-equivalent
    individuals that raw to_str would keep as distinct."""
    pop = [
        _make_ind("AND(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0)))"),
        _make_ind("AND(LT(VIXSpot, EphReal(20.0)), GT(ATM_IV, EphReal(0.18)))"),  # twin
        _make_ind("GT(ATM_IV, VIXSpot)"),
        _make_ind("LT(VIXSpot, ATM_IV)"),                                        # twin
        _make_ind("OR(GT(SessionReturn, EphReal(0.0)), GT(VIXSpot, EphReal(18.0)))"),
        _make_ind("OR(GT(VIXSpot, EphReal(18.0)), GT(SessionReturn, EphReal(0.0)))"),  # twin
        _make_ind("GT(SessionReturn, EphReal(0.0))"),  # genuinely unique
    ]
    # Raw to_str dedup (what the old code effectively did) under-collapses.
    raw_keys = {to_str(ind.entry_tree) for ind in pop}
    canon_keys = {canonical_key(ind.entry_tree) for ind in pop}
    assert len(raw_keys) == 7          # all distinct by raw string
    assert len(canon_keys) == 4        # 3 twin-pairs collapse, 1 unique

    # Simulate the gp_engine seen-set dedup pass.
    seen: set = set()
    deduped = []
    for ind in pop:
        key = canonical_key(ind.entry_tree) + "|" + canonical_key(ind.exit_tree)
        if key not in seen:
            seen.add(key)
            deduped.append(ind)
    assert len(deduped) == 4


def test_signature_uses_canonical_key():
    """Individual.signature() collapses commutative-equivalent individuals."""
    a = _make_ind("AND(GT(ATM_IV, EphReal(0.18)), LT(VIXSpot, EphReal(20.0)))")
    b = _make_ind("AND(LT(VIXSpot, EphReal(20.0)), GT(ATM_IV, EphReal(0.18)))")
    assert a.signature() == b.signature()
    c = _make_ind("GT(SessionReturn, EphReal(0.0))")
    assert a.signature() != c.signature()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_key_is_deterministic():
    s = "AND(GT(ATM_IV, EphReal(0.18)), OR(LT(VIXSpot, EphReal(20.0)), GT(SessionReturn, EphReal(0.0))))"
    assert _k(s) == _k(s)
