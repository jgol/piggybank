"""Unit tests for layer2.templates — the 13 strategy templates.

Validates:
  - All 13 templates instantiate without error
  - Each template's seed trees pass grammar validation
  - Leg counts match D87/D88 commitments
  - Credit/debit classification is correct per template
  - Direction labels are accurate
  - No naked / no single-leg (per D87 risk-management constraint)
  - Strike-width variants have correct deltas
  - Backward-compatible aliases work

Encoder-augmented _enc templates removed (2026-05-11) for ablation fairness.
"""
import pytest

from layer2.grammar import GType, validate
from layer2.templates import (
    Leg, Template,
    # Standard (renamed from originals)
    iron_condor_standard, iron_butterfly_standard,
    bull_put_credit_standard, bear_call_credit_standard,
    # Narrow variants
    iron_condor_narrow, iron_butterfly_narrow,
    bull_put_credit_narrow, bear_call_credit_narrow,
    # Wide variants
    iron_condor_wide, bull_put_credit_wide, bear_call_credit_wide,
    # Debit
    bull_call_debit, bear_put_debit,
    # Backward-compatible aliases
    iron_condor, iron_butterfly,
    bull_put_credit, bear_call_credit,
    # Registry
    TEMPLATE_FACTORIES, all_templates, template_by_name,
)


# ---------------------------------------------------------------------------
# Leg validation
# ---------------------------------------------------------------------------

def test_leg_call_positive_delta():
    """Call legs must have positive delta_target."""
    Leg("call", +0.15, qty_sign=-1)
    with pytest.raises(ValueError, match="negative delta"):
        Leg("call", -0.15, qty_sign=-1)


def test_leg_put_negative_delta():
    """Put legs must have negative delta_target."""
    Leg("put", -0.15, qty_sign=-1)
    with pytest.raises(ValueError, match="positive delta"):
        Leg("put", +0.15, qty_sign=-1)


def test_leg_qty_sign_must_be_pm1():
    Leg("call", +0.15, qty_sign=+1)
    Leg("call", +0.15, qty_sign=-1)
    with pytest.raises(ValueError, match="qty_sign"):
        Leg("call", +0.15, qty_sign=0)
    with pytest.raises(ValueError, match="qty_sign"):
        Leg("call", +0.15, qty_sign=2)


def test_leg_ratio_minimum_1():
    Leg("call", +0.15, qty_sign=-1, ratio=1)
    Leg("call", +0.15, qty_sign=-1, ratio=2)
    with pytest.raises(ValueError, match="ratio"):
        Leg("call", +0.15, qty_sign=-1, ratio=0)


# ---------------------------------------------------------------------------
# All templates instantiate cleanly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", TEMPLATE_FACTORIES)
def test_template_factory_runs(factory):
    """Each template factory produces a valid Template without error."""
    t = factory()
    assert isinstance(t, Template)
    assert t.name
    assert t.description
    assert t.direction in ("neutral", "bullish", "bearish")
    assert t.n_legs >= 2  # multi-leg only constraint


def test_all_templates_returns_thirteen():
    templates = all_templates()
    assert len(templates) == 13
    names = {t.name for t in templates}
    expected = {
        # Standard credit
        "iron_condor_standard", "iron_butterfly_standard",
        "bull_put_credit_standard", "bear_call_credit_standard",
        # Narrow credit
        "iron_condor_narrow", "iron_butterfly_narrow",
        "bull_put_credit_narrow", "bear_call_credit_narrow",
        # Wide credit
        "iron_condor_wide", "bull_put_credit_wide", "bear_call_credit_wide",
        # Debit
        "bull_call_debit", "bear_put_debit",
    }
    assert names == expected


def test_template_by_name_lookup():
    t = template_by_name("iron_condor_standard")
    assert t.name == "iron_condor_standard"
    assert t.n_legs == 4
    with pytest.raises(KeyError):
        template_by_name("naked_call")  # would violate anyway


def test_enc_templates_removed():
    """Encoder-augmented _enc templates must not exist (removed 2026-05-11)."""
    names = {t.name for t in all_templates()}
    for name in names:
        assert not name.endswith("_enc"), f"_enc template {name} still exists"
    with pytest.raises(KeyError):
        template_by_name("iron_condor_standard_enc")


# ---------------------------------------------------------------------------
# / invariants — no naked, no single-leg, all defined-risk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", TEMPLATE_FACTORIES)
def test_no_single_leg_templates(factory):
    """D87 explicitly excludes single-leg structures (Coval & Shumway 2001)."""
    t = factory()
    assert t.n_legs >= 2, f"{t.name} has {t.n_legs} leg(s); single-leg excluded per D87"


@pytest.mark.parametrize("factory", TEMPLATE_FACTORIES)
def test_template_seed_trees_validate(factory):
    """Each template's seed grammar trees must pass type validation."""
    t = factory()
    assert validate(t.entry_seed), f"{t.name} entry_seed fails validation"
    assert validate(t.exit_seed), f"{t.name} exit_seed fails validation"
    assert validate(t.size_seed), f"{t.name} size_seed fails validation"
    assert t.entry_seed.ret_type == GType.BOOL
    assert t.exit_seed.ret_type == GType.BOOL
    assert t.size_seed.ret_type == GType.REAL


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def test_backward_compatible_aliases():
    """Old names (without _standard) still work as aliases."""
    assert iron_condor is iron_condor_standard
    assert iron_butterfly is iron_butterfly_standard
    assert bull_put_credit is bull_put_credit_standard
    assert bear_call_credit is bear_call_credit_standard


# ---------------------------------------------------------------------------
# Per-template structural checks — Standard width
# ---------------------------------------------------------------------------

def test_iron_condor_standard_structure():
    t = iron_condor_standard()
    assert t.n_legs == 4
    assert t.direction == "neutral"
    assert t.is_credit  # net credit at entry
    # Two short OTM legs + two long further-OTM wings
    short_legs = [leg for leg in t.legs if leg.qty_sign == -1]
    long_legs = [leg for leg in t.legs if leg.qty_sign == +1]
    assert len(short_legs) == 2
    assert len(long_legs) == 2
    # One call + one put among shorts (defined-risk both sides)
    short_types = sorted(leg.option_type for leg in short_legs)
    assert short_types == ["call", "put"]
    # Standard deltas: 25d short, 10d long
    short_deltas = sorted(abs(leg.delta_target) for leg in short_legs)
    assert short_deltas == [0.25, 0.25]
    long_deltas = sorted(abs(leg.delta_target) for leg in long_legs)
    assert long_deltas == [0.10, 0.10]


def test_iron_butterfly_standard_structure():
    t = iron_butterfly_standard()
    assert t.n_legs == 4
    assert t.direction == "neutral"
    assert t.is_credit
    # ATM short legs (delta magnitude near 0.50)
    atm_shorts = [leg for leg in t.legs
                  if leg.qty_sign == -1 and abs(leg.delta_target) >= 0.40]
    assert len(atm_shorts) == 2  # short ATM call + short ATM put
    # Standard wing: 10d
    long_legs = [leg for leg in t.legs if leg.qty_sign == +1]
    assert all(abs(leg.delta_target) == 0.10 for leg in long_legs)


def test_bull_put_credit_standard_structure():
    t = bull_put_credit_standard()
    assert t.n_legs == 2
    assert t.direction == "bullish"
    assert t.is_credit
    # Both legs are puts; one short OTM + one long further-OTM
    assert all(leg.option_type == "put" for leg in t.legs)
    short_put = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_put = next(leg for leg in t.legs if leg.qty_sign == +1)
    # Short put closer to ATM (less negative delta), long put further OTM
    assert abs(short_put.delta_target) > abs(long_put.delta_target)
    assert abs(short_put.delta_target) == 0.25
    assert abs(long_put.delta_target) == 0.10


def test_bear_call_credit_standard_structure():
    t = bear_call_credit_standard()
    assert t.n_legs == 2
    assert t.direction == "bearish"
    assert t.is_credit
    assert all(leg.option_type == "call" for leg in t.legs)
    short_call = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_call = next(leg for leg in t.legs if leg.qty_sign == +1)
    assert short_call.delta_target == 0.25
    assert long_call.delta_target == 0.10


# ---------------------------------------------------------------------------
# Per-template structural checks — Narrow width
# ---------------------------------------------------------------------------

def test_iron_condor_narrow_structure():
    t = iron_condor_narrow()
    assert t.n_legs == 4
    assert t.direction == "neutral"
    assert t.is_credit
    short_legs = [leg for leg in t.legs if leg.qty_sign == -1]
    long_legs = [leg for leg in t.legs if leg.qty_sign == +1]
    assert len(short_legs) == 2
    assert len(long_legs) == 2
    # Narrow deltas: 35d short, 15d long
    short_deltas = sorted(abs(leg.delta_target) for leg in short_legs)
    assert short_deltas == [0.35, 0.35]
    long_deltas = sorted(abs(leg.delta_target) for leg in long_legs)
    assert long_deltas == [0.15, 0.15]


def test_iron_butterfly_narrow_structure():
    t = iron_butterfly_narrow()
    assert t.n_legs == 4
    assert t.direction == "neutral"
    assert t.is_credit
    # ATM shorts unchanged at 50d
    atm_shorts = [leg for leg in t.legs
                  if leg.qty_sign == -1 and abs(leg.delta_target) >= 0.40]
    assert len(atm_shorts) == 2
    # Narrow wings: 20d (closer than standard's 10d)
    long_legs = [leg for leg in t.legs if leg.qty_sign == +1]
    assert all(abs(leg.delta_target) == 0.20 for leg in long_legs)


def test_bull_put_credit_narrow_structure():
    t = bull_put_credit_narrow()
    assert t.n_legs == 2
    assert t.direction == "bullish"
    assert t.is_credit
    assert all(leg.option_type == "put" for leg in t.legs)
    short_put = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_put = next(leg for leg in t.legs if leg.qty_sign == +1)
    assert abs(short_put.delta_target) == 0.35
    assert abs(long_put.delta_target) == 0.15


def test_bear_call_credit_narrow_structure():
    t = bear_call_credit_narrow()
    assert t.n_legs == 2
    assert t.direction == "bearish"
    assert t.is_credit
    assert all(leg.option_type == "call" for leg in t.legs)
    short_call = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_call = next(leg for leg in t.legs if leg.qty_sign == +1)
    assert short_call.delta_target == 0.35
    assert long_call.delta_target == 0.15


# ---------------------------------------------------------------------------
# Per-template structural checks — Wide width
# ---------------------------------------------------------------------------

def test_iron_condor_wide_structure():
    t = iron_condor_wide()
    assert t.n_legs == 4
    assert t.direction == "neutral"
    assert t.is_credit
    short_legs = [leg for leg in t.legs if leg.qty_sign == -1]
    long_legs = [leg for leg in t.legs if leg.qty_sign == +1]
    assert len(short_legs) == 2
    assert len(long_legs) == 2
    # Wide deltas: 15d short, 10d long
    short_deltas = sorted(abs(leg.delta_target) for leg in short_legs)
    assert short_deltas == [0.15, 0.15]
    long_deltas = sorted(abs(leg.delta_target) for leg in long_legs)
    assert long_deltas == [0.10, 0.10]


def test_bull_put_credit_wide_structure():
    t = bull_put_credit_wide()
    assert t.n_legs == 2
    assert t.direction == "bullish"
    assert t.is_credit
    assert all(leg.option_type == "put" for leg in t.legs)
    short_put = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_put = next(leg for leg in t.legs if leg.qty_sign == +1)
    assert abs(short_put.delta_target) == 0.15
    assert abs(long_put.delta_target) == 0.10


def test_bear_call_credit_wide_structure():
    t = bear_call_credit_wide()
    assert t.n_legs == 2
    assert t.direction == "bearish"
    assert t.is_credit
    assert all(leg.option_type == "call" for leg in t.legs)
    short_call = next(leg for leg in t.legs if leg.qty_sign == -1)
    long_call = next(leg for leg in t.legs if leg.qty_sign == +1)
    assert short_call.delta_target == 0.15
    assert long_call.delta_target == 0.10


# ---------------------------------------------------------------------------
# Debit
# ---------------------------------------------------------------------------

def test_bull_call_debit_structure():
    t = bull_call_debit()
    assert t.n_legs == 2
    assert t.direction == "bullish"
    assert not t.is_credit  # net debit at entry
    assert all(leg.option_type == "call" for leg in t.legs)
    # Long leg has higher delta (closer to ATM) than short leg
    long_leg = next(leg for leg in t.legs if leg.qty_sign == +1)
    short_leg = next(leg for leg in t.legs if leg.qty_sign == -1)
    assert long_leg.delta_target > short_leg.delta_target


def test_bear_put_debit_structure():
    t = bear_put_debit()
    assert t.n_legs == 2
    assert t.direction == "bearish"
    assert not t.is_credit
    assert all(leg.option_type == "put" for leg in t.legs)


# ---------------------------------------------------------------------------
# Size seeds by width
# ---------------------------------------------------------------------------

def test_narrow_templates_size_04():
    """Narrow templates should have size_seed = 0.4."""
    for factory in [iron_condor_narrow, iron_butterfly_narrow,
                    bull_put_credit_narrow, bear_call_credit_narrow]:
        t = factory()
        assert t.size_seed.value == pytest.approx(0.4), f"{t.name} size not 0.4"


def test_standard_templates_credit_size_05():
    """Standard credit templates should have size_seed = 0.5 (except IB at 0.4)."""
    for factory in [iron_condor_standard, bull_put_credit_standard,
                    bear_call_credit_standard]:
        t = factory()
        assert t.size_seed.value == pytest.approx(0.5), f"{t.name} size not 0.5"
    # IB standard has 0.4 (historical)
    t = iron_butterfly_standard()
    assert t.size_seed.value == pytest.approx(0.4)


def test_wide_templates_size_06():
    """Wide templates should have size_seed = 0.6."""
    for factory in [iron_condor_wide, bull_put_credit_wide,
                    bear_call_credit_wide]:
        t = factory()
        assert t.size_seed.value == pytest.approx(0.6), f"{t.name} size not 0.6"


# ---------------------------------------------------------------------------
# Citations + provenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", TEMPLATE_FACTORIES)
def test_template_has_citations(factory):
    """Every template must carry at least one structural-source citation
    (D88 paper-trail requirement for Chapter 4 Appendix A)."""
    t = factory()
    assert len(t.citations) >= 1, f"{t.name} missing citations"


# ---------------------------------------------------------------------------
# Direction coverage
# ---------------------------------------------------------------------------

def test_direction_coverage():
    """The 13-template library covers all three directional biases."""
    templates = all_templates()
    by_direction = {}
    for t in templates:
        by_direction.setdefault(t.direction, []).append(t.name)
    assert "neutral" in by_direction
    assert "bullish" in by_direction
    assert "bearish" in by_direction
    # Bullish/bearish should be balanced (mirror structures)
    assert len(by_direction["bullish"]) == len(by_direction["bearish"])


# ---------------------------------------------------------------------------
# No IB wide variant
# ---------------------------------------------------------------------------

def test_no_iron_butterfly_wide():
    """IB is ATM-centered — widest viable. No wide variant should exist."""
    names = {t.name for t in all_templates()}
    assert "iron_butterfly_wide" not in names
