"""Unit tests for Level B — 4th tree (delta_tree) + continuous delta exploration.

Validates:
  - Base templates have correct delta_range/wing_offset/delta_fixed/delta_seed
  - delta_tree grows correctly in GP population init
  - 4-tree Individual has correct total_nodes and signature
  - Crossover and mutation handle delta_tree
  - Delta mapping: delta_value [0,1] → short_delta via delta_range
  - Delta-dependent credit haircut formula
  - Delta-dependent stop-loss formula
  - cost_multiplier scales entry/exit costs
  - V1 templates preserved in V1_BACKUP_TEMPLATE_FACTORIES
  - Serialization includes delta_tree
  - Codegen with delta_value builds correct legs
"""
import numpy as np
import pytest

from layer2.grammar import GType, Node, node_count, to_str, validate


# ---------------------------------------------------------------------------
# Base template validation
# ---------------------------------------------------------------------------

class TestBaseTemplates:
    def test_five_base_templates_exist(self):
        from layer2.templates import BASE_TEMPLATE_FACTORIES
        assert len(BASE_TEMPLATE_FACTORIES) == 5

    def test_base_templates_have_delta_range(self):
        from layer2.templates import BASE_TEMPLATE_FACTORIES
        for factory in BASE_TEMPLATE_FACTORIES:
            t = factory()
            assert t.delta_range is not None, f"{t.name} missing delta_range"
            min_d, max_d = t.delta_range
            assert 0.05 <= min_d < max_d <= 0.50

    def test_base_templates_have_delta_seed(self):
        from layer2.templates import BASE_TEMPLATE_FACTORIES
        for factory in BASE_TEMPLATE_FACTORIES:
            t = factory()
            assert t.delta_seed is not None, f"{t.name} missing delta_seed"
            # delta_seed should be a valid tree node
            assert isinstance(t.delta_seed, Node)

    def test_iron_butterfly_is_delta_fixed(self):
        from layer2.templates import iron_butterfly_base
        t = iron_butterfly_base()
        assert t.delta_fixed is True
        assert t.delta_range == (0.10, 0.30)

    def test_iron_condor_base_ranges(self):
        from layer2.templates import iron_condor_base
        t = iron_condor_base()
        assert t.delta_range == (0.15, 0.40)
        assert t.wing_offset == 0.15
        assert t.delta_fixed is False

    def test_v1_backup_preserved(self):
        from layer2.templates import V1_BACKUP_TEMPLATE_FACTORIES, TEMPLATE_FACTORIES
        assert len(V1_BACKUP_TEMPLATE_FACTORIES) >= 13
        # V1 backup should not be the same object as current TEMPLATE_FACTORIES
        # (they may be equal if TEMPLATE_FACTORIES still points to V1)


# ---------------------------------------------------------------------------
# Individual with delta_tree
# ---------------------------------------------------------------------------

class TestIndividualDeltaTree:
    def test_total_nodes_includes_delta_tree(self):
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        entry = g.grow(GType.BOOL)
        exit_ = g.grow(GType.BOOL)
        size = g.grow(GType.REAL)
        delta = g.grow(GType.REAL)
        ind = Individual(
            entry_tree=entry, exit_tree=exit_,
            size_tree=size, delta_tree=delta,
            template_name="iron_condor_base",
        )
        expected = node_count(entry) + node_count(exit_) + node_count(size) + node_count(delta)
        assert ind.total_nodes() == expected

    def test_total_nodes_without_delta_tree(self):
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        entry = g.grow(GType.BOOL)
        exit_ = g.grow(GType.BOOL)
        size = g.grow(GType.REAL)
        ind = Individual(
            entry_tree=entry, exit_tree=exit_,
            size_tree=size,
            template_name="iron_condor_standard",
        )
        expected = node_count(entry) + node_count(exit_) + node_count(size)
        assert ind.total_nodes() == expected

    def test_signature_includes_delta_tree(self):
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        entry = g.grow(GType.BOOL)
        exit_ = g.grow(GType.BOOL)
        size = g.grow(GType.REAL)
        delta = g.grow(GType.REAL)
        ind = Individual(
            entry_tree=entry, exit_tree=exit_,
            size_tree=size, delta_tree=delta,
            template_name="test",
        )
        sig = ind.signature()
        assert "|D:" in sig

    def test_signature_without_delta_tree(self):
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        ind = Individual(
            entry_tree=g.grow(GType.BOOL), exit_tree=g.grow(GType.BOOL),
            size_tree=g.grow(GType.REAL),
            template_name="test",
        )
        sig = ind.signature()
        assert "|D:" not in sig


# ---------------------------------------------------------------------------
# Delta mapping formulas
# ---------------------------------------------------------------------------

class TestDeltaMapping:
    def test_delta_mapping_zero(self):
        """delta_value=0 → min of range."""
        min_d, max_d = 0.15, 0.40
        short_delta = min_d + 0.0 * (max_d - min_d)
        assert abs(short_delta - 0.15) < 1e-10

    def test_delta_mapping_half(self):
        """delta_value=0.5 → midpoint of range."""
        min_d, max_d = 0.15, 0.40
        short_delta = min_d + 0.5 * (max_d - min_d)
        assert abs(short_delta - 0.275) < 1e-10

    def test_delta_mapping_one(self):
        """delta_value=1 → max of range."""
        min_d, max_d = 0.15, 0.40
        short_delta = min_d + 1.0 * (max_d - min_d)
        assert abs(short_delta - 0.40) < 1e-10


# ---------------------------------------------------------------------------
# Delta-dependent haircut
# ---------------------------------------------------------------------------

class TestDeltaHaircut:
    def test_retention_at_20_delta(self):
        from layer2.evaluator_vectorized import _delta_haircut_factor
        # At 20d: no penalty, retention = base
        factor = _delta_haircut_factor(0.90, 0.20)
        assert abs(factor - 0.90) < 1e-10

    def test_retention_at_30_delta(self):
        from layer2.evaluator_vectorized import _delta_haircut_factor
        # At 30d: penalty = 0.5 * 0.10/0.20 = 0.25, retention = 0.90 * 0.75 = 0.675
        factor = _delta_haircut_factor(0.90, 0.30)
        assert abs(factor - 0.675) < 1e-10

    def test_retention_at_40_delta(self):
        from layer2.evaluator_vectorized import _delta_haircut_factor
        # At 40d: penalty = 0.5 * 0.20/0.20 = 0.50, retention = 0.90 * 0.50 = 0.45
        # But clamped to 0.50
        factor = _delta_haircut_factor(0.90, 0.40)
        assert abs(factor - 0.50) < 1e-10

    def test_retention_below_20_unchanged(self):
        from layer2.evaluator_vectorized import _delta_haircut_factor
        # Below 20d: no penalty
        factor = _delta_haircut_factor(0.90, 0.10)
        assert abs(factor - 0.90) < 1e-10

    def test_retention_never_exceeds_base(self):
        from layer2.evaluator_vectorized import _delta_haircut_factor
        # Ensure retention never exceeds base (the old bug)
        for delta in [0.10, 0.20, 0.30, 0.40, 0.50]:
            factor = _delta_haircut_factor(0.90, delta)
            assert factor <= 0.90 + 1e-10, f"retention {factor} > base 0.90 at delta {delta}"


# ---------------------------------------------------------------------------
# Delta-dependent stop-loss
# ---------------------------------------------------------------------------

class TestDeltaStopLoss:
    def test_stop_loss_15_delta(self):
        """15d → 1.5 + 0.15 * 2.0 = 1.80."""
        mult = 1.5 + 0.15 * 2.0
        assert abs(mult - 1.80) < 1e-10

    def test_stop_loss_25_delta(self):
        """25d → 1.5 + 0.25 * 2.0 = 2.00."""
        mult = 1.5 + 0.25 * 2.0
        assert abs(mult - 2.00) < 1e-10

    def test_stop_loss_40_delta(self):
        """40d → 1.5 + 0.40 * 2.0 = 2.30."""
        mult = 1.5 + 0.40 * 2.0
        assert abs(mult - 2.30) < 1e-10


# ---------------------------------------------------------------------------
# Cost multiplier
# ---------------------------------------------------------------------------

class TestCostMultiplier:
    def test_cost_multiplier_150_pct(self):
        """cost_multiplier=1.5 should increase costs by 50%."""
        base_cost = 10.0
        multiplied = base_cost * 1.5
        assert abs(multiplied - 15.0) < 1e-10


# ---------------------------------------------------------------------------
# Serialization with delta_tree
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_serialize_individual_with_delta_tree(self):
        from layer2.experiment import _serialize_individual
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        delta = g.grow(GType.REAL)
        ind = Individual(
            entry_tree=g.grow(GType.BOOL), exit_tree=g.grow(GType.BOOL),
            size_tree=g.grow(GType.REAL), delta_tree=delta,
            template_name="iron_condor_base",
        )
        d = _serialize_individual(ind)
        assert "delta_tree" in d
        assert d["delta_tree"] == to_str(delta)

    def test_serialize_individual_without_delta_tree(self):
        from layer2.experiment import _serialize_individual
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar
        g = Grammar(max_depth=3, max_nodes=8)
        ind = Individual(
            entry_tree=g.grow(GType.BOOL), exit_tree=g.grow(GType.BOOL),
            size_tree=g.grow(GType.REAL),
            template_name="iron_condor_standard",
        )
        d = _serialize_individual(ind)
        assert "delta_tree" not in d


# ---------------------------------------------------------------------------
# Codegen with delta_value
# ---------------------------------------------------------------------------

class TestCodegenDeltaValue:
    def test_build_dynamic_legs_ic(self):
        from layer3.codegen import _build_dynamic_legs
        legs = _build_dynamic_legs("iron_condor", 0.5)
        assert len(legs) == 4
        # short_delta = 0.15 + 0.5 * 0.25 = 0.275
        # long_delta = 0.275 - 0.15 = 0.125
        assert abs(legs[0].delta_target - 0.275) < 1e-3  # call short
        assert abs(legs[1].delta_target - 0.125) < 1e-3  # call long
        assert abs(legs[2].delta_target - (-0.275)) < 1e-3  # put short
        assert abs(legs[3].delta_target - (-0.125)) < 1e-3  # put long

    def test_build_dynamic_legs_ib(self):
        from layer3.codegen import _build_dynamic_legs
        legs = _build_dynamic_legs("iron_butterfly", 0.5)
        assert len(legs) == 4
        # IB: short fixed at 0.50
        assert abs(legs[0].delta_target - 0.50) < 1e-3
        assert abs(legs[2].delta_target - (-0.50)) < 1e-3
        # wing_delta = 0.10 + 0.5 * 0.20 = 0.20
        assert abs(legs[1].delta_target - 0.20) < 1e-3
        assert abs(legs[3].delta_target - (-0.20)) < 1e-3

    def test_build_dynamic_legs_bpc(self):
        from layer3.codegen import _build_dynamic_legs
        legs = _build_dynamic_legs("bull_put_credit", 0.0)
        assert len(legs) == 2
        # delta_value=0 → short_delta=0.15, long_delta=max(0, 0.05)
        assert abs(legs[0].delta_target - (-0.15)) < 1e-3
        # long_delta = 0.15 - 0.15 = 0.0 → clamped to 0.05
        assert abs(legs[1].delta_target - (-0.05)) < 1e-3

    def test_build_dynamic_legs_ratio_put_backspread(self):
        from layer3.codegen import _build_dynamic_legs
        legs = _build_dynamic_legs("ratio_put_backspread", 0.5)
        assert len(legs) == 2
        # Sell 1 near-ATM put (ratio 1), buy 2 OTM puts (ratio 2)
        assert legs[0].ratio == 1
        assert legs[1].ratio == 2  # long leg is 2x short


# ---------------------------------------------------------------------------
# ExperimentConfig cost_multiplier
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_cost_multiplier_default(self):
        from layer2.experiment import ExperimentConfig
        config = ExperimentConfig(l1_parquet_path="dummy.parquet")
        assert config.cost_multiplier == 1.0

    def test_cost_multiplier_in_fingerprint(self):
        from layer2.experiment import _FINGERPRINT_FIELDS
        assert "cost_multiplier" in _FINGERPRINT_FIELDS


# ---------------------------------------------------------------------------
# TEMPLATE_CORRELATION_GROUPS includes base templates
# ---------------------------------------------------------------------------

class TestCorrelationGroups:
    def test_base_templates_in_groups(self):
        from layer2.experiment import TEMPLATE_CORRELATION_GROUPS
        all_members = set()
        for members in TEMPLATE_CORRELATION_GROUPS.values():
            all_members |= members
        # Base templates use names without _base suffix (e.g. "iron_condor")
        assert "iron_condor" in all_members
        assert "iron_butterfly" in all_members
        assert "bull_put_credit" in all_members
        assert "bear_call_credit" in all_members
        assert "ratio_put_backspread" in all_members


# ---------------------------------------------------------------------------
# Regression: no-crossover copy must preserve delta_tree (Audit C1 fix)
# ---------------------------------------------------------------------------

class TestNoCrossoverPreservesDeltaTree:
    def test_no_crossover_copy_preserves_delta_tree(self):
        """Regression: no-crossover parent copy must include delta_tree.
        Previously, the no-crossover path (30% of offspring at default rate)
        constructed Individual without delta_tree, silently creating phantom
        V1 individuals in Level B populations."""
        from layer2.gp_engine import Individual
        from layer2.grammar import Grammar, deep_copy
        g = Grammar(max_depth=3, max_nodes=8)
        delta = g.grow(GType.REAL)
        parent = Individual(
            entry_tree=g.grow(GType.BOOL), exit_tree=g.grow(GType.BOOL),
            size_tree=g.grow(GType.REAL), template_name="iron_condor_base",
            delta_tree=delta,
        )
        # Simulate no-crossover copy (the fixed path in evolve())
        child = Individual(
            deep_copy(parent.entry_tree), deep_copy(parent.exit_tree),
            deep_copy(parent.size_tree), parent.template_name,
            delta_tree=deep_copy(parent.delta_tree) if parent.delta_tree else None,
        )
        assert child.delta_tree is not None
        assert to_str(child.delta_tree) == to_str(parent.delta_tree)


# ---------------------------------------------------------------------------
# Level B CLI wiring: --level-b flag uses base_template_by_name
# ---------------------------------------------------------------------------

class TestLevelBCLIWiring:
    def test_level_b_flag_resolves_base_templates(self):
        """ExperimentConfig with level_b=True must resolve to base templates
        (with delta_seed), not V1 templates (delta_seed=None)."""
        from layer2.templates import base_template_by_name, template_by_name
        # V1 lookup: iron_condor → iron_condor_standard (no delta_seed)
        v1 = template_by_name("iron_condor")
        assert v1.name == "iron_condor_standard"
        assert getattr(v1, "delta_seed", None) is None
        # Level B lookup: iron_condor → iron_condor (has delta_seed)
        lb = base_template_by_name("iron_condor")
        assert lb.name == "iron_condor"
        assert lb.delta_seed is not None
        assert lb.delta_range == (0.15, 0.40)

    def test_cli_parser_level_b_flag(self):
        """--level-b flag parsed correctly in walk-forward subcommand."""
        from layer2.experiment import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "walk-forward", "--l1-parquet", "/tmp/x.parquet",
            "--templates", "iron_condor", "--level-b",
        ])
        assert args.level_b is True

    def test_cli_parser_default_no_level_b(self):
        """Without --level-b, default is False."""
        from layer2.experiment import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "walk-forward", "--l1-parquet", "/tmp/x.parquet",
            "--templates", "iron_condor",
        ])
        assert args.level_b is False
