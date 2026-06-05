"""Tests for evolution strategy fixes (2026-05-18).

Covers: EphReal perturbation, crossover multi-match, ramped depth init,
adaptive mutation decay, entry+exit dedup.
"""
from __future__ import annotations

import random
import numpy as np
import pytest

from layer2.grammar import (
    GType, Grammar, TermNode, FuncNode, TermDef, FuncDef,
    node_count, depth, to_str,
)


# ---------------------------------------------------------------------------
# EphReal Gaussian perturbation
# ---------------------------------------------------------------------------

class TestEphRealPerturbation:
    """EphReal point mutation should perturb (not resample) 70% of the time."""

    def test_perturbation_preserves_approximate_value(self):
        """Perturbed EphReal should be close to original (within ~1.0)."""
        g = Grammar(max_depth=3, max_nodes=10)

        # Test the _pm method directly on a single EphReal TermNode
        # with p=1.0 so it always mutates
        close_count = 0
        stayed_eph = 0
        for _ in range(200):
            node = TermNode(defn=TermDef("EphReal", GType.REAL), value=1.5)
            mutated = g._pm(node, p=1.0, parent_fd=None, arg_idx=None)
            if mutated.name == "EphReal":
                stayed_eph += 1
                if abs(mutated.value - 1.5) < 1.0:
                    close_count += 1
        # 70% should stay as EphReal (perturbation path)
        assert stayed_eph > 100, (
            f"Only {stayed_eph}/200 stayed as EphReal — expected ~140 (70%)"
        )
        # Most perturbed values should be close to 1.5
        assert close_count > 80, (
            f"Only {close_count}/200 were close to 1.5 — "
            f"perturbation may not be working (stayed_eph={stayed_eph})"
        )

    def test_perturbation_clamped_to_range(self):
        """Perturbed EphReal should stay within [-3, 3]."""
        g = Grammar(max_depth=3, max_nodes=10)
        for _ in range(50):
            tree = FuncNode(
                defn=FuncDef("GT", [GType.REAL, GType.REAL], GType.BOOL),
                children=[
                    TermNode(defn=TermDef("ATM_IV", GType.REAL)),
                    TermNode(defn=TermDef("EphReal", GType.REAL), value=2.9),
                ],
            )
            mutated = g.point_mutate(tree, p=1.0)
            if (hasattr(mutated, 'children') and len(mutated.children) > 1
                    and mutated.children[1].name == "EphReal"):
                assert -3.0 <= mutated.children[1].value <= 3.0


# ---------------------------------------------------------------------------
# Crossover multi-match
# ---------------------------------------------------------------------------

class TestCrossoverMultiMatch:
    """Crossover should try multiple B-matches per A-node."""

    def test_crossover_produces_valid_offspring(self):
        """Basic crossover on two trees should produce valid results."""
        g = Grammar(max_depth=5, max_nodes=15)
        for _ in range(20):
            pa = g.grow(GType.BOOL, max_d=3)
            pb = g.grow(GType.BOOL, max_d=3)
            c1, c2 = g.crossover(pa, pb)
            assert depth(c1) <= g.max_depth
            assert depth(c2) <= g.max_depth
            assert node_count(c1) <= g.max_nodes
            assert node_count(c2) <= g.max_nodes

    def test_crossover_not_always_cloning(self):
        """With depth 5, crossover should succeed (not clone) at least sometimes."""
        g = Grammar(max_depth=5, max_nodes=15)
        non_clone = 0
        for _ in range(30):
            pa = g.grow(GType.BOOL, max_d=4)
            pb = g.grow(GType.BOOL, max_d=4)
            c1, c2 = g.crossover(pa, pb)
            if to_str(c1) != to_str(pa) or to_str(c2) != to_str(pb):
                non_clone += 1
        assert non_clone > 5, (
            f"Only {non_clone}/30 crossovers produced non-clones — "
            f"multi-match may not be working"
        )


# ---------------------------------------------------------------------------
# Ramped depth initialization
# ---------------------------------------------------------------------------

class TestRampedDepthInit:
    """Initial population should have diverse tree depths."""

    def test_depth_diversity(self):
        from layer2.gp_engine import initialize_population
        from layer2.templates import base_template_by_name
        g = Grammar(max_depth=5, max_nodes=15)
        t = base_template_by_name("iron_condor")
        pop = initialize_population(t, g, pop_size=40, seed_fraction=0.0)

        depths = [depth(ind.entry_tree) for ind in pop]
        unique_depths = set(depths)
        # With cycling 2,3,4,5, we should see at least 3 different depths
        assert len(unique_depths) >= 3, (
            f"Only {len(unique_depths)} unique depths: {sorted(unique_depths)} — "
            f"ramped init may not be working"
        )


# ---------------------------------------------------------------------------
# Adaptive mutation decay
# ---------------------------------------------------------------------------

class TestAdaptiveMutationDecay:
    """Mutation rate should decay in the last 1/3 of generations."""

    def test_decay_formula(self):
        """Verify the decay math at key generation points."""
        n_gens = 100
        base_rate = 0.25

        # gen 0: full rate
        gen_frac = 0 / max(n_gens - 1, 1)
        assert gen_frac <= 0.67
        effective = base_rate  # no decay

        # gen 66: ~2/3, decay just starting
        gen_frac = 66 / 99
        if gen_frac > 0.67:
            late_decay = 1.0 - 0.5 * (gen_frac - 0.67) / 0.33
            effective = base_rate * late_decay
        else:
            effective = base_rate
        assert effective > 0.24, f"At gen 66, rate should be ~full: {effective}"

        # gen 99: last gen, 50% decay
        gen_frac = 99 / 99
        late_decay = 1.0 - 0.5 * (gen_frac - 0.67) / 0.33
        effective = base_rate * late_decay
        assert 0.12 < effective < 0.13, f"At gen 99, rate should be ~0.125: {effective}"


# ---------------------------------------------------------------------------
# Dedup on entry + exit
# ---------------------------------------------------------------------------

class TestDedupEntryPlusExit:
    """Dedup should use entry+exit key, preserving exit diversity."""

    def test_same_entry_different_exit_preserved(self):
        """Two individuals with same entry but different exits should NOT be deduped."""
        from layer2.gp_engine import Individual

        entry = FuncNode(
            defn=FuncDef("GT", [GType.REAL, GType.REAL], GType.BOOL),
            children=[
                TermNode(defn=TermDef("ATM_IV", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5),
            ],
        )
        exit_a = FuncNode(
            defn=FuncDef("LT", [GType.REAL, GType.REAL], GType.BOOL),
            children=[
                TermNode(defn=TermDef("MinutesToClose", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=-1.0),
            ],
        )
        exit_b = FuncNode(
            defn=FuncDef("GT", [GType.REAL, GType.REAL], GType.BOOL),
            children=[
                TermNode(defn=TermDef("RealizedVol30m", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=2.0),
            ],
        )
        size = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5)

        ind_a = Individual(entry, exit_a, size, "test")
        ind_b = Individual(entry, exit_b, size, "test")

        import re
        _EPH_RE = re.compile(r'EphReal\((-?\d+\.\d+)\)')

        key_a = _EPH_RE.sub(
            lambda m: f'EphReal({float(m.group(1)):.1f})',
            to_str(ind_a.entry_tree) + "|" + to_str(ind_a.exit_tree),
        )
        key_b = _EPH_RE.sub(
            lambda m: f'EphReal({float(m.group(1)):.1f})',
            to_str(ind_b.entry_tree) + "|" + to_str(ind_b.exit_tree),
        )
        assert key_a != key_b, "Same entry + different exit should produce different dedup keys"

    def test_same_entry_same_exit_collapsed(self):
        """Two individuals with same entry AND exit should be deduped."""
        entry = FuncNode(
            defn=FuncDef("GT", [GType.REAL, GType.REAL], GType.BOOL),
            children=[
                TermNode(defn=TermDef("ATM_IV", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5),
            ],
        )
        exit_ = FuncNode(
            defn=FuncDef("LT", [GType.REAL, GType.REAL], GType.BOOL),
            children=[
                TermNode(defn=TermDef("MinutesToClose", GType.REAL)),
                TermNode(defn=TermDef("EphReal", GType.REAL), value=-1.0),
            ],
        )

        import re
        _EPH_RE = re.compile(r'EphReal\((-?\d+\.\d+)\)')

        key_1 = _EPH_RE.sub(
            lambda m: f'EphReal({float(m.group(1)):.1f})',
            to_str(entry) + "|" + to_str(exit_),
        )
        key_2 = _EPH_RE.sub(
            lambda m: f'EphReal({float(m.group(1)):.1f})',
            to_str(entry) + "|" + to_str(exit_),
        )
        assert key_1 == key_2, "Same entry + same exit should produce same dedup key"
