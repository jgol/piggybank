"""STVGP evolution loop using NSGA-III selection (pymoo) + hand-rolled tree
genetic operators.

Per AI Engineer review: use pymoo for what's hard to get right (non-dominated
sorting + reference-point niching), hand-roll what's already implemented in
`grammar.py` (typed tree crossover and mutation).

Per AI Engineer recommendation: per-template islands (one population per
template, evolved independently with optional migration). Methodologically
cleaner than mixed populations because crossover across structurally
different templates rarely makes sense (an iron condor's entry condition
doesn't transfer to a butterfly).

Reproducibility model (B3 fix):
  - `evolve()` re-seeds the module-global `random` and `np.random` state at
    the top of each call from `config.seed`. Because grammar.py uses the
    module-global `random` for tree generation and genetic operators, and
    templates are evolved SEQUENTIALLY in `experiment.run_experiment`, each
    template's run is fully isolated from sibling templates as long as the
    caller passes a per-template seed (see `_derive_template_seed` in
    experiment.py).
  - Parallel (threaded) template execution is NOT safe under this model;
    future parallelization requires threading a `random.Random` instance
    through grammar.py and every operator in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import signal
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import (
    HyperplaneNormalization, associate_to_niches, calc_niche_count, niching,
)


# Concurrency guard for evolve(). The B3 fix is predicated on SEQUENTIAL
# per-template execution (grammar.py uses module-global `random`). If a
# caller ever wraps `evolve()` in a ThreadPoolExecutor/ProcessPool and
# dispatches templates concurrently, the shared global RNG state will
# silently re-introduce B3 — inter-template contamination that would only
# surface as a reproducibility failure downstream. This non-blocking lock
# converts that silent footgun into a loud RuntimeError.
_EVOLVE_INFLIGHT_LOCK = threading.Lock()

from layer2.fitness import (
    DEFAULT_OBJECTIVES,
    FAILED_FITNESS_SENTINEL,
    FitnessEvaluator,
)
from layer2.grammar import (
    GType, Grammar, Node, canonical_key, deep_copy, is_structurally_degenerate,
    node_count, to_str,
)
from layer2.templates import Template


# ---------------------------------------------------------------------------
# Per-individual state
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stop-loss gene (Phase 1, docs/viable_run_plan_2026_06_03.md)
# ---------------------------------------------------------------------------
# The per-individual evolvable scalar `stop_mult` is the BASE credit-multiple at
# which a defined-risk credit spread is force-closed on intraday M2M loss. It
# replaces the hardcoded `stop_loss_credit_multiple` base in
# evaluator_vectorized.vectorized_backtest; the existing vol-conditional + time-
# dependent stop adjustments continue to apply ON TOP of this evolved base.
# stop_mult == 0.0 ⇒ no stop / hold to expiry (the measured-best policy for
# defined-risk premium sellers — §1.1 of the plan).
#
# S_MAX derivation (data-driven; see plan §Phase-1 "Provenance"):
# A defined-risk credit spread's STRUCTURAL max loss is width − credit, i.e.
# (width/credit − 1) × credit. On the production base templates that ratio is
# enormous (measured 250–570× on thin 0DTE credits), so the structural cap is
# NEVER the binding constraint for a single-digit stop. The meaningful upper
# bound is therefore the point at which a larger stop is behaviourally
# indistinguishable from "no stop": a stop_mult sweep through the vectorized
# evaluator on a volatile fixture shows stop-exit counts SATURATE by ~3×
# credit (IC stops stop firing past ~0.5×; BPC/IB never reach even 0.5× on
# wider-credit structures). S_MAX = 4.0 sits just above that saturation with
# headroom, spans the plan's validation sweep {0,1,2.5,4} exactly, and matches
# the pre-derived width/credit ≈ 4 for the seeded ~25Δ/10Δ widths under a
# realistic (full-IV-surface) credit ≈ 25% of width. Confirmed empirically
# 2026-06-03; re-confirm in the Phase-1 production validation.
STOP_MULT_MAX: float = 4.0
# Seeded individuals start at hold-to-expiry (the measured-best policy for
# premium sellers, §1.1). Backward-compat default for any Individual constructed
# without the gene (V1 path / legacy callers): 2.5 preserves the prior hardcoded
# stop behaviour. initialize_population / seeds set it explicitly regardless.
STOP_MULT_SEED: float = 0.0
STOP_MULT_DEFAULT: float = 2.5


@dataclass
class Individual:
    """One GP individual: (entry, exit, size[, delta]) trees bound to a template.

    Level B adds an optional 4th tree (delta_tree) for continuous delta
    exploration. V1 templates have delta_tree=None (fixed deltas in legs).

    `stop_mult` is an evolvable scalar gene (NOT a tree): the base credit-
    multiple for the intraday stop-loss, range [0, STOP_MULT_MAX]; 0.0 = hold to
    expiry. It flows into vectorized_backtest as `stop_loss_credit_multiple`.
    """
    entry_tree: Node
    exit_tree: Node
    size_tree: Node
    template_name: str
    delta_tree: Optional[Node] = None    # Level B: GType.REAL [0,1] → short_delta
    stop_mult: float = STOP_MULT_DEFAULT  # evolvable scalar gene ∈ [0, STOP_MULT_MAX]; 0 = no stop
    fitness: Optional[np.ndarray] = None  # (n_objectives,) — set after evaluation
    age: int = 0                           # generations survived
    _nds_rank: Optional[int] = field(default=None, init=False, repr=False)
    _crowding_dist: float = field(default=0.0, init=False, repr=False)

    def total_nodes(self) -> int:
        total = (
            node_count(self.entry_tree)
            + node_count(self.exit_tree)
            + node_count(self.size_tree)
        )
        if self.delta_tree is not None:
            total += node_count(self.delta_tree)
        return total

    def signature(self) -> str:
        """Canonical tree identifier for caching / dedup.

        Uses canonical_key so commutative-equivalent twins
        (AND(a,b)==AND(b,a), GT(x,y)==LT(y,x)) share one signature.
        """
        base = (
            f"E:{canonical_key(self.entry_tree)}"
            f"|X:{canonical_key(self.exit_tree)}"
            f"|S:{canonical_key(self.size_tree)}"
        )
        if self.delta_tree is not None:
            base += f"|D:{canonical_key(self.delta_tree)}"
        # stop_mult is part of the phenotype: two individuals with identical
        # trees but different stop levels are DIFFERENT strategies and must not
        # share a signature. Round to 3 dp to collapse Gaussian-mutation jitter
        # (matches the fitness cache key rounding).
        base += f"|SM:{self.stop_mult:.3f}"
        return base


# ---------------------------------------------------------------------------
# Population initialization
# ---------------------------------------------------------------------------

def initialize_population(template: Template, grammar: Grammar,
                          pop_size: int, seed_fraction: float = 0.5,
                          target_embproj_fraction: float = 0.50,
                          seed_entry_random: bool = False) -> List[Individual]:
    """Seed population with template seeds + random trees.

    Args:
        template: the structural template (legs + seed entry/exit/size)
        grammar: shared Grammar instance for random tree generation
        pop_size: total population
        seed_fraction: fraction of population that uses template seed trees
            (the rest are random trees of the right return types)
        target_embproj_fraction: fix #3 (2026-04-25). When the grammar has
            EmbProj_* operators (real-l1 arm), enforce that ≥ this fraction
            of the gen=0 population contains at least one EmbProj operator
            in any of the 3 trees (entry/exit/size). When the grammar lacks
            EmbProj_* (scalar-only / shuffled-l1 if the latter ever
            re-enters the case), this knob is a no-op (no EmbProj in the
            grammar → 0% target by definition). Default 0.50 means roughly
            half of gen=0 individuals carry an EmbProj operator in real-l1.

    Returns:
        List of `pop_size` Individuals — first n_seeded are seeded copies,
        remainder are randomly generated under the EmbProj-density
        constraint.
    """
    individuals: List[Individual] = []
    n_seeded = max(0, int(pop_size * seed_fraction))  # 0 when seed_fraction=0 (ablation fairness)

    _has_delta = template.delta_seed is not None  # Level B template
    for _ in range(n_seeded):
        # Use deep_copy so subsequent mutations don't share state with template.
        # seed_entry_random: Condition C entry seeds collapse after terminal
        # substitution (ATM_IV→SessionReturn → always-false). Random entry
        # trees use EmbProj operators; exit/delta seeds are still valid.
        if seed_entry_random:
            _entry = grammar.random_tree(GType.BOOL, max_depth=4, max_nodes=12)
        else:
            _entry = deep_copy(template.entry_seed)
        individuals.append(Individual(
            entry_tree=_entry,
            exit_tree=deep_copy(template.exit_seed),
            size_tree=deep_copy(template.size_seed),
            template_name=template.name,
            delta_tree=deep_copy(template.delta_seed) if _has_delta else None,
            # Seeded individuals start at the measured-best hold-to-expiry stop
            # (template.stop_seed, default 0.0). Mirrors how delta_seed seeds the
            # 4th tree. The GP then explores [0, STOP_MULT_MAX] via mutation.
            stop_mult=(template.stop_seed if template.stop_seed is not None
                       else STOP_MULT_SEED),
        ))

    # Fix #3 (2026-04-25): EmbProj-density constraint at gen=0.
    # Detect whether the grammar exposes any EmbProj_* operators. If yes,
    # enforce that ≥ target_embproj_fraction of the FULL pop (post-seed +
    # random) carries at least one EmbProj_* node. The seeded copies are
    # EmbProj-free by template design (templates.py uses raw scalar
    # terminals only in seed trees), so the random portion must do the
    # heavy lifting.
    grammar_has_embproj = any(
        f.name.startswith("EmbProj_") for f in grammar.functions
    )
    target_n_embproj = (
        int(pop_size * target_embproj_fraction) if grammar_has_embproj else 0
    )

    def _has_embproj(ind: Individual) -> bool:
        trees = [ind.entry_tree, ind.exit_tree, ind.size_tree]
        if ind.delta_tree is not None:
            trees.append(ind.delta_tree)
        for tree in trees:
            if "EmbProj_" in to_str(tree):
                return True
        return False

    # Count EmbProj-using individuals in seeded portion (always 0 for v9
    # templates, but generic).
    n_embproj_so_far = sum(1 for ind in individuals if _has_embproj(ind))

    # Random fill phase. We continue accepting random trees until we both
    # (a) reach pop_size and (b) reach target_n_embproj. If (a) is
    # reached first but (b) is not, REPLACE the most-recent non-EmbProj
    # randomly-generated individual with another random draw that DOES
    # contain EmbProj. The replacement loop has a bounded max_attempts
    # to avoid infinite spinning if the grammar happens to make EmbProj-
    # containing trees impossible (defensive — should never happen with
    # default real-l1 grammar).
    max_attempts = max(2000, pop_size * 25)
    attempts = 0
    while len(individuals) < pop_size and attempts < max_attempts:
        attempts += 1
        try:
            # Reject structurally degenerate trees (e.g., CrossAbove(X, X)
            # always-False). Check BOTH entry and exit — a degenerate entry
            # wastes a population slot just as much as a degenerate exit.
            # Ramped depth: cycle through depths 2-5 to ensure structural
            # diversity. Plain grow() biases heavily toward depth 1-2 trees
            # (50% terminal probability at each level). Cycling forces deeper
            # trees into the initial population. (Koza 1992)
            _target_d = 2 + (len(individuals) % 4)  # cycles 2,3,4,5
            entry = grammar.grow(GType.BOOL, max_d=_target_d)
            for _degen_retry in range(5):
                if not is_structurally_degenerate(entry):
                    break
                entry = grammar.grow(GType.BOOL, max_d=_target_d)
            exit_ = grammar.grow(GType.BOOL, max_d=_target_d)
            for _degen_retry in range(5):
                if not is_structurally_degenerate(exit_):
                    break
                exit_ = grammar.grow(GType.BOOL, max_d=_target_d)
            size = deep_copy(template.size_seed)  # frozen — not evolved
            delta = grammar.grow(GType.REAL, max_d=_target_d) if _has_delta else None
        except (ValueError, RecursionError):
            continue
        ind = Individual(
            entry_tree=entry, exit_tree=exit_, size_tree=size,
            template_name=template.name,
            delta_tree=delta,
            # Random individuals sample the stop gene uniformly across its full
            # range [0, STOP_MULT_MAX] (0.0 = hold-to-expiry is included). Uses
            # the module-global `random` (re-seeded per template in evolve()),
            # consistent with grammar.py's tree-generation RNG.
            stop_mult=random.uniform(0.0, STOP_MULT_MAX),
        )
        individuals.append(ind)
        if _has_embproj(ind):
            n_embproj_so_far += 1

    # Boost phase: if still under target, replace EmbProj-free random
    # individuals (NOT the seeded copies) with EmbProj-containing draws.
    if grammar_has_embproj and n_embproj_so_far < target_n_embproj:
        replace_attempts = 0
        # Iterate in reverse so we touch the random tail first.
        for i in range(len(individuals) - 1, -1, -1):
            if n_embproj_so_far >= target_n_embproj:
                break
            if i < n_seeded:
                continue  # don't replace seeded individuals
            if _has_embproj(individuals[i]):
                continue
            # Try up to N times to grow an EmbProj-containing replacement.
            for _ in range(50):
                replace_attempts += 1
                if replace_attempts > max_attempts:
                    break
                try:
                    _boost_d = 2 + (len(individuals) % 4)
                    entry = grammar.grow(GType.BOOL, max_d=_boost_d)
                    exit_ = grammar.grow(GType.BOOL, max_d=_boost_d)
                    size = deep_copy(template.size_seed)  # frozen
                    delta = grammar.grow(GType.REAL, max_d=_boost_d) if _has_delta else None
                except (ValueError, RecursionError):
                    continue
                cand = Individual(
                    entry_tree=entry, exit_tree=exit_, size_tree=size,
                    template_name=template.name,
                    delta_tree=delta,
                    stop_mult=random.uniform(0.0, STOP_MULT_MAX),  # random draw → sample gene
                )
                if _has_embproj(cand):
                    individuals[i] = cand
                    n_embproj_so_far += 1
                    break
            if replace_attempts > max_attempts:
                # Bailed — accept under-target. The stratified sampler
                # should make this rare in practice (one of 11 op classes
                # is emb_proj for real-l1), but guard against pathologies.
                break

    return individuals


# ---------------------------------------------------------------------------
# Genetic operators (hand-rolled — wrap grammar's typed crossover / mutation)
# ---------------------------------------------------------------------------

def _reflect_into_range(x: float, lo: float, hi: float) -> float:
    """Reflect a scalar back into [lo, hi] (used by stop_mult Gaussian mutation).

    Unlike clamping (which piles probability mass at the bounds), reflection
    bounces an out-of-range proposal off the boundary: x<lo → lo+(lo-x), x>hi →
    hi-(x-hi). Applied iteratively so a value reflected past the OTHER bound is
    folded back, leaving the result in [lo, hi] for any finite input. lo==hi is
    handled by returning lo.
    """
    if hi <= lo:
        return lo
    span = hi - lo
    # Fold into the [lo, hi] triangle wave. Reduce modulo 2*span around lo first
    # so a many-σ outlier can't loop forever.
    y = x - lo
    y = y % (2.0 * span)
    if y < 0.0:
        y += 2.0 * span
    if y > span:
        y = 2.0 * span - y
    return lo + y


def crossover(parent_a: Individual, parent_b: Individual,
              grammar: Grammar) -> Tuple[Individual, Individual]:
    """Subtree crossover on each tree slot independently.

    Crossover happens within each of (entry, exit, size) slots so the
    structural meaning of each slot is preserved.
    """
    # Cross entry and exit slots independently. Size tree is FROZEN.
    e1, e2 = grammar.crossover(parent_a.entry_tree, parent_b.entry_tree)
    x1, x2 = grammar.crossover(parent_a.exit_tree, parent_b.exit_tree)
    s1, s2 = deep_copy(parent_a.size_tree), deep_copy(parent_b.size_tree)
    # Level B: cross delta_tree when both parents have one
    d1, d2 = None, None
    if parent_a.delta_tree is not None and parent_b.delta_tree is not None:
        d1, d2 = grammar.crossover(parent_a.delta_tree, parent_b.delta_tree)
    elif parent_a.delta_tree is not None:
        d1, d2 = deep_copy(parent_a.delta_tree), deep_copy(parent_a.delta_tree)
    elif parent_b.delta_tree is not None:
        d1, d2 = deep_copy(parent_b.delta_tree), deep_copy(parent_b.delta_tree)
    # Simplify identity patterns (NOT(NOT(x)) → x, NOT(LT(x,y)) → GT(y,x))
    # to prevent NOT accumulation that leads to tautological entries.
    from layer2.grammar import simplify_tree
    e1, e2 = simplify_tree(e1), simplify_tree(e2)
    x1, x2 = simplify_tree(x1), simplify_tree(x2)
    if d1 is not None:
        d1 = simplify_tree(d1)
    if d2 is not None:
        d2 = simplify_tree(d2)
    # Per-tree max_nodes cap (same as mutation path). Without this,
    # crossover can produce oversized single trees that only the
    # total-nodes check catches, creating inconsistent structural constraints.
    max_n = grammar.max_nodes
    if node_count(e1) > max_n: e1 = deep_copy(parent_a.entry_tree)
    if node_count(e2) > max_n: e2 = deep_copy(parent_b.entry_tree)
    if node_count(x1) > max_n: x1 = deep_copy(parent_a.exit_tree)
    if node_count(x2) > max_n: x2 = deep_copy(parent_b.exit_tree)
    if node_count(s1) > max_n: s1 = deep_copy(parent_a.size_tree)
    if node_count(s2) > max_n: s2 = deep_copy(parent_b.size_tree)
    if d1 is not None and node_count(d1) > max_n: d1 = deep_copy(parent_a.delta_tree)
    if d2 is not None and node_count(d2) > max_n: d2 = deep_copy(parent_b.delta_tree)
    # stop_mult crossover: arithmetic (BLX-0.0) blend of the parents' scalar
    # genes — child_a leans toward parent_a, child_b toward parent_b — so the
    # offspring explore between the parents' stop levels while staying in
    # [0, STOP_MULT_MAX] (a convex combination of two in-range values is always
    # in range; clamp defensively against float drift). Mirrors delta_tree's
    # "cross when both present" intent for the scalar case.
    _alpha = random.random()
    sm1 = _alpha * parent_a.stop_mult + (1.0 - _alpha) * parent_b.stop_mult
    sm2 = _alpha * parent_b.stop_mult + (1.0 - _alpha) * parent_a.stop_mult
    sm1 = min(STOP_MULT_MAX, max(0.0, sm1))
    sm2 = min(STOP_MULT_MAX, max(0.0, sm2))
    return (
        Individual(e1, x1, s1, parent_a.template_name, delta_tree=d1, stop_mult=sm1),
        Individual(e2, x2, s2, parent_b.template_name, delta_tree=d2, stop_mult=sm2),
    )


def mutate(ind: Individual, grammar: Grammar,
           mutation_rate: float = 0.15,
           subtree_mutation_rate: float = 0.10) -> Individual:
    """Mutation operator — point-mutate at `mutation_rate` and subtree-mutate
    at `subtree_mutation_rate`, applied independently to each of the three
    tree slots (entry / exit / size).

    v9 Bug 3 fix: previously only `point_mutate` was applied. `point_mutate`
    cannot insert an EmbProj_X_k subtree into a tree whose REAL slots are
    all terminals (e.g. ATM_IV) because the terminal-mutation branch only
    swaps one terminal for another. As a result, mutation could not GROW
    EmbProj into established trees — EmbProj diversity came solely from
    initial-population random tree generation + crossover-driven
    propagation. `subtree_mutate` replaces a random non-root subtree with
    a fresh `grammar.grow(...)` subtree, which CAN insert any function in
    the grammar (including EmbProj_*) given its type matches. Adding it
    at 10% rate gives mutation a real path to introduce EmbProj into
    EmbProj-free trees.
    """
    # Mutate entry and exit slots. Size tree is FROZEN (not evolved) because:
    # - Sharpe is scale-invariant (size cancels in mean/std ratio)
    # - Drawdown/size normalization makes drawdown objective size-neutral
    # - No objective selects for or against any size value
    # - Size mutation wasted 25-33% of mutation budget on junk DNA
    # Size is fixed at the template's size_seed (constant 0.5 for all templates).
    new_entry = ind.entry_tree
    new_exit = ind.exit_tree
    new_size = ind.size_tree
    _mut_fails = 0
    if random.random() < mutation_rate:
        try:
            new_entry = grammar.point_mutate(ind.entry_tree, p=0.2)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if random.random() < mutation_rate:
        try:
            new_exit = grammar.point_mutate(ind.exit_tree, p=0.2)
        except (ValueError, RecursionError):
            _mut_fails += 1
    # Size tree FROZEN — no point or subtree mutation (see comment above).
    # v9 Bug 3: subtree mutation pass — gives mutation a path to introduce
    # EmbProj_* into established trees.
    if random.random() < subtree_mutation_rate:
        try:
            new_entry = grammar.subtree_mutate(new_entry, max_d=3)
        except (ValueError, RecursionError):
            _mut_fails += 1
    if random.random() < subtree_mutation_rate:
        try:
            new_exit = grammar.subtree_mutate(new_exit, max_d=3)
        except (ValueError, RecursionError):
            _mut_fails += 1
    # Size subtree mutation also DISABLED (frozen size tree).
    # Level B: mutate delta_tree independently
    new_delta = ind.delta_tree
    if new_delta is not None:
        if random.random() < mutation_rate:
            try:
                new_delta = grammar.point_mutate(new_delta, p=0.2)
            except (ValueError, RecursionError):
                _mut_fails += 1
        if random.random() < subtree_mutation_rate:
            try:
                new_delta = grammar.subtree_mutate(new_delta, max_d=3)
            except (ValueError, RecursionError):
                _mut_fails += 1
    # stop_mult mutation: Gaussian perturbation REFLECTED into [0, STOP_MULT_MAX].
    # Reflection (vs clamp) preserves the proposal density near the boundaries —
    # a value pushed past 0 bounces to |x|, past STOP_MULT_MAX bounces to
    # 2*MAX - x — so the gene does not pile up at the bounds. σ = 10% of the
    # range gives meaningful single-step moves while allowing fine-tuning.
    new_stop = ind.stop_mult
    if random.random() < mutation_rate:
        new_stop = _reflect_into_range(
            ind.stop_mult + random.gauss(0.0, 0.10 * STOP_MULT_MAX),
            0.0, STOP_MULT_MAX,
        )
    # Track mutation failures for diagnostics
    if _mut_fails > 0:
        if not hasattr(mutate, '_total_fails'):
            mutate._total_fails = 0
            mutate._total_calls = 0
        mutate._total_fails += _mut_fails
        mutate._total_calls += 1
    # Simplify identity patterns after mutation (same rationale as crossover)
    from layer2.grammar import simplify_tree
    new_entry = simplify_tree(new_entry)
    new_exit = simplify_tree(new_exit)
    if new_delta is not None:
        new_delta = simplify_tree(new_delta)
    # BLOCKER fix: hard-cap per-tree node count after all mutation passes.
    # Without this, accumulated point_mutate + subtree_mutate can grow trees
    # past max_nodes (IC entries reached 82 nodes in early runs vs cap,
    # causing massive overfitting). Reject over-cap mutations by reverting to parent.
    max_n = grammar.max_nodes
    if node_count(new_entry) > max_n:
        new_entry = ind.entry_tree
    if node_count(new_exit) > max_n:
        new_exit = ind.exit_tree
    if node_count(new_size) > max_n:
        new_size = ind.size_tree
    if new_delta is not None and node_count(new_delta) > max_n:
        new_delta = ind.delta_tree
    return Individual(new_entry, new_exit, new_size, ind.template_name,
                      delta_tree=new_delta, stop_mult=new_stop)


# ---------------------------------------------------------------------------
# NSGA-III selection (pymoo for non-dominated sorting + reference-point niching)
# ---------------------------------------------------------------------------

def nsga3_select(population: List[Individual],
                 ref_dirs: np.ndarray,
                 n_survivors: int,
                 rng: Optional[np.random.Generator] = None) -> List[Individual]:
    """NSGA-III environmental selection: pick `n_survivors` from `population`.

    Canonical Deb & Jain 2014 implementation via pymoo's `niching` and
    `associate_to_niches` — perpendicular-distance niching after
    hyperplane-based ideal/nadir normalization.

    Args:
        population: list of Individuals with fitness already computed
        ref_dirs: reference directions matrix (n_refs, n_objectives)
        n_survivors: how many individuals to keep (typically pop_size)
        rng: numpy Generator for reproducible niching tie-breaking.
             The production caller `evolve()` derives this from
             `config.seed`. When called directly (e.g. unit tests), an
             explicit rng is REQUIRED — previously defaulted to
             `default_rng(0)` silently but Model QA (Commit 5 review
             R8) flagged this as a reproducibility footgun.

    Returns:
        List of n_survivors Individuals (selected by NSGA-III mechanics).
    """
    if not population:
        return []
    if rng is None:
        raise ValueError(
            "nsga3_select requires an explicit `rng: np.random.Generator` — "
            "the previous `default_rng(0)` fallback was removed per Model QA "
            "Commit 5 R8 (silent reproducibility footgun). Production callers "
            "derive this via `np.random.default_rng(config.seed)`; test "
            "callers should do the same."
        )

    F = np.array([ind.fitness for ind in population])
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False, n_stop_if_ranked=n_survivors)

    # Fast path: first front alone already exceeds n_survivors → niche among
    # the first front directly (no earlier-front accumulation needed).
    # Slow path: take whole fronts until one overflows (the "splitting" front).
    survivors_indices: List[int] = []
    last_front_idx = 0
    for i, front in enumerate(fronts):
        if len(survivors_indices) + len(front) <= n_survivors:
            survivors_indices.extend(int(x) for x in front)
            last_front_idx = i
            if len(survivors_indices) == n_survivors:
                # Exactly full — return
                return [population[i] for i in survivors_indices]
        else:
            last_front_idx = i
            break

    splitting_front = fronts[last_front_idx]
    until_last_front = np.concatenate(fronts[:last_front_idx]) if last_front_idx > 0 else np.array([], dtype=int)

    # Hyperplane-based normalization (Deb & Jain 2014). `nds` is the first
    # non-dominated front — used to compute the ideal point and nadir via
    # hyperplane intercepts.
    norm = HyperplaneNormalization(ref_dirs.shape[1])
    norm.update(F, nds=fronts[0])
    ideal, nadir = norm.ideal_point, norm.nadir_point

    # Associate ALL relevant individuals (until + last front) to niches
    all_indices = np.concatenate((until_last_front, splitting_front)).astype(int)
    F_all = F[all_indices]
    niche_of_individuals, dist_to_niche, _ = associate_to_niches(
        F_all, ref_dirs, ideal, nadir
    )

    # Count niches already filled by the earlier (fully-accepted) fronts.
    # Order-dependent slice: must match the all_indices concatenation above
    # (until_last_front first, then splitting_front). If that order ever
    # changes, update this slice too.
    n_until = len(until_last_front)
    niche_count = calc_niche_count(len(ref_dirs), niche_of_individuals[:n_until])
    n_remaining = n_survivors - n_until

    # Niching on the splitting front's local view
    last_niche_of  = niche_of_individuals[n_until:]
    last_dist      = dist_to_niche[n_until:]
    chosen_local   = niching(
        pop=np.arange(len(splitting_front)),   # placeholder; niching indexes by position
        n_remaining=n_remaining,
        niche_count=niche_count,
        niche_of_individuals=last_niche_of,
        dist_to_niche=last_dist,
        random_state=rng,
    )

    # chosen_local are indices into the splitting front; translate back to
    # population indices.
    survivors_indices.extend(int(splitting_front[i]) for i in chosen_local)
    return [population[i] for i in survivors_indices]


# ---------------------------------------------------------------------------
# Generation-level metrics
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    generation: int
    population_size: int
    front_size: int             # Pareto front cardinality
    best_per_objective: List[float]
    median_per_objective: List[float]
    worst_per_objective: List[float]
    mean_node_count: float
    nan_count: int
    cache_hit_rate: float
    wall_time_s: float
    mutation_fail_rate: float = 0.0  # fraction of mutation attempts that failed


def compute_metrics(generation: int, population: List[Individual],
                    fitness_evaluator: FitnessEvaluator,
                    wall_time_s: float) -> GenerationMetrics:
    F = np.array([ind.fitness for ind in population])
    n_obj = F.shape[1]
    # Pareto front size
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False)
    front_size = len(fronts[0]) if fronts else 0
    # Mutation failure rate
    _mfr = 0.0
    if hasattr(mutate, '_total_calls') and mutate._total_calls > 0:
        _mfr = mutate._total_fails / (mutate._total_calls * 6)  # 6 mutation attempts per call max
    return GenerationMetrics(
        generation=generation,
        population_size=len(population),
        front_size=front_size,
        best_per_objective=[float(F[:, k].min()) for k in range(n_obj)],
        median_per_objective=[float(np.median(F[:, k])) for k in range(n_obj)],
        worst_per_objective=[float(F[:, k].max()) for k in range(n_obj)],
        mean_node_count=float(np.mean([ind.total_nodes() for ind in population])),
        nan_count=int(np.sum(~np.isfinite(F))),
        cache_hit_rate=fitness_evaluator.cache_hit_rate,
        wall_time_s=wall_time_s,
        mutation_fail_rate=_mfr,
    )


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    pop_size: int = 256
    n_generations: int = 100
    seed: int = 42
    crossover_rate: float = 0.7
    mutation_rate: float = 0.25
    tournament_size: int = 3
    seed_fraction: float = 0.5
    seed_entry_random: bool = False  # Condition C: randomize entry trees in seeded fraction
    # NSGA-III reference directions
    n_partitions: int = 12         # Das-Dennis partitions; 12 → 91 ref points for 3-obj


# ---------------------------------------------------------------------------
# Generation-level checkpoint / resume (SIGHUP/SIGTERM-safe)
# ---------------------------------------------------------------------------
#
# A single template-fold evolution can run 4-11h. Before this, the run
# persisted NOTHING resumable mid-flight: a SIGHUP (closed terminal / dropped
# SSH) or SIGTERM lost the entire unit. This block adds an ATOMIC,
# fingerprint-guarded checkpoint written every K generations AND on receipt of
# SIGTERM/SIGHUP. Resume restores the population, all three RNG streams, the
# ramped evaluator knobs, and the inner-val tracking dict so a resumed run is
# byte-for-byte identical to an uninterrupted one with the same seed.
#
# The checkpoint format version. Bump if the on-disk schema changes so old
# checkpoints are rejected (they would otherwise restore into a changed loop).
# v3 (2026-06-03): Individual gained the `stop_mult` gene. Pre-v3 checkpoints
# have no per-individual stop_mult and were evolved under the fixed-2.5× stop,
# so resuming one into the stop-gene loop would silently mix policies — reject
# them (forces a clean restart) rather than back-fill a default.
_CHECKPOINT_FORMAT_VERSION = 3

# Silent-collapse guard (2026-06-02). If EVERY individual is infeasible
# (FAILED_FITNESS_SENTINEL on objective 0) for this many CONSECUTIVE generations,
# NSGA-III has nothing to select on and the run silently returns an arbitrary
# churn front instead of erroring — the exact failure that masqueraded as a "GP
# can't discover signal" bug (root-caused to warmup_bars >= bars/day starving the
# per-day fitness gates; see tests/test_gp_positive_control.py). 3 consecutive
# all-sentinel generations is decisive: the min_trades ramp is at its MOST lenient
# early (starts at 1), so an all-infeasible gen-0/1/2 is structural, not chance
# (P(3 consecutive zero-feasible) at even a 1% true rate with pop=256 is ~4e-4).
# Disable via GP_DISABLE_FEASIBILITY_GUARD=1 for deliberate infeasible-path tests.
_FEAS_GUARD_CONSECUTIVE_GENS = 3
# Only enforce on production-size populations; tiny test pops can legitimately
# sit infeasible for a few generations without indicating a structural fault.
_FEAS_GUARD_MIN_POP = 64

# Env var the launch script sets so checkpointing is enabled WITHOUT editing
# experiment.py (which does not pass checkpoint_dir). When evolve()'s
# checkpoint_dir arg is None we fall back to this directory.
_CHECKPOINT_DIR_ENV = "GP_CHECKPOINT_DIR"

# Sentinel raised by the signal handler so the (already in-progress)
# generation loop can flush a checkpoint and exit cleanly instead of dying
# mid-write. We translate the signal into this exception only after a
# checkpoint has been flushed.
class _CheckpointAndExit(BaseException):
    """Internal: signal received → checkpoint flushed → exit the run."""
    def __init__(self, signum: int):
        super().__init__(f"signal {signum}")
        self.signum = signum


def _serialize_individual(ind: "Individual") -> Dict:
    """Serialize one Individual to a plain dict via the to_str/from_sexpr
    round-trip (verified lossless for BOOL/REAL trees, incl. delta trees and
    EphReal/EphInt constants). Carries fitness + age + per-individual NSGA
    bookkeeping so the resumed population is structurally identical."""
    return {
        "entry": to_str(ind.entry_tree),
        "exit": to_str(ind.exit_tree),
        "size": to_str(ind.size_tree),
        "delta": to_str(ind.delta_tree) if ind.delta_tree is not None else None,
        "stop_mult": float(ind.stop_mult),
        "template_name": ind.template_name,
        "fitness": (ind.fitness.tolist() if ind.fitness is not None else None),
        "age": ind.age,
        "nds_rank": ind._nds_rank,
        "crowding_dist": ind._crowding_dist,
    }


def _deserialize_individual(d: Dict) -> "Individual":
    """Inverse of _serialize_individual."""
    from layer2.grammar import from_sexpr as _from_sexpr
    ind = Individual(
        entry_tree=_from_sexpr(d["entry"]),
        exit_tree=_from_sexpr(d["exit"]),
        size_tree=_from_sexpr(d["size"]),
        template_name=d["template_name"],
        delta_tree=(_from_sexpr(d["delta"]) if d["delta"] is not None else None),
        # Backward-compat: pre-stop_mult checkpoints lack the key → fall back to
        # the legacy default so an old checkpoint still deserializes.
        stop_mult=float(d.get("stop_mult", STOP_MULT_DEFAULT)),
    )
    ind.fitness = (np.asarray(d["fitness"], dtype=np.float64)
                   if d["fitness"] is not None else None)
    ind.age = int(d.get("age", 0))
    ind._nds_rank = d.get("nds_rank")
    ind._crowding_dist = float(d.get("crowding_dist", 0.0))
    return ind


def _run_fingerprint(template: "Template", grammar: "Grammar",
                     config: "EvolutionConfig",
                     fitness_evaluator: "FitnessEvaluator") -> str:
    """Stable identity hash so a stale / incompatible checkpoint is rejected.

    Mixes (a) the config that drives the search, (b) the grammar signature
    (function + terminal names — a grammar mutation must invalidate the
    checkpoint), (c) the template identity, and (d) the objective set. A
    checkpoint whose fingerprint differs from the live run is ignored.
    """
    func_names = tuple(sorted(f.name for f in grammar.functions))
    term_names = tuple(sorted(t.name for t in grammar.terminals))
    try:
        obj_names = tuple(getattr(o, "__name__", str(o))
                          for o in fitness_evaluator.objectives)
    except Exception:
        obj_names = ()
    # The evaluator's data fingerprint makes the checkpoint FOLD-SPECIFIC: a
    # walk-forward run evolves the SAME template + config across folds with
    # DIFFERENT train windows, and the checkpoint file name
    # (checkpoint_<template>.pkl) is identical across folds. Without the data
    # fingerprint, a fold-2 checkpoint could be wrongly accepted when fold-1
    # restarts. Including it guarantees a stale-fold checkpoint is rejected.
    data_fp = getattr(fitness_evaluator, "_data_fingerprint", "") or ""
    payload = {
        "fmt": _CHECKPOINT_FORMAT_VERSION,
        "template": template.name,
        "data_fingerprint": data_fp,
        "seed": config.seed,
        "pop_size": config.pop_size,
        "n_generations": config.n_generations,
        "crossover_rate": config.crossover_rate,
        "mutation_rate": config.mutation_rate,
        "tournament_size": config.tournament_size,
        "seed_fraction": config.seed_fraction,
        "seed_entry_random": config.seed_entry_random,
        "n_partitions": config.n_partitions,
        # stop_mult gene range — changing S_MAX changes the search space, so a
        # checkpoint written under a different bound must be rejected.
        "stop_mult_max": STOP_MULT_MAX,
        "grammar_max_depth": grammar.max_depth,
        "grammar_max_nodes": grammar.max_nodes,
        "grammar_funcs": func_names,
        "grammar_terms": term_names,
        "objectives": obj_names,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _checkpoint_path(checkpoint_dir: "Path", template_name: str) -> "Path":
    return Path(checkpoint_dir) / f"checkpoint_{template_name}.pkl"


def _write_checkpoint_atomic(checkpoint_dir: "Path", template_name: str,
                             state: Dict) -> None:
    """ATOMIC checkpoint write: dump to a temp file in the SAME directory,
    fsync, then os.replace() over the target so a reader (or a crash mid-write)
    never observes a torn checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = _checkpoint_path(checkpoint_dir, template_name)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)  # atomic on POSIX


def _delete_checkpoint(checkpoint_dir: "Path", template_name: str) -> None:
    try:
        _checkpoint_path(checkpoint_dir, template_name).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _load_checkpoint(checkpoint_dir: "Path", template_name: str,
                     expected_fingerprint: str) -> Optional[Dict]:
    """Load checkpoint_<template>.pkl if present AND its fingerprint matches.

    Returns the state dict on a clean, matching checkpoint; None if absent,
    unreadable, format-mismatched, or fingerprint-mismatched (warns in the
    mismatch case so a stale checkpoint never silently corrupts a run)."""
    path = _checkpoint_path(checkpoint_dir, template_name)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
    except Exception as exc:
        warnings.warn(
            f"[checkpoint] {path} is unreadable ({type(exc).__name__}: {exc}); "
            f"ignoring and starting fresh.",
            stacklevel=2,
        )
        return None
    if state.get("format_version") != _CHECKPOINT_FORMAT_VERSION:
        warnings.warn(
            f"[checkpoint] {path} has format_version "
            f"{state.get('format_version')} != current "
            f"{_CHECKPOINT_FORMAT_VERSION}; ignoring stale checkpoint.",
            stacklevel=2,
        )
        return None
    if state.get("fingerprint") != expected_fingerprint:
        warnings.warn(
            f"[checkpoint] {path} fingerprint mismatch "
            f"(config/grammar changed since it was written); ignoring stale "
            f"checkpoint and starting fresh.",
            stacklevel=2,
        )
        return None
    return state


def evolve(template: Template, fitness_evaluator: FitnessEvaluator,
           grammar: Grammar, config: EvolutionConfig,
           on_generation: Optional[Callable[[GenerationMetrics, List[Individual]], None]] = None,
           val_evaluator: Optional[Callable[["List[Individual]"], float]] = None,
           val_checkpoint_interval: int = 10,
           val_patience: int = 30,
           checkpoint_dir: Optional[str] = None,
           checkpoint_every: int = 5,
           ) -> Tuple[List[Individual], List[GenerationMetrics], Dict]:
    """Run NSGA-III evolution for one template.

    Args:
        val_evaluator: Optional callback for inner validation monitoring.
            Called every `val_checkpoint_interval` generations with the
            current Pareto front. Returns the mean val_sharpe of the front.
            OBSERVATION ONLY -- val fitness is NOT used for NSGA-III selection.
            Contaminating the validation set with selection pressure would
            invalidate the out-of-sample guarantee (Vanneschi et al. 2010).
        val_checkpoint_interval: How often (in generations) to run validation.
            Default 10: evaluating ~40 individuals at ~1ms each = ~40ms.
        val_patience: If val_sharpe does not improve for this many generations
            (consecutive checkpoints), log a stagnation warning. Default 30
            (= 3 consecutive checkpoints at interval 10). Does NOT stop
            evolution -- the GP may recover.
        checkpoint_dir: Directory for the SIGHUP/SIGTERM-safe generation-level
            checkpoint (``checkpoint_<template>.pkl``). If None, falls back to
            the ``GP_CHECKPOINT_DIR`` environment variable (set by
            ``scripts/launch_gp_run.sh``) so checkpointing can be enabled
            WITHOUT editing experiment.py. If neither is set, checkpointing is
            disabled. On entry, a matching checkpoint is RESTORED and evolution
            resumes from the saved generation; the checkpoint is DELETED on
            clean completion.
        checkpoint_every: Write a checkpoint every K generations (default 5).
            A checkpoint is ALSO written immediately on SIGTERM/SIGHUP.

    Returns:
        (final_population, per_generation_metrics, val_tracking) where
        val_tracking is a dict with keys:
          - val_checkpoints: list of (gen, mean_val_sharpe) tuples
          - best_val_sharpe: float or None
          - best_val_generation: int or None
          - best_val_front: list of Individuals at best-val gen, or None

    Raises:
        RuntimeError: if another thread is already inside `evolve()`. See the
            B3 concurrency guard note at the top of this module.
    """
    if not _EVOLVE_INFLIGHT_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "evolve() is not safe to call concurrently from multiple threads: "
            "grammar.py uses module-global `random` and `evolve()` re-seeds "
            "module-global RNG state at entry. Concurrent calls would "
            "silently corrupt each other's RNG streams. Run templates "
            "sequentially, or refactor grammar.py + gp_engine.py to thread "
            "a per-template random.Random / np.random.Generator through "
            "every call site."
        )
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        # P1-B (DSR gate): record every distinct individual's RAW training
        # Sharpe across the whole run, so the front-level Deflated-Sharpe gate
        # can deflate champions against the evolution's full trial population
        # (N, V) rather than the tiny — and variance-inverting — final front.
        # Opt-in on the evaluator; surfaced via _val_tracking["evo_trial_records"].
        if hasattr(fitness_evaluator, "enable_trial_recording"):
            fitness_evaluator.enable_trial_recording()
        # Dedicated niching RNG derived from config.seed so pymoo's canonical
        # niching is reproducible AND decoupled from the module-global stream
        # used by grammar.py.
        niching_rng = np.random.default_rng(config.seed)

        n_obj = len(fitness_evaluator.objectives)
        # Auto-scale n_partitions so ref_dirs <= pop_size (Deb & Jain 2014).
        # With pop < ref_dirs, NSGA-III niching degenerates to near-random
        # selection because most niches are permanently empty.
        n_part = config.n_partitions
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)
        while ref_dirs.shape[0] > config.pop_size and n_part > 1:
            n_part -= 1
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_part)

        # Auto-scale seed_fraction: at small pop sizes, too many identical
        # seeds waste diversity. Cap seed count at 32 regardless of pop.
        effective_seed_frac = min(config.seed_fraction, 32.0 / max(config.pop_size, 1))

        # Initialize
        population = initialize_population(
            template, grammar, config.pop_size, effective_seed_frac,
            seed_entry_random=config.seed_entry_random,
        )

        metrics_log: List[GenerationMetrics] = []

        # min_trades ramp: start at 1, linearly increase to the target by gen 30.
        # With min_trades=30, a longer ramp prevents cold-start catastrophe
        # while still reaching full stringency by gen 30 (30% of evolution).
        _target_min_trades = fitness_evaluator.min_trades
        _ramp_gens = min(30, config.n_generations // 3)

        # Random-entry nursery ramp: margin starts at 0, linearly increases to
        # 0.30 over the first 50% of generations. Extended from 40% to match
        # the longer min_trades ramp — gives GP time to find frequent patterns.
        _target_re_margin = fitness_evaluator.random_entry_margin
        _re_ramp_gens = max(1, int(0.5 * config.n_generations))

        # Inner validation state (task #34: overfitting monitor).
        # Tracks val_sharpe at periodic checkpoints for observational
        # monitoring. Val fitness is NEVER used for selection -- only
        # training fitness drives NSGA-III.
        _val_checkpoints: List[Tuple[int, float]] = []  # (generation, mean_val_sharpe)
        _best_val_sharpe: float = -float('inf')
        _best_val_generation: int = 0
        _best_val_front_snapshot: Optional[List[Individual]] = None
        _val_stagnation_warned: bool = False
        _consecutive_overfit_checks: int = 0  # early stopping counter
        _consecutive_zero_feasible: int = 0   # silent-collapse guard counter

        # -- Generation-level checkpoint / resume setup --------------------
        # Resolve the checkpoint directory: explicit arg wins, else the
        # GP_CHECKPOINT_DIR env var (set by scripts/launch_gp_run.sh) so
        # checkpointing can be enabled without editing experiment.py.
        _ckpt_dir = checkpoint_dir or os.environ.get(_CHECKPOINT_DIR_ENV) or None
        # Allow the launch script to tune the cadence via env without editing
        # experiment.py's evolve() call (which never passes checkpoint_every).
        _env_every = os.environ.get("GP_CHECKPOINT_EVERY")
        if _env_every:
            try:
                checkpoint_every = max(1, int(_env_every))
            except ValueError:
                pass
        _ckpt_fp = _run_fingerprint(template, grammar, config, fitness_evaluator) \
            if _ckpt_dir else None
        _start_gen = 0
        _resumed = False  # set True iff we restored from an existing checkpoint

        def _capture_state(next_gen: int) -> Dict:
            """Snapshot everything needed to resume AT the start of next_gen.

            Called after generation (next_gen - 1) is fully complete (population
            selected + aged), so all three RNG streams hold exactly the state
            an uninterrupted run would have at the top of `range`'s next_gen
            iteration — giving determinism across the checkpoint boundary."""
            return {
                "format_version": _CHECKPOINT_FORMAT_VERSION,
                "fingerprint": _ckpt_fp,
                "template_name": template.name,
                "gen": int(next_gen),                 # resume at this generation
                "population": [_serialize_individual(i) for i in population],
                "random_state": random.getstate(),
                "np_random_state": np.random.get_state(),
                "niching_rng_state": niching_rng.bit_generator.state,
                "min_trades": fitness_evaluator.min_trades,
                "random_entry_margin": fitness_evaluator.random_entry_margin,
                # Keep ONLY generations strictly before the resume point. On the
                # clean boundary this is a no-op (all entries already satisfy
                # generation < next_gen). On the signal path it drops the
                # in-flight generation's (partial) metrics, which would otherwise
                # be double-counted when that generation is re-run on resume.
                "metrics_log": [m.__dict__.copy() for m in metrics_log
                                if m.generation < next_gen],
                "val_tracking": {
                    "val_checkpoints": list(_val_checkpoints),
                    "best_val_sharpe": _best_val_sharpe,
                    "best_val_generation": _best_val_generation,
                    "best_val_front": (
                        [_serialize_individual(i) for i in _best_val_front_snapshot]
                        if _best_val_front_snapshot is not None else None),
                    "val_stagnation_warned": _val_stagnation_warned,
                    "consecutive_overfit_checks": _consecutive_overfit_checks,
                },
            }

        if _ckpt_dir:
            _restored = _load_checkpoint(_ckpt_dir, template.name, _ckpt_fp)
            if _restored is not None:
                _resumed = True
                _start_gen = int(_restored["gen"])
                if _start_gen >= config.n_generations:
                    # Already complete — nothing to resume; fall through to the
                    # post-loop bookkeeping with the restored (final) population.
                    warnings.warn(
                        f"[checkpoint] {template.name}: restored gen "
                        f"{_start_gen} >= n_generations "
                        f"{config.n_generations}; run already complete.",
                        stacklevel=2,
                    )
                population = [_deserialize_individual(d)
                             for d in _restored["population"]]
                random.setstate(_restored["random_state"])
                np.random.set_state(_restored["np_random_state"])
                niching_rng.bit_generator.state = _restored["niching_rng_state"]
                fitness_evaluator.min_trades = _restored["min_trades"]
                fitness_evaluator.random_entry_margin = _restored["random_entry_margin"]
                # Restore the per-generation metrics history so the returned
                # metrics_log is complete (heartbeat/convergence checks rely on it).
                metrics_log = [GenerationMetrics(**m)
                               for m in _restored.get("metrics_log", [])]
                _vt = _restored.get("val_tracking", {})
                _val_checkpoints = list(_vt.get("val_checkpoints", []))
                _best_val_sharpe = _vt.get("best_val_sharpe", -float('inf'))
                _best_val_generation = _vt.get("best_val_generation", 0)
                _val_stagnation_warned = _vt.get("val_stagnation_warned", False)
                _consecutive_overfit_checks = _vt.get("consecutive_overfit_checks", 0)
                _bv = _vt.get("best_val_front")
                _best_val_front_snapshot = (
                    [_deserialize_individual(d) for d in _bv]
                    if _bv is not None else None)
                print(f"[checkpoint] {template.name}: RESUMING from generation "
                      f"{_start_gen}/{config.n_generations} "
                      f"({len(population)} individuals restored).")

        # H2 FIX — gen-0 checkpoint hole. The every-K-gens write only fires at
        # the END of a generation (gen+1 boundary), and the signal handler
        # snapshots `_gen_cursor["gen"]` — but gen 0 is not on disk until its
        # FULL-population evaluation completes (minutes-to-hours for a large
        # pop). A SIGTERM/SIGHUP that lands DURING the gen-0 evaluate loop
        # therefore left NO checkpoint, so that template-fold had to restart
        # from scratch while siblings resumed mid-flight (observed:
        # iron_butterfly lost its whole gen-0). FIX: as soon as the population
        # is initialized (and we are NOT resuming an existing checkpoint),
        # flush an initial resume@0 checkpoint of the freshly-initialized,
        # not-yet-evaluated population. It is fully resume-valid: the RNG
        # streams hold their exact post-init state, `gen: 0` makes the loop
        # re-run generation 0 from the top, and the restored individuals have
        # fitness=None so they are (re-)evaluated in gen 0 — identical to an
        # uninterrupted fresh start. Same fingerprint/format the resume path
        # already verifies. Atomic + non-fatal on error (a failed initial
        # write must never block the run).
        if _ckpt_dir and not _resumed:
            try:
                _write_checkpoint_atomic(_ckpt_dir, template.name,
                                         _capture_state(0))
            except Exception as _init_ck_exc:
                warnings.warn(
                    f"[checkpoint] initial gen-0 write failed "
                    f"({type(_init_ck_exc).__name__}: {_init_ck_exc}); a gen-0 "
                    f"interrupt will restart this template from scratch.",
                    stacklevel=2,
                )

        # Arm SIGTERM/SIGHUP handlers ONLY while inside evolve(), and only in
        # the main thread (signal.signal raises ValueError off-thread). On a
        # signal we flush a checkpoint synchronously, then raise
        # _CheckpointAndExit so the loop unwinds cleanly. If we cannot install
        # handlers (worker thread), checkpoint-every-K-gens still protects the
        # run; we just lose the on-signal flush.
        _signals_to_trap = [
            s for s in (getattr(signal, "SIGTERM", None),
                        getattr(signal, "SIGHUP", None))
            if s is not None
        ]
        _prev_handlers: Dict[int, object] = {}
        _signal_received: Dict[str, Optional[int]] = {"signum": None}
        _handlers_armed = False
        # Mutable cursor the loop updates each iteration so the signal handler
        # (which can fire at any instant) knows which generation is in flight.
        _gen_cursor: Dict[str, int] = {"gen": _start_gen}

        def _signal_handler(signum, _frame):
            # Flush a checkpoint SYNCHRONOUSLY before unwinding. The handler
            # runs in the main thread, so it can safely read the loop's
            # population/RNG state via closure. `population` is only REASSIGNED
            # at end-of-generation (offspring live in a separate list), so
            # mid-generation it still references the last fully-completed
            # generation — we therefore resume AT the in-progress generation
            # `_gen_cursor["gen"]` (re-doing it; valid, though not bit-identical
            # to an uninterrupted run, which only the every-K boundary
            # guarantees). Then raise so the run unwinds through the lock's
            # finally: and exits cleanly.
            _signal_received["signum"] = signum
            try:
                _g = int(_gen_cursor["gen"])
                _write_checkpoint_atomic(_ckpt_dir, template.name,
                                         _capture_state(_g))
                print(f"[checkpoint] {template.name}: caught signal {signum}; "
                      f"flushed checkpoint at generation {_g}; exiting.")
            except Exception as _flush_exc:  # never let a flush error mask exit
                print(f"[checkpoint] signal-flush FAILED "
                      f"({type(_flush_exc).__name__}: {_flush_exc}).")
            raise _CheckpointAndExit(signum)

        if _ckpt_dir and threading.current_thread() is threading.main_thread():
            try:
                for _s in _signals_to_trap:
                    _prev_handlers[_s] = signal.getsignal(_s)
                    signal.signal(_s, _signal_handler)
                _handlers_armed = True
            except (ValueError, OSError, RuntimeError) as _sig_exc:
                # Not the main thread (or platform refuses) — degrade to
                # checkpoint-every-K-gens without the on-signal flush.
                _handlers_armed = False
                warnings.warn(
                    f"[checkpoint] could not install signal handlers "
                    f"({type(_sig_exc).__name__}: {_sig_exc}); relying on "
                    f"checkpoint-every-{checkpoint_every}-generations only.",
                    stacklevel=2,
                )

        def _restore_signal_handlers():
            if not _handlers_armed:
                return
            for _s, _h in _prev_handlers.items():
                try:
                    signal.signal(_s, _h)
                except (ValueError, OSError, RuntimeError):
                    pass

        for gen in range(_start_gen, config.n_generations):
            gen_start = time.time()
            _gen_cursor["gen"] = gen  # so an async signal knows the live gen

            # Change A: Per-generation noise injection — apply calibrated
            # Gaussian noise to terminal data. Each generation sees different
            # noise, preventing GP from overfitting to proxy-specific terminal
            # values that diverge from QC reality. Noise magnitudes calibrated
            # from proxy-QC terminal comparison (MAE in normalized space).
            if hasattr(fitness_evaluator, 'apply_generation_noise'):
                fitness_evaluator.apply_generation_noise(gen, config.seed)
                # Noise invalidates all cached fitness — force re-evaluation.
                for ind in population:
                    ind.fitness = None

            # Ramp min_trades from 1 to target over first _ramp_gens generations.
            # When min_trades changes, invalidate all cached fitness values
            # because individuals that passed at min_trades=1 might need
            # sentinel at min_trades=10.
            _prev_mt = fitness_evaluator.min_trades
            if _ramp_gens > 0 and gen < _ramp_gens:
                fitness_evaluator.min_trades = max(1, int(
                    1 + (_target_min_trades - 1) * gen / _ramp_gens
                ))
            else:
                fitness_evaluator.min_trades = _target_min_trades
            if fitness_evaluator.min_trades != _prev_mt:
                # Invalidate fitness so individuals get re-evaluated at new threshold
                for ind in population:
                    ind.fitness = None
                if hasattr(fitness_evaluator, '_cache'):
                    fitness_evaluator._cache.clear()

            # Ramp random-entry margin from 0 to target over first 40% of gens.
            # Round to 2 decimals to avoid clearing cache every single generation
            # (float division produces unique values each gen, defeating cache).
            _prev_rem = fitness_evaluator.random_entry_margin
            if _re_ramp_gens > 0 and gen < _re_ramp_gens:
                fitness_evaluator.random_entry_margin = round(
                    _target_re_margin * gen / _re_ramp_gens, 2
                )
            else:
                fitness_evaluator.random_entry_margin = _target_re_margin
            if fitness_evaluator.random_entry_margin != _prev_rem:
                for ind in population:
                    ind.fitness = None
                if hasattr(fitness_evaluator, '_cache'):
                    fitness_evaluator._cache.clear()

            # Evaluate fitness for all individuals (cache handles repeats)
            for ind in population:
                if ind.fitness is None:
                    ind.fitness = fitness_evaluator.evaluate(
                        ind.entry_tree, ind.exit_tree, ind.size_tree,
                        delta_tree=ind.delta_tree,
                        stop_mult=ind.stop_mult,
                    )
                    # Note: degenerate exit trees (CrossBelow(X, X) etc.) are
                    # caught structurally by is_structurally_degenerate() at
                    # initialization + crossover, not via fitness pressure.

            # Phenotypic dedup: remove individuals with identical entry trees.
            # Without this, 80%+ of the Pareto front is syntactic near-duplicates
            # (same entry with minor exit/size variations). Dedup keeps the
            # individual with best neg_sharpe (fitness[0]) per unique entry.
            # Only run on production-size populations (skip in small-pop tests).
            _seen_entries: Dict[str, int] = {}  # entry_str → population index
            _to_remove: set = set()
            if config.pop_size >= 64:  # skip dedup for small test populations
              # (at pop<64, dedup can remove a large fraction and destabilize NSGA-III)
              for _pi, _ind in enumerate(population):
                # Dedup on entry + exit (not just entry). Preserves exit-tree
                # diversity for the same entry pattern. canonical_key collapses
                # commutative twins (AND(a,b)==AND(b,a), GT(x,y)==LT(y,x)) so the
                # front isn't filled with semantically-identical clones; _eph_round=1
                # keeps the prior EphReal threshold-jitter tolerance.
                _estr = (
                    canonical_key(_ind.entry_tree, _eph_round=1)
                    + "|"
                    + canonical_key(_ind.exit_tree, _eph_round=1)
                )
                if _estr in _seen_entries:
                    _prev_idx = _seen_entries[_estr]
                    # Keep the one with better neg_sharpe (lower = better)
                    if _ind.fitness[0] < population[_prev_idx].fitness[0]:
                        _to_remove.add(_prev_idx)
                        _seen_entries[_estr] = _pi
                    else:
                        _to_remove.add(_pi)
                else:
                    _seen_entries[_estr] = _pi
            if _to_remove:
                # Replace duplicates with fresh random individuals (screened
                # for structural degeneracy, same as initialize_population).
                for _ri in _to_remove:
                    for _attempt in range(5):
                        _e = grammar.grow(GType.BOOL, max_d=grammar.max_depth)
                        _x = grammar.grow(GType.BOOL, max_d=grammar.max_depth)
                        if not (is_structurally_degenerate(_e) or is_structurally_degenerate(_x)):
                            break
                    population[_ri] = Individual(
                        entry_tree=_e,
                        exit_tree=_x,
                        size_tree=deep_copy(template.size_seed),  # frozen
                        template_name=template.name,
                        delta_tree=(grammar.grow(GType.REAL, max_d=grammar.max_depth)
                                   if template.delta_seed is not None else None),
                        # Fresh random replacement → sample the stop gene too.
                        stop_mult=random.uniform(0.0, STOP_MULT_MAX),
                    )
                    population[_ri].fitness = None  # force re-evaluation

            # Re-evaluate any new individuals from dedup replacement
            for ind in population:
                if ind.fitness is None:
                    ind.fitness = fitness_evaluator.evaluate(
                        ind.entry_tree, ind.exit_tree, ind.size_tree,
                        delta_tree=ind.delta_tree,
                        stop_mult=ind.stop_mult,
                    )

            # Silent-collapse guard: fail FAST if the WHOLE population is infeasible
            # for _FEAS_GUARD_CONSECUTIVE_GENS consecutive generations. An all-sentinel
            # population gives NSGA-III nothing to select on, so the run would silently
            # return an arbitrary churn front (the failure that masqueraded as a GP
            # discovery bug — root cause warmup_bars >= bars/day starving the per-day
            # fitness gates). The `finally:` at function exit restores signal handlers +
            # releases the lock, so a bare raise unwinds cleanly to the caller.
            if (config.pop_size >= _FEAS_GUARD_MIN_POP
                    and not os.environ.get("GP_DISABLE_FEASIBILITY_GUARD")):
                # Feasible := neg_sharpe strictly below the sentinel midpoint (5e5).
                # Infeasible fitness is 1e6 + violation (violation in [0,1), fitness.py
                # graded-infeasibility), so every infeasible value is >= 1e6 > 5e5; real
                # feasible neg_sharpe is ~±tens (the Sharpe-std 1e-9 floor is non-binding,
                # evaluator_vectorized.py). 5e5 is the midpoint of that [~±50, 1e6) gap and
                # MATCHES experiment.py's heartbeat feasible_count definition (same concept,
                # same threshold) — keep them identical so observability and the guard agree.
                _n_feasible = sum(
                    1 for _i in population
                    if _i.fitness is not None
                    and float(_i.fitness[0]) < FAILED_FITNESS_SENTINEL * 0.5
                )
                _consecutive_zero_feasible = (
                    _consecutive_zero_feasible + 1 if _n_feasible == 0 else 0
                )
                if _consecutive_zero_feasible >= _FEAS_GUARD_CONSECUTIVE_GENS:
                    _wb = getattr(fitness_evaluator, "warmup_bars", "?")
                    raise RuntimeError(
                        f"[feasibility guard] {template.name}: 0/{len(population)} "
                        f"individuals feasible for {_consecutive_zero_feasible} "
                        f"consecutive generations (through gen {gen}); every tree is at "
                        f"the FAILED_FITNESS sentinel, so NSGA-III cannot select and the "
                        f"run would silently return an arbitrary churn front. Most likely "
                        f"cause: warmup_bars ({_wb}) >= the data's bars-per-day, so the "
                        f"PER-DAY fire-rate / conditional-Sharpe gates see 0 eligible bars "
                        f"and sentinel every individual. Also check min_trades "
                        f"(={fitness_evaluator.min_trades}) vs available bars, and "
                        f"terminal/data wiring. Aborting fast instead of burning the full "
                        f"run. (Set GP_DISABLE_FEASIBILITY_GUARD=1 to override.)"
                    )

            # L2 grammar fix: compute dominance ranks for tournament selection.
            # Non-dominated sorting assigns rank 0 to Pareto front, rank 1
            # to next front, etc. Attached to each Individual for O(1) lookup
            # during tournament selection.
            F_pop = np.array([ind.fitness for ind in population])
            pop_fronts = NonDominatedSorting().do(F_pop, return_rank=False)
            for rank_idx, front in enumerate(pop_fronts):
                for idx in front:
                    population[int(idx)]._nds_rank = rank_idx
            # Compute crowding distance within each front for tiebreaking
            for front_arr in pop_fronts:
                front_list = list(front_arr)
                if len(front_list) <= 2:
                    for idx in front_list:
                        population[int(idx)]._crowding_dist = float('inf')
                    continue
                F_front = F_pop[front_list]
                n_obj_front = F_front.shape[1]
                crowding = np.zeros(len(front_list))
                for m in range(n_obj_front):
                    sorted_idx = np.argsort(F_front[:, m])
                    crowding[sorted_idx[0]] = float('inf')
                    crowding[sorted_idx[-1]] = float('inf')
                    f_range = F_front[sorted_idx[-1], m] - F_front[sorted_idx[0], m]
                    if f_range < 1e-12:
                        continue
                    for j in range(1, len(sorted_idx) - 1):
                        crowding[sorted_idx[j]] += (
                            F_front[sorted_idx[j + 1], m] - F_front[sorted_idx[j - 1], m]
                        ) / f_range
                for i, idx in enumerate(front_list):
                    population[int(idx)]._crowding_dist = float(crowding[i])

            # Generation metrics
            metrics = compute_metrics(gen, population, fitness_evaluator,
                                       time.time() - gen_start)
            metrics_log.append(metrics)
            if on_generation is not None:
                on_generation(metrics, population)

            # Inner validation checkpoint (task #34): evaluate Pareto front
            # on validation data every val_checkpoint_interval generations.
            # OBSERVATION ONLY — val fitness is NOT used for NSGA-III selection.
            if (val_evaluator is not None
                    and gen % val_checkpoint_interval == 0
                    and gen > 0):
                try:
                    _front_for_val = pareto_front(population)
                    if _front_for_val:
                        _val_sharpe = val_evaluator(_front_for_val)
                        _val_checkpoints.append((gen, float(_val_sharpe)))
                        if _val_sharpe > _best_val_sharpe:
                            _best_val_sharpe = _val_sharpe
                            _best_val_generation = gen
                            # Deep-copy the best-val front for snapshot
                            _best_val_front_snapshot = [
                                Individual(
                                    entry_tree=deep_copy(ind.entry_tree),
                                    exit_tree=deep_copy(ind.exit_tree),
                                    size_tree=deep_copy(ind.size_tree),
                                    template_name=ind.template_name,
                                    delta_tree=(deep_copy(ind.delta_tree)
                                                if ind.delta_tree else None),
                                    stop_mult=ind.stop_mult,  # preserve evolved stop gene
                                    fitness=ind.fitness.copy() if ind.fitness is not None else None,
                                    age=ind.age,
                                )
                                for ind in _front_for_val
                            ]
                            _val_stagnation_warned = False
                        elif (gen - _best_val_generation >= val_patience
                              and not _val_stagnation_warned):
                            warnings.warn(
                                f"Val stagnation: best val_sharpe "
                                f"{_best_val_sharpe:.4f} at gen "
                                f"{_best_val_generation}, current gen {gen} "
                                f"({gen - _best_val_generation} gens without "
                                f"improvement). GP may be overfitting to "
                                f"training data.",
                                stacklevel=2,
                            )
                            _val_stagnation_warned = True

                        # Early stopping: if the train-val gap > 1.0 for 3
                        # consecutive checkpoints (30 gens), stop evolution.
                        # Population-level early stop — does NOT contaminate
                        # OOS (no individual uses val for selection).
                        #
                        # B1 FIX (two coupled bugs that truncated EVERY run to
                        # ~gen 31 of 100):
                        #
                        # (a) Apples-to-oranges comparator. `_val_sharpe` is the
                        # MEAN of (-fitness[0]) over the Pareto FRONT, with
                        # sentinel individuals (<= -1e5, e.g. regime-gated)
                        # filtered out (see experiment._val_evaluator_fn).
                        # The old train side was -min(fitness[0]) over the
                        # WHOLE population — i.e. the single BEST (max) train
                        # Sharpe, penalty-laden, unfiltered. Max-vs-mean on
                        # different sets inflates the gap by construction.
                        # FIX: compute the train side the SAME way — mean of
                        # (-fitness[0]) over the SAME front (_front_for_val),
                        # same sentinel filter — so the gap is a true
                        # like-for-like train-minus-val on identical members.
                        #
                        # (b) Firing with no in-sample edge to overfit. On a
                        # LOSING fold (train front Sharpe <= 0, val even
                        # worse) the gap is large and positive, but that is
                        # ordinary OOS degradation, NOT overfitting — there
                        # is no in-sample edge being memorised. FIX: require
                        # a POSITIVE in-sample edge (train front Sharpe > 0)
                        # as a PRECONDITION before the gap may increment the
                        # overfit counter; otherwise reset it. Overfitting is
                        # only meaningful when there is in-sample edge to
                        # lose. Genuine overfit (train front Sharpe > 0, val
                        # much worse) STILL trips the gate.
                        _train_front_sharpes = [
                            -float(ind.fitness[0]) for ind in _front_for_val
                            if ind.fitness is not None
                            and -float(ind.fitness[0]) > -1e5  # match val filter
                        ]
                        _train_front_sharpe = (
                            float(np.mean(_train_front_sharpes))
                            if _train_front_sharpes else 0.0
                        )
                        _tv_gap = _train_front_sharpe - _val_sharpe
                        # Precondition (b): only an in-sample EDGE can be
                        # overfit. With no positive edge, a large gap is OOS
                        # degradation, not overfitting — reset, never stop.
                        _OVERFIT_EDGE_FLOOR = 0.0
                        if _train_front_sharpe > _OVERFIT_EDGE_FLOOR and _tv_gap > 1.0:
                            _consecutive_overfit_checks += 1
                        else:
                            _consecutive_overfit_checks = 0
                        if _consecutive_overfit_checks >= 3:
                            warnings.warn(
                                f"Early stop at gen {gen}: train-val gap "
                                f"{_tv_gap:.2f} (train_front={_train_front_sharpe:.2f},"
                                f" val={_val_sharpe:.2f}) with positive in-sample "
                                f"edge persisted for {_consecutive_overfit_checks} "
                                f"consecutive checkpoints. Stopping to prevent "
                                f"further overfitting.",
                                stacklevel=2,
                            )
                            break  # exit the generation loop
                except Exception as _val_exc:
                    import sys
                    print(f"[inner-val] gen {gen}: {type(_val_exc).__name__}: "
                          f"{_val_exc}", file=sys.stderr)

            # Reproduce: tournament selection → crossover → mutation
            # Level B: 4 trees → higher total cap. V1 (3 trees) also gets
            # the updated formula but the practical cap is the same since
            # min(40, 3*12) = 36 > min(25, 3*12) = 25 for V1 configs.
            _MAX_TOTAL_NODES = min(40, 4 * grammar.max_nodes)

            def _is_offspring_invalid(ind):
                if ind.total_nodes() > _MAX_TOTAL_NODES:
                    return True
                if is_structurally_degenerate(ind.entry_tree):
                    return True
                if is_structurally_degenerate(ind.exit_tree):
                    return True
                return False

            # Adaptive mutation: full rate for first 2/3 of generations,
            # then linearly decay to 50% for the last 1/3. Shifts from
            # exploration to exploitation in late stages, allowing EphReal
            # Gaussian perturbation to fine-tune thresholds.
            _gen_frac = gen / max(config.n_generations - 1, 1)
            if _gen_frac > 0.67:
                _late_decay = 1.0 - 0.5 * (_gen_frac - 0.67) / 0.33
                _effective_mut_rate = config.mutation_rate * _late_decay
            else:
                _effective_mut_rate = config.mutation_rate

            offspring: List[Individual] = []
            while len(offspring) < config.pop_size:
                pa = _tournament_select(population, config.tournament_size)
                pb = _tournament_select(population, config.tournament_size)
                if random.random() < config.crossover_rate:
                    child_a, child_b = crossover(pa, pb, grammar)
                else:
                    child_a = Individual(
                        deep_copy(pa.entry_tree), deep_copy(pa.exit_tree),
                        deep_copy(pa.size_tree), pa.template_name,
                        delta_tree=deep_copy(pa.delta_tree) if pa.delta_tree else None,
                        stop_mult=pa.stop_mult,  # inherit parent's stop gene
                    )
                    child_b = Individual(
                        deep_copy(pb.entry_tree), deep_copy(pb.exit_tree),
                        deep_copy(pb.size_tree), pb.template_name,
                        delta_tree=deep_copy(pb.delta_tree) if pb.delta_tree else None,
                        stop_mult=pb.stop_mult,  # inherit parent's stop gene
                    )
                child_a = mutate(child_a, grammar, _effective_mut_rate)
                child_b = mutate(child_b, grammar, _effective_mut_rate)
                # Reject offspring that violate structural or size constraints.
                # _MAX_TOTAL_NODES and _is_offspring_invalid defined outside loop
                # for performance (was being redefined 256× per generation).

                if _is_offspring_invalid(child_a):
                    # G2 fix: apply point-mutation to parent copy so offspring
                    # is genetically distinct. Cloning parents without mutation
                    # causes premature convergence (clone epidemic).
                    # Re-check validity: point_mutate can grow tree past cap.
                    fallback_a = Individual(
                        grammar.point_mutate(deep_copy(pa.entry_tree), p=0.3),
                        grammar.point_mutate(deep_copy(pa.exit_tree), p=0.3),
                        deep_copy(pa.size_tree), pa.template_name,
                        delta_tree=deep_copy(pa.delta_tree) if pa.delta_tree else None,
                        stop_mult=pa.stop_mult,  # inherit parent's stop gene
                    )
                    if _is_offspring_invalid(fallback_a):
                        fallback_a = Individual(
                            deep_copy(pa.entry_tree), deep_copy(pa.exit_tree),
                            deep_copy(pa.size_tree), pa.template_name,
                            delta_tree=deep_copy(pa.delta_tree) if pa.delta_tree else None,
                            stop_mult=pa.stop_mult,  # inherit parent's stop gene
                        )
                    child_a = fallback_a
                if _is_offspring_invalid(child_b):
                    fallback_b = Individual(
                        grammar.point_mutate(deep_copy(pb.entry_tree), p=0.3),
                        grammar.point_mutate(deep_copy(pb.exit_tree), p=0.3),
                        deep_copy(pb.size_tree), pb.template_name,
                        delta_tree=deep_copy(pb.delta_tree) if pb.delta_tree else None,
                        stop_mult=pb.stop_mult,  # inherit parent's stop gene
                    )
                    if _is_offspring_invalid(fallback_b):
                        fallback_b = Individual(
                            deep_copy(pb.entry_tree), deep_copy(pb.exit_tree),
                            deep_copy(pb.size_tree), pb.template_name,
                            delta_tree=deep_copy(pb.delta_tree) if pb.delta_tree else None,
                            stop_mult=pb.stop_mult,  # inherit parent's stop gene
                        )
                    child_b = fallback_b
                offspring.append(child_a)
                if len(offspring) < config.pop_size:
                    offspring.append(child_b)

            # Evaluate offspring fitness
            for ind in offspring:
                if ind.fitness is None:
                    ind.fitness = fitness_evaluator.evaluate(
                        ind.entry_tree, ind.exit_tree, ind.size_tree,
                        delta_tree=ind.delta_tree,
                        stop_mult=ind.stop_mult,
                    )

            # Combined population for environmental selection (μ + λ)
            combined = population + offspring
            population = nsga3_select(combined, ref_dirs, config.pop_size,
                                       rng=niching_rng)
            # Age survivors
            for ind in population:
                ind.age += 1

            # Generation-level checkpoint (CLEAN boundary): generation `gen` is
            # now fully complete (selected + aged) and all three RNG streams
            # hold exactly the state an uninterrupted run would have at the top
            # of iteration `gen + 1`. Writing _capture_state(gen + 1) here makes
            # resume bit-identical to a run that was never interrupted. Atomic
            # (temp + os.replace) so a reader never sees a torn file. Write
            # every K generations AND on the final generation.
            if _ckpt_dir and (
                (gen + 1) % max(1, checkpoint_every) == 0
                or gen == config.n_generations - 1
            ):
                try:
                    _write_checkpoint_atomic(_ckpt_dir, template.name,
                                             _capture_state(gen + 1))
                except Exception as _ck_exc:  # never let checkpointing kill a run
                    warnings.warn(
                        f"[checkpoint] write at gen {gen} failed "
                        f"({type(_ck_exc).__name__}: {_ck_exc}); evolution "
                        f"continues but this generation is not resumable.",
                        stacklevel=2,
                    )

        # : Convergence warning for search space asymmetry.
        # Conditions C/D have 10-100x larger search spaces than A but run
        # the same generation budget. If fitness is still improving at the
        # final generation, the run may be under-converged.
        if len(metrics_log) >= 20:
            # Compare best Sharpe (objective 0, negated) over last 20 gens
            _last20 = [m.best_per_objective[0] for m in metrics_log[-20:]]
            _first10 = _last20[:10]
            _final10 = _last20[10:]
            _improvement = np.mean(_first10) - np.mean(_final10)  # positive = still improving
            if _improvement > 0.05:
                warnings.warn(
                    f"Convergence warning: best Sharpe improved by "
                    f"{_improvement:.3f} over the last 20 generations "
                    f"(gens {len(metrics_log)-20}-{len(metrics_log)-1}). "
                    f"Evolution may be under-converged — consider extending "
                    f"n_generations for this condition/template.",
                    stacklevel=2,
                )

        # Build inner validation tracking dict (task #34).
        # Returned to caller for heartbeat logging and best-val snapshot saving.
        _val_tracking: Dict = {
            "val_checkpoints": _val_checkpoints,  # [(gen, mean_val_sharpe), ...]
            "best_val_sharpe": float(_best_val_sharpe) if _best_val_sharpe > -float('inf') else None,
            "best_val_generation": _best_val_generation if _best_val_front_snapshot else None,
            "best_val_front": _best_val_front_snapshot,  # List[Individual] or None
            # P1-B DSR gate: {canonical_sig: (raw_train_sharpe_ann, n_days)} for
            # every distinct individual evaluated this run. N = len(this dict),
            # V = Var(daily Sharpes) over members with n_days >= min_days.
            "evo_trial_records": dict(
                getattr(fitness_evaluator, "evo_trial_records", {})),
        }

        # Reaching this normal return is a legitimate TERMINAL state — either
        # all generations completed, or the inner-val overfit gate `break`'d out
        # (an intentional early stop, not an interruption). In both cases the
        # checkpoint is obsolete: delete it so a future re-run with the same
        # output dir starts fresh rather than instantly "resuming" a finished
        # run. ONLY a SIGTERM/SIGHUP interruption (which raises _CheckpointAndExit
        # and unwinds through the except branch below) KEEPS its checkpoint.
        if _ckpt_dir:
            _delete_checkpoint(_ckpt_dir, template.name)

        _restore_signal_handlers()
        return population, metrics_log, _val_tracking
    except _CheckpointAndExit:
        # SIGTERM/SIGHUP: the handler already flushed a checkpoint. Restore the
        # prior signal handlers, release the lock (via the outer finally) and
        # exit the process cleanly so the caller / shell sees a normal stop.
        _rsh = locals().get("_restore_signal_handlers")
        if _rsh is not None:
            try:
                _rsh()
            except Exception:
                pass
        import sys as _sys
        _sys.exit(0)
    finally:
        # Defensive: ensure handlers are restored even on an unexpected raise
        # (idempotent; guarded because the helper may not be defined yet if an
        # exception fires during early setup before the loop).
        _rsh = locals().get("_restore_signal_handlers")
        if _rsh is not None:
            try:
                _rsh()
            except Exception:
                pass
        _EVOLVE_INFLIGHT_LOCK.release()


def _tournament_select(population: List[Individual], k: int) -> Individual:
    """k-way dominance-rank binary tournament.

    L2 grammar fix: replaced `sum(fitness)` with dominance-rank comparison.
    The old sum was dominated by neg_sharpe (range -10 to +1e6) while other
    objectives range [-1, 0], making tournament selection effectively
    single-objective. Now uses non-dominated sorting rank computed once per
    generation, with crowding distance as tiebreaker.

    For efficiency, dominance ranks are pre-computed in `evolve()` and
    attached to each Individual's `.rank` attribute. If ranks are not
    available (e.g., in standalone tests), falls back to normalized fitness sum.
    """
    competitors = random.sample(population, min(k, len(population)))
    def _sort_key(ind: Individual) -> tuple:
        rank = getattr(ind, '_nds_rank', None)
        if rank is not None:
            # Lower rank is better (front 0 = best), use crowding as tiebreaker
            crowd = getattr(ind, '_crowding_dist', 0.0)
            return (rank, -crowd)  # min rank, max crowding
        # Fallback when _nds_rank not set (first gen or standalone tests).
        # Raw sum is scale-dominated by neg_sharpe but this path should
        # rarely execute in production (evolve() sets ranks before tournament).
        f = ind.fitness
        return (float(np.sum(f)),)
    return min(competitors, key=_sort_key)


def pareto_front(population: List[Individual]) -> List[Individual]:
    """Extract the Pareto-optimal subset of a population."""
    F = np.array([ind.fitness for ind in population])
    fronts = NonDominatedSorting().do(F, return_rank=False)
    if not fronts:
        return []
    return [population[i] for i in fronts[0]]
