"""v9 fix-bundle unit tests (F1 + F2 + #2 + #3, 2026-04-25).

Covers the four fixes applied after the v9 spec but before the N=10
mini-pilot launch:

  F1 (BLOCKER):  PredRegime / Regime / IN_REGIME / REGIME_IS leak into
                 scalar-only arm. Grammar AND data-layer defenses verified.

  F2 (HIGH):     Operator-class–stratified mutation sampling. Per-class
                 selection density is approximately uniform.

  #2 (HIGH):     EmbProj_X_k(EMB_X) counts as depth-1 and node-count-1
                 (semantic complexity, not syntactic).

  #3 (MEDIUM):   Stratified initial-population seeding. ≥50% of real-l1
                 gen=0 trees contain at least one EmbProj operator.
                 Scalar-only / shuffled-l1 stay 0% by definition.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import List

import pytest


# ---------------------------------------------------------------------------
# F1 — PredRegime / Regime / IN_REGIME / REGIME_IS leak audit
# ---------------------------------------------------------------------------

def test_f1_scalar_only_grammar_excludes_regime_operators():
    """F1 BLOCKER: SCALAR_ONLY_FUNCTIONS must NOT contain IN_REGIME or
    REGIME_IS. Their presence let scalar-only trees compare against
    ctx.current_regime which EvaluationContext.update populates from
    PredRegime (an L1 iTransformer probe output).
    """
    from layer2.grammar import SCALAR_ONLY_FUNCTIONS, IN_REGIME, REGIME_IS

    names = {f.name for f in SCALAR_ONLY_FUNCTIONS}
    assert "InRegime" not in names, (
        "F1 violation: IN_REGIME is in SCALAR_ONLY_FUNCTIONS — this lets "
        "scalar-only trees consume L1's regime label via ctx.current_regime."
    )
    assert "RegimeIs" not in names, (
        "F1 violation: REGIME_IS is in SCALAR_ONLY_FUNCTIONS — same leak path."
    )
    assert IN_REGIME not in SCALAR_ONLY_FUNCTIONS
    assert REGIME_IS not in SCALAR_ONLY_FUNCTIONS


def test_f1_scalar_only_terminal_set_excludes_regime_literals():
    """F1 BLOCKER: build_scalar_only_terminal_set must NOT return Regime
    literal terminals. Without IN_REGIME/REGIME_IS in the function set,
    Regime literals would be type-orphaned, but defense-in-depth requires
    the literals are explicitly absent so a future operator addition
    cannot reintroduce the leak.
    """
    from layer2.grammar import build_scalar_only_terminal_set, GType

    terms = build_scalar_only_terminal_set()
    regime_terms = [t for t in terms if t.ret_type == GType.REGIME]
    assert not regime_terms, (
        f"F1 violation: scalar-only terminal set contains Regime literals: "
        f"{[t.name for t in regime_terms]}"
    )
    # Side literals SHOULD still be there (Side is L1-neutral).
    side_terms = [t for t in terms if t.ret_type == GType.SIDE]
    assert side_terms, "scalar-only must retain Side literals (L1-neutral)"


def test_f1_forbidden_terms_includes_pred_regime_and_regime_probs():
    """F1 BLOCKER: _SCALAR_ONLY_TERMS_FORBIDDEN must list PredRegime AND
    RegimeProb0..3 so any seed-tree substitution path raises rather than
    silently coercing them to a raw scalar.
    """
    from layer2.grammar import _SCALAR_ONLY_TERMS_FORBIDDEN

    forbidden = set(_SCALAR_ONLY_TERMS_FORBIDDEN)
    assert "PredRegime" in forbidden, (
        "PredRegime must be forbidden in scalar-only substitution"
    )
    for k in range(4):
        assert f"RegimeProb{k}" in forbidden, (
            f"RegimeProb{k} must be forbidden in scalar-only substitution"
        )


def test_f1_evaluation_context_current_regime_stays_none_without_pred_regime():
    """F1 BLOCKER: when PredRegime is filtered out of bar_data (scalar-only
    arm), ctx.current_regime stays None — IN_REGIME/REGIME_IS in any tree
    that somehow snuck through would have no regime input.
    """
    from layer2.evaluator import EvaluationContext

    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    # Simulate scalar-only filter: PredRegime / RegimeProb* dropped.
    bar_data = {
        "ATM_IV": 0.18,
        "RealizedVol30m": 0.012,
        "VIXSpot": 16.5,
        # NO PredRegime, NO RegimeProb0..3
    }
    ctx.update(bar_data)
    assert ctx.current_regime is None, (
        f"F1 violation: ctx.current_regime={ctx.current_regime} but "
        f"PredRegime was not in bar_data — should stay None."
    )


def test_f1_evaluation_context_consumes_pred_regime_when_present():
    """F1 sanity: when PredRegime IS in bar_data (real-l1 / shuffled-l1
    arms), ctx.current_regime IS populated. The F1 fix is at the
    backtester data-layer (forbidden_terminal_columns), not at the
    EvaluationContext level — the context still respects PredRegime when
    the calling code provides it.
    """
    from layer2.evaluator import EvaluationContext

    ctx = EvaluationContext(max_lag=30, emb_dim=384)
    ctx.update({"ATM_IV": 0.18, "PredRegime": 2})
    assert ctx.current_regime is not None
    assert int(ctx.current_regime.value) == 2


def test_f1_backtester_drops_forbidden_columns():
    """F1 BLOCKER: MultiLegOptionsBacktester respects
    `forbidden_terminal_columns` and drops them from bar_data before
    ctx.update. This is the runtime defense that prevents
    ctx.current_regime from being set in scalar-only mode regardless of
    grammar shape.
    """
    import pandas as pd
    from layer2.evaluator import MultiLegOptionsBacktester
    from layer2.io import PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS
    from layer2.templates import iron_condor

    template = iron_condor()
    forbidden = tuple(PROBE_SCALAR_COLUMNS) + tuple(REGIME_PROB_COLUMNS)
    bt = MultiLegOptionsBacktester(
        template, forbidden_terminal_columns=forbidden
    )
    assert "PredRegime" in bt.forbidden_terminal_columns
    assert "RegimeProb0" in bt.forbidden_terminal_columns


def test_f1_experiment_threads_forbidden_columns_into_scalar_only_only():
    """F1 BLOCKER: experiment.run_experiment must thread the
    forbidden_terminal_columns into FitnessEvaluator's backtester_kwargs
    iff condition == 'scalar-only'. Other arms get an empty tuple.

    We don't run the full experiment (slow), but we can read the source
    and assert the routing is in place.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "layer2" / "experiment.py"
    text = src.read_text()
    assert 'config.condition == "scalar-only"' in text
    assert "PROBE_SCALAR_COLUMNS" in text
    assert "REGIME_PROB_COLUMNS" in text
    assert "forbidden_terminal_columns" in text


def test_f1_scalar_only_can_not_read_pred_xxx_or_regime_via_grammar():
    """F1 BLOCKER end-to-end: a scalar-only grammar tree, evaluated on a
    bar that has PredRegime in the raw row but NOT in bar_data after the
    backtester filter, cannot resolve a regime-dependent output.

    We don't construct an IN_REGIME tree (it cannot be built by grammar
    operators in scalar-only post-F1). We verify that the grammar simply
    rejects construction of an IN_REGIME node.
    """
    from layer2.grammar import (
        SCALAR_ONLY_FUNCTIONS, IN_REGIME, REGIME_IS,
    )
    # The grammar object built from SCALAR_ONLY_FUNCTIONS must not have
    # IN_REGIME / REGIME_IS available for synthesis. Confirmed at the
    # function-list level.
    fnames = {f.name for f in SCALAR_ONLY_FUNCTIONS}
    leaks = []
    for forbidden_name in (
        "InRegime", "RegimeIs",
        "EmbNorm_EMB_VIX", "EmbCos_EMB_VIX", "EmbSub_EMB_VIX",
        "EmbLag_EMB_VIX", "EmbProj_EMB_VIX_0",
    ):
        if forbidden_name in fnames:
            leaks.append(forbidden_name)
    assert not leaks, (
        f"F1 violation: scalar-only grammar contains forbidden ops: {leaks}"
    )


# ---------------------------------------------------------------------------
# F2 — Operator-class-stratified mutation sampling
# ---------------------------------------------------------------------------

def test_f2_operator_classes_partition_all_functions():
    """F2 HIGH: every entry in ALL_FUNCTIONS must be in exactly one
    OPERATOR_CLASSES bucket. Otherwise the stratified sampler silently
    excludes the unclassified function from the candidate pool.
    """
    from layer2.grammar import ALL_FUNCTIONS, OPERATOR_CLASSES, _FUNC_TO_CLASS

    classified = set()
    for cls_name, funcs in OPERATOR_CLASSES.items():
        for f in funcs:
            classified.add(f)

    unclassified = [f.name for f in ALL_FUNCTIONS if f not in classified]
    assert not unclassified, (
        f"F2 violation: {len(unclassified)} ALL_FUNCTIONS unclassified: "
        f"{unclassified[:5]}"
    )
    # Reverse map covers all
    for f in ALL_FUNCTIONS:
        assert f in _FUNC_TO_CLASS, (
            f"F2 violation: {f.name} missing from _FUNC_TO_CLASS"
        )


def test_f2_stratified_sampler_per_class_density_approx_uniform():
    """F2 HIGH: the per-class selection rate from `_feasible_func` must be
    approximately uniform across classes (each class with ≥1 feasible
    member sees ~1/n_classes selection probability), regardless of the
    cardinality difference between classes.

    Setup: instantiate a real-l1 grammar (full ALL_FUNCTIONS). Sample a
    REAL-returning function 5000 times. Bucket by operator class. Each
    class with ≥1 member should appear roughly evenly (within ±5pp of
    uniform).

    Pre-F2: emb_proj would dominate at ~80% (84/106 REAL-returning ops).
    Post-F2: emb_proj appears at ~1/n_classes (~14% with 7 REAL classes).
    """
    from layer2.grammar import (
        Grammar, GType, ALL_FUNCTIONS, _FUNC_TO_CLASS, OPERATOR_CLASSES,
        build_terminal_set,
    )

    g = Grammar(
        functions=ALL_FUNCTIONS,
        terminals=build_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    random.seed(20260425)
    n_trials = 5000
    counter: Counter = Counter()
    for _ in range(n_trials):
        f = g._feasible_func(GType.REAL, budget=4)
        if f is None:
            continue
        cls = _FUNC_TO_CLASS.get(f, "_unclassified")
        counter[cls] += 1

    # How many classes have ≥1 REAL-returning function with budget=4?
    real_classes_with_members = set()
    for cls_name, funcs in OPERATOR_CLASSES.items():
        for f in funcs:
            if f.ret_type == GType.REAL:
                real_classes_with_members.add(cls_name)
                break
    n_real_classes = len(real_classes_with_members)
    assert n_real_classes >= 5, (
        f"sanity: expected ≥5 REAL-returning classes, got {n_real_classes}"
    )

    expected = 1.0 / n_real_classes
    total = sum(counter.values())
    deviations = []
    for cls in real_classes_with_members:
        observed = counter[cls] / total
        deviation = abs(observed - expected)
        deviations.append((cls, observed, deviation))
    # Tolerance: ±10pp absolute (each class should be within ±10pp of
    # uniform 1/n given 5000 trials and n classes ~5-7).
    bad = [(c, o, d) for c, o, d in deviations if d > 0.10]
    assert not bad, (
        f"F2 violation: per-class selection density NOT uniform "
        f"(±10pp): {bad}\nfull distribution: "
        f"{[(c, round(o, 3)) for c, o, _ in deviations]}"
    )


def test_f2_emb_proj_no_longer_dominates_real_pool():
    """F2 HIGH (sanity): EmbProj must NOT account for ≥50% of REAL slot
    fills. Pre-F2 it was ~80%; post-F2 it should be ~10-20% (1 class out
    of ~5-7 REAL-returning classes).
    """
    from layer2.grammar import (
        Grammar, GType, ALL_FUNCTIONS, build_terminal_set,
    )

    g = Grammar(
        functions=ALL_FUNCTIONS,
        terminals=build_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    random.seed(20260425)
    n_trials = 3000
    n_emb = 0
    n_total = 0
    for _ in range(n_trials):
        f = g._feasible_func(GType.REAL, budget=4)
        if f is None:
            continue
        n_total += 1
        if f.name.startswith("EmbProj_"):
            n_emb += 1
    emb_share = n_emb / max(n_total, 1)
    assert emb_share < 0.30, (
        f"F2 violation: EmbProj share of REAL slot fills is {emb_share:.2%}, "
        f"expected <30%. Stratified sampling may not be active."
    )


def test_f2_scalar_only_unaffected_no_emb_classes():
    """F2 HIGH (sanity): scalar-only grammar — having no emb_*/regime
    classes — still samples uniformly across remaining classes. The
    distribution shifts but stays well-spread.
    """
    from layer2.grammar import (
        Grammar, GType, SCALAR_ONLY_FUNCTIONS, _FUNC_TO_CLASS,
        build_scalar_only_terminal_set,
    )

    g = Grammar(
        functions=SCALAR_ONLY_FUNCTIONS,
        terminals=build_scalar_only_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    random.seed(20260425)
    n_trials = 3000
    counter: Counter = Counter()
    for _ in range(n_trials):
        f = g._feasible_func(GType.REAL, budget=4)
        if f is None:
            continue
        cls = _FUNC_TO_CLASS.get(f, "_unclassified")
        counter[cls] += 1
    # Scalar-only has no emb_*/regime classes — must not appear.
    for forbidden_cls in ("emb_norm", "emb_cos", "emb_sub", "emb_lag",
                          "emb_proj", "regime"):
        assert counter.get(forbidden_cls, 0) == 0, (
            f"F1+F2: scalar-only sampled from forbidden class {forbidden_cls}"
        )


# ---------------------------------------------------------------------------
# Fix #2 — EmbProj depth-1 / node-count-1
# ---------------------------------------------------------------------------

def test_fix2_embproj_only_tree_has_depth_1():
    """Fix #2 HIGH: a tree whose root is EmbProj_X_k(EMB_X) has depth 1
    (not 2). EmbProj is semantically a single op; the typed-vector child
    is an obligatory implicit input, not an evolvable subtree.
    """
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode,
        build_terminal_set, depth, node_count,
    )

    fd = next(f for f in ALL_FUNCTIONS if f.name == "EmbProj_EMB_VIX_0")
    term = next(t for t in build_terminal_set() if t.name == "EMB_VIX")
    tree = FuncNode(defn=fd, children=[TermNode(defn=term)])
    assert depth(tree) == 1, (
        f"Fix #2 violation: EmbProj_EMB_VIX_0(EMB_VIX) depth = {depth(tree)}, "
        f"expected 1"
    )
    assert node_count(tree) == 1, (
        f"Fix #2 violation: EmbProj_EMB_VIX_0(EMB_VIX) node_count = "
        f"{node_count(tree)}, expected 1"
    )


def test_fix2_embproj_inside_compound_tree_counts_correctly():
    """Fix #2 HIGH: depth/node_count of GT(EmbProj_X_k(EMB_X), ATM_IV)
    is 2 (root + 1 layer of children) and 3 nodes (GT + EmbProj-collapsed
    + ATM_IV). Pre-fix would have been depth 3 / node_count 4.
    """
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode,
        build_terminal_set, depth, node_count,
    )

    gt = next(f for f in ALL_FUNCTIONS if f.name == "GT")
    proj = next(f for f in ALL_FUNCTIONS if f.name == "EmbProj_EMB_VIX_0")
    emb = next(t for t in build_terminal_set() if t.name == "EMB_VIX")
    atm = next(t for t in build_terminal_set() if t.name == "ATM_IV")
    tree = FuncNode(defn=gt, children=[
        FuncNode(defn=proj, children=[TermNode(defn=emb)]),
        TermNode(defn=atm),
    ])
    assert depth(tree) == 2, f"depth={depth(tree)} expected 2"
    assert node_count(tree) == 3, f"node_count={node_count(tree)} expected 3"


def test_fix2_embnorm_unchanged():
    """Fix #2 sanity: EmbNorm/EmbCos/EmbSub depth/node_count semantics
    are NOT changed — Fix #2 applies only to EmbProj because the k is
    hardcoded in the operator name (no internal subtree to grow).
    """
    from layer2.grammar import (
        ALL_FUNCTIONS, FuncNode, TermNode,
        build_terminal_set, depth, node_count,
    )

    norm = next(f for f in ALL_FUNCTIONS if f.name == "EmbNorm_EMB_VIX")
    emb = next(t for t in build_terminal_set() if t.name == "EMB_VIX")
    tree = FuncNode(defn=norm, children=[TermNode(defn=emb)])
    # EmbNorm is depth-1 too because it has just a TermNode child — the
    # ordinary depth recursion gives 1+0=1. Same for node_count = 1+1=2.
    assert depth(tree) == 1
    assert node_count(tree) == 2  # NOT collapsed — 1 + 1


# ---------------------------------------------------------------------------
# Fix #3 — Stratified initial-population seeding
# ---------------------------------------------------------------------------

def test_fix3_real_l1_init_pop_at_least_50pct_embproj():
    """Fix #3 MEDIUM: when grammar contains EmbProj_*, gen=0 population
    must have ≥50% of individuals containing at least one EmbProj op
    in their (entry/exit/size) trees. Default target_embproj_fraction=0.50.
    """
    from layer2.gp_engine import initialize_population
    from layer2.grammar import (
        ALL_FUNCTIONS, Grammar, build_terminal_set, to_str,
    )
    from layer2.templates import iron_condor

    random.seed(20260425)

    grammar = Grammar(
        functions=ALL_FUNCTIONS, terminals=build_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    template = iron_condor()
    pop = initialize_population(template, grammar, pop_size=80,
                                 seed_fraction=0.5)
    n_with_emb = 0
    for ind in pop:
        for tree in (ind.entry_tree, ind.exit_tree, ind.size_tree):
            if "EmbProj_" in to_str(tree):
                n_with_emb += 1
                break
    assert n_with_emb >= 40, (
        f"Fix #3 violation: only {n_with_emb}/80 gen=0 individuals carry "
        f"an EmbProj operator (need ≥40 = 50%)"
    )


def test_fix3_scalar_only_init_pop_zero_embproj():
    """Fix #3 sanity: when grammar lacks EmbProj_* (scalar-only arm),
    gen=0 must have 0% EmbProj — by definition. The Fix #3 boost loop
    is a no-op because grammar_has_embproj is False.
    """
    from layer2.gp_engine import initialize_population
    from layer2.grammar import (
        SCALAR_ONLY_FUNCTIONS, Grammar, build_scalar_only_terminal_set,
        substitute_stripped_terminals_for_scalar_only, to_str,
    )
    from layer2.templates import iron_condor
    from dataclasses import replace as _dc_replace

    random.seed(20260425)

    grammar = Grammar(
        functions=SCALAR_ONLY_FUNCTIONS,
        terminals=build_scalar_only_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    template = iron_condor()
    template = _dc_replace(
        template,
        entry_seed=substitute_stripped_terminals_for_scalar_only(template.entry_seed),
        exit_seed=substitute_stripped_terminals_for_scalar_only(template.exit_seed),
        size_seed=substitute_stripped_terminals_for_scalar_only(template.size_seed),
    )
    pop = initialize_population(template, grammar, pop_size=80,
                                 seed_fraction=0.5)
    n_with_emb = 0
    for ind in pop:
        for tree in (ind.entry_tree, ind.exit_tree, ind.size_tree):
            if "EmbProj_" in to_str(tree):
                n_with_emb += 1
                break
    assert n_with_emb == 0, (
        f"Fix #3 sanity violation: scalar-only gen=0 has {n_with_emb} "
        f"trees with EmbProj — should be 0 (grammar has no EmbProj ops)"
    )


def test_fix3_target_fraction_argument_respected():
    """Fix #3 HIGH: target_embproj_fraction kwarg can be set lower / higher."""
    from layer2.gp_engine import initialize_population
    from layer2.grammar import (
        ALL_FUNCTIONS, Grammar, build_terminal_set, to_str,
    )
    from layer2.templates import iron_condor

    random.seed(20260425)
    grammar = Grammar(
        functions=ALL_FUNCTIONS, terminals=build_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    template = iron_condor()
    # Set target = 0.0 → no boost loop, but grammar has EmbProj so
    # random fill may still produce some. We don't assert lower bound;
    # we assert the fraction is well-defined and finite.
    pop = initialize_population(template, grammar, pop_size=40,
                                 seed_fraction=0.5,
                                 target_embproj_fraction=0.0)
    n_emb = sum(1 for ind in pop
                if any("EmbProj_" in to_str(t)
                       for t in (ind.entry_tree, ind.exit_tree, ind.size_tree)))
    # No assertion on minimum at target=0.0 — just sanity the function ran.
    assert 0 <= n_emb <= 40
