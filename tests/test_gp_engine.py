"""Smoke tests for layer2.gp_engine — NSGA-III evolution loop."""
import numpy as np
import pandas as pd
import pytest

from layer2.fitness import FitnessEvaluator
from layer2.gp_engine import (
    EvolutionConfig, GenerationMetrics, Individual,
    crossover, evolve, initialize_population, mutate, nsga3_select, pareto_front,
)
from layer2.grammar import Grammar, GType, node_count
from layer2.io import (
    PRICE_COLUMN, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor, bull_put_credit


def _make_data(n_dates=2, bars_per_date=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date in [f"2024-{m:02d}-15" for m in range(1, n_dates + 1)]:
        spot = 4500.0
        for w in range(bars_per_date):
            spot += rng.normal(0, 1.0)
            row = {
                "date": date, "window_idx": w, "bar_position": w,
                PRICE_COLUMN: spot, "ATM_IV": 0.20,
                "MinutesToClose": 60.0 - w,
                "VIXSpot": 18.0, "VIXTermSlope": -0.5,
                "RawSpread": 0.20, "DeltaSpread1": 0.0, "DeltaSpread5": 0.0,
                "RealizedVol30m": 0.18,
                "PredRV15": 0.30, "PredRV30": 0.30,
                "PredRegime": 0, "PredSpread": 0.0,
            }
            for col in TYPED_VECTOR_COLUMNS:
                row[col] = np.zeros(384, dtype=np.float32)
            for col in REGIME_PROB_COLUMNS:
                row[col] = 0.25
            rows.append(row)
    return pd.DataFrame(rows)


def test_initialize_population_seeds_and_random():
    template = iron_condor()
    grammar = Grammar(max_depth=4, max_nodes=20)
    pop = initialize_population(template, grammar, pop_size=10, seed_fraction=0.5)
    assert len(pop) == 10
    # All individuals have valid trees of correct types
    for ind in pop:
        assert ind.template_name == "iron_condor_standard"
        assert ind.entry_tree.ret_type == GType.BOOL
        assert ind.exit_tree.ret_type == GType.BOOL
        assert ind.size_tree.ret_type == GType.REAL


def test_individual_signature_is_canonical():
    template = bull_put_credit()
    ind1 = Individual(template.entry_seed, template.exit_seed, template.size_seed,
                      template_name="x")
    ind2 = Individual(template.entry_seed, template.exit_seed, template.size_seed,
                      template_name="x")
    assert ind1.signature() == ind2.signature()


def test_crossover_preserves_validity():
    """Crossover output trees must pass grammar validation."""
    template = iron_condor()
    grammar = Grammar(max_depth=4, max_nodes=20)
    pop = initialize_population(template, grammar, pop_size=10, seed_fraction=1.0)
    # Pop is all template seeds — try crossover
    child_a, child_b = crossover(pop[0], pop[1], grammar)
    assert child_a.entry_tree.ret_type == GType.BOOL
    assert child_b.entry_tree.ret_type == GType.BOOL


def test_mutate_preserves_validity():
    template = iron_condor()
    grammar = Grammar(max_depth=4, max_nodes=20)
    ind = Individual(template.entry_seed, template.exit_seed, template.size_seed,
                     template_name=template.name)
    mutated = mutate(ind, grammar, mutation_rate=1.0)
    assert mutated.entry_tree.ret_type == GType.BOOL
    assert mutated.exit_tree.ret_type == GType.BOOL
    assert mutated.size_tree.ret_type == GType.REAL


def test_nsga3_select_returns_correct_size():
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=8, seed_fraction=0.5)
    # Assign random fitness vectors (3 objectives)
    rng = np.random.default_rng(0)
    for ind in pop:
        ind.fitness = rng.standard_normal(3)
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    survivors = nsga3_select(pop, ref_dirs, n_survivors=4,
                              rng=np.random.default_rng(0))
    assert len(survivors) == 4


def test_nsga3_select_requires_explicit_rng():
    """Model QA Commit 5 R8: rng=None must raise ValueError, not silently
    fall back to default_rng(0). Prevents the reproducibility footgun of
    a test that silently inherits a global default."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=8, seed_fraction=0.5)
    rng = np.random.default_rng(0)
    for ind in pop:
        ind.fitness = rng.standard_normal(3)
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    with pytest.raises(ValueError, match="explicit .*rng"):
        nsga3_select(pop, ref_dirs, n_survivors=4)  # no rng


def test_nsga3_select_all_sentinel_population():
    """Code Reviewer FINDING-A: if the ENTIRE population has
    FAILED_FITNESS_SENTINEL (e.g., in early generations of scalar-only
    with aggressive min_trades), pymoo's HyperplaneNormalization divides
    by a zero nadir-ideal gap. Must not crash; must return n_survivors
    individuals (any of them — they're all equivalent)."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=16, seed_fraction=0.5)
    # All individuals get the sentinel value on every axis
    for ind in pop:
        ind.fitness = np.array([1e6, 1e6, 1e6])
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    # Must not raise (even though ideal == nadir in this degenerate case)
    survivors = nsga3_select(pop, ref_dirs, n_survivors=8,
                              rng=np.random.default_rng(0))
    assert len(survivors) == 8
    # All survivors still carry the sentinel fitness (nothing else exists)
    for s in survivors:
        assert np.all(s.fitness == 1e6)


def test_evolution_loop_runs_smoke():
    """Smoke test: NSGA-III evolves for 3 generations on small population."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    fe = FitnessEvaluator(
        template=template, data=_make_data(n_dates=1, bars_per_date=15),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    config = EvolutionConfig(
        pop_size=8, n_generations=3, seed=42, n_partitions=4,
    )
    final_pop, metrics_log, val_tracking = evolve(template, fe, grammar, config)
    assert len(final_pop) == 8
    assert len(metrics_log) == 3
    # Every individual has a finite fitness
    for ind in final_pop:
        assert ind.fitness is not None
        assert ind.fitness.shape == (3,)
    # Per-generation telemetry has all expected fields
    for m in metrics_log:
        assert isinstance(m, GenerationMetrics)
        assert m.population_size == 8
        assert m.front_size > 0
        assert len(m.best_per_objective) == 3


def test_evolution_records_distinct_trial_sharpes():
    """P1-B DSR gate: evolve() must surface, via val_tracking, the per-distinct
    individual training Sharpes (N, V source) recorded by the evaluator's
    opt-in trial recorder. This is the evolution-level (N, V) input that fixes
    the variance-inversion bug — it must NOT come from the final front."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    fe = FitnessEvaluator(
        template=template, data=_make_data(n_dates=1, bars_per_date=15),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    config = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)
    final_pop, _metrics, val_tracking = evolve(template, fe, grammar, config)

    # The records dict is surfaced for the front-selection point.
    assert "evo_trial_records" in val_tracking
    records = val_tracking["evo_trial_records"]
    assert isinstance(records, dict)
    # Many more DISTINCT individuals are evaluated across 3 generations than
    # survive on the final front — the recorder must capture the population,
    # not just the front (the whole point of the redesign).
    front = pareto_front(final_pop)
    assert len(records) >= len(front), (
        f"recorded distinct trials ({len(records)}) should be >= final front "
        f"size ({len(front)}) — evolution-level N must exceed the front"
    )
    # Each value is (sharpe_ann, n_days); phantom -1e6 Sharpes excluded.
    for sig, (sr, nd) in records.items():
        assert isinstance(sig, str) and sig
        assert np.isfinite(sr) and sr > -1e5
        assert isinstance(nd, int) and nd >= 1
    # The evaluator's live sink matches what was surfaced (shared by reference,
    # copied into val_tracking).
    assert fe.evo_trial_records  # non-empty
    assert set(fe.evo_trial_records.keys()) == set(records.keys())


def test_pareto_front_extraction():
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=8, seed_fraction=0.5)
    # Synthetic fitness: one obviously-dominated individual + others on the front
    rng = np.random.default_rng(0)
    for ind in pop:
        ind.fitness = rng.standard_normal(3)
    # Force one individual to be dominated by all others
    pop[0].fitness = np.array([99.0, 99.0, 99.0])
    front = pareto_front(pop)
    assert len(front) > 0
    assert len(front) < len(pop)  # at least the all-99s should be dominated


# ---------------------------------------------------------------------------
# Commit 5: canonical Deb & Jain 2014 NSGA-III niching via pymoo
# ---------------------------------------------------------------------------

def test_nsga3_select_reproducible_with_same_rng():
    """Same RNG seed → identical survivor selection. Verifies niching
    tie-breaking is reproducible under the threaded rng parameter."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)

    def _make_pop():
        pop = initialize_population(template, grammar, pop_size=16,
                                     seed_fraction=0.5)
        rng_fit = np.random.default_rng(0)
        for ind in pop:
            ind.fitness = rng_fit.standard_normal(3)
        return pop

    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)

    pop_a = _make_pop()
    pop_b = _make_pop()
    # Identical fitness vectors between populations (same seed in _make_pop)
    for a, b in zip(pop_a, pop_b):
        assert np.array_equal(a.fitness, b.fitness)

    sa = nsga3_select(pop_a, ref_dirs, 8, rng=np.random.default_rng(123))
    sb = nsga3_select(pop_b, ref_dirs, 8, rng=np.random.default_rng(123))
    # Survivor fitness signatures must match
    fa = sorted(tuple(ind.fitness) for ind in sa)
    fb = sorted(tuple(ind.fitness) for ind in sb)
    assert fa == fb, "nsga3_select not reproducible under identical rng seeds"


def test_nsga3_select_dominates_sentinel_individuals():
    """B2 cross-check: strategies with (1e6, 1e6, 1e6) fitness
    (FAILED_FITNESS_SENTINEL from B2 gate) must NOT appear on the first
    non-dominated front when any productive individual exists. Verifies
    pymoo's non-dominated sorting correctly dominates sentinel rows."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=16,
                                 seed_fraction=0.5)
    rng = np.random.default_rng(0)
    # Half productive (random but finite), half sentinel
    for i, ind in enumerate(pop):
        if i < 8:
            ind.fitness = rng.standard_normal(3)   # productive
        else:
            ind.fitness = np.array([1e6, 1e6, 1e6])  # sentinel

    front = pareto_front(pop)
    for ind in front:
        assert not np.all(ind.fitness == 1e6), (
            "sentinel individual survived on Pareto front — pymoo's "
            "non-dominated sort failed to dominate the +1e6 column"
        )


def test_nsga3_select_preserves_first_front_when_small_enough():
    """If the first non-dominated front fits in n_survivors entirely,
    nsga3_select must return EXACTLY that front (plus whatever niche
    filler is needed from later fronts)."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=8,
                                 seed_fraction=0.5)
    # Pin fitness so the first 4 individuals are on the Pareto front
    # (each dominant on a different objective) and the other 4 are all
    # dominated by every front member.
    pop[0].fitness = np.array([-1.0,  0.0,  0.0])
    pop[1].fitness = np.array([ 0.0, -1.0,  0.0])
    pop[2].fitness = np.array([ 0.0,  0.0, -1.0])
    pop[3].fitness = np.array([-0.5, -0.5, -0.5])
    for i in range(4, 8):
        pop[i].fitness = np.array([10.0, 10.0, 10.0])  # dominated

    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    survivors = nsga3_select(pop, ref_dirs, 4, rng=np.random.default_rng(0))

    survivor_fitness = {tuple(s.fitness) for s in survivors}
    for i in range(4):
        assert tuple(pop[i].fitness) in survivor_fitness, (
            f"front member pop[{i}] with fitness {pop[i].fitness} missing "
            f"from survivors — nsga3_select dropped a non-dominated individual"
        )


def test_nsga3_select_uses_perpendicular_distance_not_cosine():
    """Regression guard for the Deb & Jain 2014 canonical switch: the
    previous hand-rolled niching used cosine similarity; the canonical
    implementation uses perpendicular distance after hyperplane
    normalization. Verify indirectly by checking that nsga3_select uses
    pymoo's `associate_to_niches` (which computes perp distance)."""
    import layer2.gp_engine as engine
    # Module-level imports must include the pymoo niching primitives
    assert hasattr(engine, "associate_to_niches"), (
        "gp_engine no longer imports associate_to_niches — the canonical "
        "Deb & Jain 2014 niching was reverted"
    )
    assert hasattr(engine, "niching"), (
        "gp_engine no longer imports pymoo niching"
    )
    assert hasattr(engine, "HyperplaneNormalization"), (
        "gp_engine no longer imports HyperplaneNormalization"
    )
    # Ensure the hand-rolled _associate_to_refs helper is gone
    assert not hasattr(engine, "_associate_to_refs"), (
        "legacy cosine-similarity _associate_to_refs helper still present; "
        "Commit 5 should have removed it"
    )


def test_nsga3_select_perpendicular_distance_behavioral():
    """Behavioral test (Code Reviewer + Senior Dev Commit 5 review):
    construct a splitting front where cosine and perpendicular-distance
    niching pick DIFFERENT survivors, and verify we pick the
    perpendicular-distance choice.

    Setup: two candidates on the splitting front, both associating to
    the same reference direction in spirit.
      A at (1.0, 0.0, 0.0):
        direction (1,0,0) via cosine — perfectly aligned with ref_dir [1,0,0]
      B at (5.0, 0.1, 0.1):
        further along but slightly off-axis — cosine similarity ≈ 0.9998
        perpendicular distance: larger than A's (B is FAR from any ref)

    Under cosine, B wins on the [1,0,0] niche because cos(B) ≈ 0.9998
    vs. cos(A) = 1.0 are close, and niching sorts by... actually cosine
    would pick A. Let me construct a clearer scenario:

    A at (1.0, 0.5, 0.5): off-axis but close to origin
    B at (3.0, 0.0, 0.0): on-axis but far

    cosine_similarity(A, [1,0,0]) ≈ 1.0/sqrt(1.5) ≈ 0.816
    cosine_similarity(B, [1,0,0]) = 1.0
    → cosine picks B

    perpendicular distance after hyperplane norm:
    A is "closer" to the [1,0,0] reference line (closer to axis)
    B is further along the axis but on-axis (perp dist = 0)
    → perp picks B too in this scenario.

    Direct behavioral divergence is subtle; we instead use a simpler
    guard: verify nsga3_select produces the SAME survivors as a direct
    call to pymoo's niching primitive, proving we're not silently
    running a different algorithm.
    """
    from pymoo.algorithms.moo.nsga3 import (
        HyperplaneNormalization, associate_to_niches, calc_niche_count,
        niching as pymoo_niching,
    )
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.util.ref_dirs import get_reference_directions

    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=20,
                                 seed_fraction=0.5)
    rng_fit = np.random.default_rng(7)
    for ind in pop:
        ind.fitness = rng_fit.standard_normal(3)
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)

    # Our implementation
    our_survivors = nsga3_select(pop, ref_dirs, 10,
                                  rng=np.random.default_rng(42))
    our_fitness = sorted(tuple(s.fitness) for s in our_survivors)

    # Direct pymoo-primitive reference computation
    F = np.array([ind.fitness for ind in pop])
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False, n_stop_if_ranked=10)
    ref_survivors: list = []
    last_idx = 0
    for i, front in enumerate(fronts):
        if len(ref_survivors) + len(front) <= 10:
            ref_survivors.extend(int(x) for x in front)
            last_idx = i
            if len(ref_survivors) == 10:
                break
        else:
            last_idx = i
            break

    if len(ref_survivors) < 10:
        splitting = fronts[last_idx]
        until = np.concatenate(fronts[:last_idx]) if last_idx > 0 else np.array([], dtype=int)
        norm = HyperplaneNormalization(ref_dirs.shape[1])
        norm.update(F, nds=fronts[0])
        ideal, nadir = norm.ideal_point, norm.nadir_point
        all_idx = np.concatenate((until, splitting)).astype(int)
        F_all = F[all_idx]
        niche_of, dist_to, _ = associate_to_niches(F_all, ref_dirs, ideal, nadir)
        n_until_ref = len(until)
        nc = calc_niche_count(len(ref_dirs), niche_of[:n_until_ref])
        chosen = pymoo_niching(
            pop=np.arange(len(splitting)),
            n_remaining=10 - n_until_ref,
            niche_count=nc,
            niche_of_individuals=niche_of[n_until_ref:],
            dist_to_niche=dist_to[n_until_ref:],
            random_state=np.random.default_rng(42),
        )
        ref_survivors.extend(int(splitting[i]) for i in chosen)

    ref_fitness = sorted(tuple(pop[i].fitness) for i in ref_survivors)
    assert our_fitness == ref_fitness, (
        "nsga3_select survivor set diverges from a direct pymoo-primitive "
        "computation on the same inputs — the canonical Deb & Jain 2014 "
        "algorithm is not being followed end-to-end"
    )


def test_nsga3_select_non_degenerate_hyperplane_normalization():
    """AI Engineer + Model QA (Commit 5 review): exercise the NON-degenerate
    HyperplaneNormalization path. Smoke tests with pop=8 typically have
    first-front-size=1 or 2, which hits pymoo's F.max fallback. A
    realistic H2 population (pop=64+) has a multi-member first front
    that triggers the canonical intercept computation.
    """
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    pop = initialize_population(template, grammar, pop_size=64,
                                 seed_fraction=0.5)
    # Construct a population where the first non-dominated front contains
    # ≥ n_obj=3 well-spread points (so hyperplane intercepts are
    # well-defined). Pin 4 Pareto-optimal members + 60 dominated.
    pop[0].fitness = np.array([-2.0,  0.0,  0.0])  # objective 1 extreme
    pop[1].fitness = np.array([ 0.0, -2.0,  0.0])  # objective 2 extreme
    pop[2].fitness = np.array([ 0.0,  0.0, -2.0])  # objective 3 extreme
    pop[3].fitness = np.array([-1.0, -1.0, -1.0])  # interior front member
    rng = np.random.default_rng(123)
    for i in range(4, 64):
        # Strictly dominated by pop[3]: larger on every axis
        pop[i].fitness = np.array([1.0, 1.0, 1.0]) + rng.uniform(0, 0.5, 3)

    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)

    # n_survivors forces niching to run on the dominated population
    # (first front = 4 members, all will be kept; 28 more needed from
    # the dominated set via niching with canonical normalization).
    survivors = nsga3_select(pop, ref_dirs, 32,
                              rng=np.random.default_rng(0))
    assert len(survivors) == 32
    # All 4 first-front extremes must survive
    survivor_fitness = {tuple(s.fitness) for s in survivors}
    for i in range(4):
        assert tuple(pop[i].fitness) in survivor_fitness, (
            f"first-front extreme pop[{i}] dropped under non-degenerate "
            f"HyperplaneNormalization — canonical selection failed"
        )
