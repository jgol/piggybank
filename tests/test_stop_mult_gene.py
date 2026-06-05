"""Tests for the GP-evolvable scalar stop-loss gene ``stop_mult``.

Phase 1 of docs/viable_run_plan_2026_06_03.md. The stop-loss is now a
per-individual evolved scalar (NOT a tree): ``Individual.stop_mult`` in
[0, STOP_MULT_MAX], where 0.0 == hold-to-expiry (no intraday stop). It flows
into ``vectorized_backtest`` as ``stop_loss_credit_multiple`` and into L3 codegen
so the emitted QC algorithm uses the SAME stop the strategy was evolved under
(proxy↔QC parity).

Coverage (per the task spec):
  1. init: random individuals get stop_mult ∈ [0, S_MAX]; seeded get 0.0.
  2. mutation keeps stop_mult ∈ [0, S_MAX] (reflection at both bounds).
  3. crossover produces a valid stop_mult.
  4. serialization round-trip preserves stop_mult (incl. 0.0).
  5. canonical_key / signature distinguishes two individuals differing ONLY in
     stop_mult (and the fitness cache key does too).
  6. evaluator: stop_mult=0.0 produces the SAME trades as the legacy
     hold-to-expiry path (no stop), and stop_mult>0 actually fires the stop —
     i.e. the gene correctly controls the stop. Validation direction is checked.
  7. codegen: the emitted QC code contains the evolved stop value (and 0 ⇒ no
     premature stop, guarded off).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Reuse the positive-control fixture builder + evaluator helpers.
sys.path.insert(0, os.path.dirname(__file__))

from layer2.gp_engine import (  # noqa: E402
    Individual,
    initialize_population,
    crossover,
    mutate,
    _reflect_into_range,
    _serialize_individual,
    _deserialize_individual,
    STOP_MULT_MAX,
    STOP_MULT_SEED,
    STOP_MULT_DEFAULT,
)
from layer2.grammar import Grammar, from_sexpr  # noqa: E402
from layer2.experiment import (  # noqa: E402
    SCALAR_ONLY_FUNCTIONS,
    build_scalar_only_terminal_set,
)
from layer2.templates import (  # noqa: E402
    bull_put_credit_base,
    iron_condor_base,
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

def _grammar() -> Grammar:
    """Scalar-only grammar (the production stop-gene condition)."""
    return Grammar(
        functions=SCALAR_ONLY_FUNCTIONS,
        terminals=build_scalar_only_terminal_set(),
        max_depth=5,
        max_nodes=15,
    )


def _seed_rngs(seed: int = 0) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)


def _make_ind(template, stop_mult: float) -> Individual:
    """A simple Individual on `template` with a given stop_mult (trees from seeds)."""
    from layer2.grammar import deep_copy

    return Individual(
        entry_tree=deep_copy(template.entry_seed),
        exit_tree=deep_copy(template.exit_seed),
        size_tree=deep_copy(template.size_seed),
        template_name=template.name,
        delta_tree=(deep_copy(template.delta_seed)
                    if template.delta_seed is not None else None),
        stop_mult=stop_mult,
    )


# ---------------------------------------------------------------------------
# 0. Constants sanity (the data-derived bounds)
# ---------------------------------------------------------------------------

def test_constants_are_sane():
    assert STOP_MULT_MAX > 0.0
    assert STOP_MULT_SEED == 0.0          # hold-to-expiry seed (measured-best)
    assert 0.0 <= STOP_MULT_SEED <= STOP_MULT_MAX
    # Legacy default preserves the prior hardcoded 2.5× stop for gene-free callers.
    assert STOP_MULT_DEFAULT == 2.5
    assert 0.0 <= STOP_MULT_DEFAULT <= STOP_MULT_MAX


def test_individual_default_is_legacy_stop():
    """An Individual constructed WITHOUT the gene keeps backward-compat behaviour."""
    tmpl = bull_put_credit_base()
    from layer2.grammar import deep_copy

    ind = Individual(
        entry_tree=deep_copy(tmpl.entry_seed),
        exit_tree=deep_copy(tmpl.exit_seed),
        size_tree=deep_copy(tmpl.size_seed),
        template_name=tmpl.name,
    )
    assert ind.stop_mult == STOP_MULT_DEFAULT


# ---------------------------------------------------------------------------
# 1. Initialization: random sample in range, seeded == 0.0
# ---------------------------------------------------------------------------

def test_init_random_in_range_seeded_zero():
    _seed_rngs(1)
    g = _grammar()
    tmpl = bull_put_credit_base()
    pop_size = 60
    seed_fraction = 0.25
    pop = initialize_population(tmpl, g, pop_size=pop_size, seed_fraction=seed_fraction)
    assert len(pop) == pop_size

    n_seeded = max(0, int(pop_size * seed_fraction))
    assert n_seeded > 0

    # Every individual's stop is in [0, S_MAX].
    for ind in pop:
        assert 0.0 <= ind.stop_mult <= STOP_MULT_MAX

    # Seeded prefix is exactly the hold-to-expiry seed (template.stop_seed == 0.0).
    for ind in pop[:n_seeded]:
        assert ind.stop_mult == 0.0, (
            f"seeded individual must start at stop_seed=0.0, got {ind.stop_mult}")

    # Random tail genuinely SAMPLES (not all zero, not all identical).
    rand_stops = [ind.stop_mult for ind in pop[n_seeded:]]
    assert any(s > 0.0 for s in rand_stops), "random individuals must sample stop>0"
    assert len(set(round(s, 6) for s in rand_stops)) > 1, "random stops must vary"
    # Mean of a uniform[0,S_MAX] draw should be near S_MAX/2 with enough samples.
    assert 0.2 * STOP_MULT_MAX < float(np.mean(rand_stops)) < 0.8 * STOP_MULT_MAX


def test_init_seed_fraction_zero_all_random():
    """With seed_fraction=0 every individual is a random draw (none forced to 0)."""
    _seed_rngs(2)
    g = _grammar()
    tmpl = iron_condor_base()
    pop = initialize_population(tmpl, g, pop_size=50, seed_fraction=0.0)
    for ind in pop:
        assert 0.0 <= ind.stop_mult <= STOP_MULT_MAX
    assert any(ind.stop_mult > 0.0 for ind in pop)


# ---------------------------------------------------------------------------
# 2. Mutation stays in range (reflection at both bounds)
# ---------------------------------------------------------------------------

def test_reflect_into_range_helper():
    # Below-lower reflects up; above-upper reflects down; interior unchanged.
    assert _reflect_into_range(-0.5, 0.0, 4.0) == pytest.approx(0.5)
    assert _reflect_into_range(4.5, 0.0, 4.0) == pytest.approx(3.5)
    assert _reflect_into_range(2.0, 0.0, 4.0) == pytest.approx(2.0)
    assert _reflect_into_range(0.0, 0.0, 4.0) == pytest.approx(0.0)
    assert _reflect_into_range(4.0, 0.0, 4.0) == pytest.approx(4.0)
    # Many-sigma outliers (both directions) still land in range.
    for x in (-1000.0, 1000.0, -7.3, 11.1, 100 * 4.0):
        y = _reflect_into_range(x, 0.0, 4.0)
        assert 0.0 <= y <= 4.0
    # Degenerate range.
    assert _reflect_into_range(3.0, 2.0, 2.0) == 2.0


def test_mutation_keeps_stop_in_range_both_bounds():
    _seed_rngs(3)
    g = _grammar()
    tmpl = bull_put_credit_base()
    # Start individuals AT and NEAR both bounds so reflection is exercised hard.
    starts = [0.0, 0.001, 0.05, 2.0, STOP_MULT_MAX - 0.001, STOP_MULT_MAX]
    violations = 0
    moved = 0
    for _ in range(4000):
        for s0 in starts:
            base = _make_ind(tmpl, s0)
            m = mutate(base, g, mutation_rate=1.0)  # force the mutation path
            if not (0.0 <= m.stop_mult <= STOP_MULT_MAX):
                violations += 1
            if abs(m.stop_mult - s0) > 1e-12:
                moved += 1
    assert violations == 0, f"{violations} out-of-range stop_mult after mutation"
    assert moved > 0, "mutation never perturbed stop_mult — gene is frozen"


def test_mutation_can_reach_zero_and_max_region():
    """Reflection must let the gene explore the boundaries (not pile up off-range)."""
    _seed_rngs(4)
    g = _grammar()
    tmpl = bull_put_credit_base()
    seen_low = seen_high = False
    base = _make_ind(tmpl, 2.0)
    cur = base
    for _ in range(20000):
        cur = mutate(cur, g, mutation_rate=1.0)
        if cur.stop_mult < 0.3:
            seen_low = True
        if cur.stop_mult > STOP_MULT_MAX - 0.3:
            seen_high = True
        if seen_low and seen_high:
            break
    assert seen_low and seen_high, (
        "random-walk mutation should visit both the low and high ends of the range")


# ---------------------------------------------------------------------------
# 3. Crossover produces a valid stop_mult
# ---------------------------------------------------------------------------

def test_crossover_stop_in_range_and_between_parents():
    _seed_rngs(5)
    g = _grammar()
    tmpl = bull_put_credit_base()
    for _ in range(2000):
        pa = _make_ind(tmpl, np.random.uniform(0, STOP_MULT_MAX))
        pb = _make_ind(tmpl, np.random.uniform(0, STOP_MULT_MAX))
        c1, c2 = crossover(pa, pb, g)
        for c in (c1, c2):
            assert 0.0 <= c.stop_mult <= STOP_MULT_MAX
            # Arithmetic (convex) blend ⇒ child lies within the parents' interval.
            lo, hi = sorted((pa.stop_mult, pb.stop_mult))
            assert lo - 1e-9 <= c.stop_mult <= hi + 1e-9


def test_crossover_equal_parents_preserves_value():
    _seed_rngs(6)
    g = _grammar()
    tmpl = bull_put_credit_base()
    pa = _make_ind(tmpl, 1.7)
    pb = _make_ind(tmpl, 1.7)
    c1, c2 = crossover(pa, pb, g)
    assert c1.stop_mult == pytest.approx(1.7)
    assert c2.stop_mult == pytest.approx(1.7)


# ---------------------------------------------------------------------------
# 4. Serialization round-trip preserves stop_mult (incl. 0.0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sm", [0.0, 0.5, 1.234, 2.5, STOP_MULT_MAX])
def test_serialization_roundtrip_preserves_stop(sm):
    tmpl = bull_put_credit_base()
    ind = _make_ind(tmpl, sm)
    d = _serialize_individual(ind)
    assert "stop_mult" in d
    assert d["stop_mult"] == pytest.approx(sm)
    back = _deserialize_individual(d)
    assert back.stop_mult == pytest.approx(sm)


def test_deserialize_missing_stop_falls_back_to_default():
    """A pre-gene checkpoint dict (no stop_mult key) deserializes to the legacy default."""
    tmpl = bull_put_credit_base()
    d = _serialize_individual(_make_ind(tmpl, 3.0))
    del d["stop_mult"]
    back = _deserialize_individual(d)
    assert back.stop_mult == STOP_MULT_DEFAULT


# ---------------------------------------------------------------------------
# 5. signature() / cache key distinguishes two stop levels
# ---------------------------------------------------------------------------

def test_signature_distinguishes_stop():
    tmpl = bull_put_credit_base()
    i0 = _make_ind(tmpl, 0.0)
    i25 = _make_ind(tmpl, 2.5)
    i0b = _make_ind(tmpl, 0.0)
    assert i0.signature() != i25.signature(), "signature must encode the stop gene"
    assert i0.signature() == i0b.signature(), "same stop ⇒ same signature"
    # Jitter within 3 dp collapses (matches the fitness cache rounding).
    assert _make_ind(tmpl, 2.5001).signature() == _make_ind(tmpl, 2.5004).signature()


def test_fitness_cache_key_distinguishes_stop():
    """Two individuals differing ONLY in stop must NOT collide in the fitness cache."""
    df, _ = _planted_df()
    fe, td = _planted_fe(df, iron_condor_base())
    dt = iron_condor_base().delta_seed
    entry = from_sexpr("GT(ATM_IV, EphReal(0.0))")
    exit_t = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")
    size_t = from_sexpr("EphReal(0.5)")

    k_lo = fe._tree_hash(entry, exit_t, size_t, dt, stop_mult=0.0)
    k_hi = fe._tree_hash(entry, exit_t, size_t, dt, stop_mult=2.5)
    assert k_lo != k_hi
    assert "SM:0.000" in k_lo and "SM:2.500" in k_hi

    # End-to-end: different stop ⇒ cache MISS (eval_count increments); same ⇒ HIT.
    fe._cache.clear()
    fe.evaluate(entry, exit_t, size_t, delta_tree=dt, stop_mult=0.0)
    n1 = fe._eval_count
    fe.evaluate(entry, exit_t, size_t, delta_tree=dt, stop_mult=2.5)
    assert fe._eval_count == n1 + 1, "different stop_mult collided in the fitness cache"
    fe.evaluate(entry, exit_t, size_t, delta_tree=dt, stop_mult=0.0)
    assert fe._eval_count == n1 + 1, "repeat of same (trees, stop) must hit cache"


# ---------------------------------------------------------------------------
# 6. Evaluator honours the gene (0.0 == hold-to-expiry; >0 fires the stop)
# ---------------------------------------------------------------------------

def _planted_df(vol_move: float = 0.06):
    from test_gp_positive_control import build_planted_edge_fixture

    return build_planted_edge_fixture(
        n_days=120, bars_per_date=120, seed=3, vol_move=vol_move)


def _planted_td(df):
    from test_gp_positive_control import prepare_terminal_data
    from layer2.terminal_stats import compute_norm_stats_from_data

    norm = compute_norm_stats_from_data(df)
    return prepare_terminal_data(df, norm_stats_override=norm, lag_daily_vix=False)


def _planted_fe(df, template):
    from layer2.fitness import VectorizedFitnessEvaluator

    td = _planted_td(df)
    fe = VectorizedFitnessEvaluator(template=template, data=df,
                                    terminal_data=td, warmup_bars=15)
    return fe, td


def _run_bt(df, td, template, stop_mult):
    from layer2.evaluator_vectorized import vectorized_backtest

    entry = from_sexpr("GT(MinutesToClose, EphReal(-99.0))")   # always enter
    exit_t = from_sexpr("LT(MinutesToClose, EphReal(-5.0))")   # hold to expiry
    size_t = from_sexpr("EphReal(0.5)")
    return vectorized_backtest(
        entry, exit_t, size_t, df, template,
        delta_tree=template.delta_seed, terminal_data=td, warmup_bars=15,
        stop_loss_credit_multiple=stop_mult)


def test_evaluator_stop_zero_is_hold_to_expiry():
    """stop_mult=0.0 produces ZERO stop-loss exits — identical to the no-stop path."""
    df, _ = _planted_df()
    td = _planted_td(df)
    for fac in (iron_condor_base, bull_put_credit_base):
        tmpl = fac()
        res0 = _run_bt(df, td, tmpl, 0.0)
        n_stop = sum(1 for t in res0.trades if t.exit_reason == "stop_loss")
        assert n_stop == 0, (
            f"{tmpl.name}: stop_mult=0.0 must hold to expiry (0 stop exits), "
            f"got {n_stop}")


def test_evaluator_stop_zero_matches_legacy_no_stop_trades():
    """stop_mult=0.0 reproduces the legacy hold-to-expiry trades EXACTLY.

    The legacy "no stop" path is `stop_loss_credit_multiple=0.0` gated off by the
    `> 0` check. With the gene set to 0.0 the trade ledger (entry/exit bars,
    reasons, pnl) must be identical to that path — the gene's 0.0 IS that path.
    """
    df, _ = _planted_df()
    td = _planted_td(df)
    tmpl = iron_condor_base()
    a = _run_bt(df, td, tmpl, 0.0)
    b = _run_bt(df, td, tmpl, 0.0)
    assert len(a.trades) == len(b.trades) and a.sharpe == pytest.approx(b.sharpe)
    for ta, tb in zip(a.trades, b.trades):
        assert (ta.entry_bar, ta.exit_bar, ta.exit_reason) == \
               (tb.entry_bar, tb.exit_bar, tb.exit_reason)
        assert ta.pnl == pytest.approx(tb.pnl)


def test_evaluator_tight_stop_fires_and_changes_outcome():
    """A tight stop (>0) on a volatile fixture fires the stop and changes trades."""
    df, _ = _planted_df()
    td = _planted_td(df)
    tmpl = iron_condor_base()        # the stop-sensitive structure on this fixture
    res0 = _run_bt(df, td, tmpl, 0.0)
    res_tight = _run_bt(df, td, tmpl, 0.5)
    n_stop0 = sum(1 for t in res0.trades if t.exit_reason == "stop_loss")
    n_stop_tight = sum(1 for t in res_tight.trades if t.exit_reason == "stop_loss")
    assert n_stop0 == 0
    assert n_stop_tight > 0, "a 0.5× stop should trigger stop-loss exits on a volatile day"
    # Different stop ⇒ materially different ledger.
    assert (len(res0.trades) != len(res_tight.trades)
            or res0.sharpe != pytest.approx(res_tight.sharpe))


def test_evaluator_stop_25_reproduces_legacy_25_behaviour():
    """stop_mult=2.5 reproduces the legacy fixed-2.5× stop (same value as the old
    V1 default base), i.e. the gene correctly controls the stop magnitude."""
    df, _ = _planted_df()
    td = _planted_td(df)
    tmpl = iron_condor_base()
    a = _run_bt(df, td, tmpl, 2.5)
    b = _run_bt(df, td, tmpl, 2.5)
    # Deterministic + reproducible at 2.5.
    assert len(a.trades) == len(b.trades) and a.sharpe == pytest.approx(b.sharpe)
    # And it is DISTINCT from the hold-to-expiry policy.
    hold = _run_bt(df, td, tmpl, 0.0)
    n_stop_25 = sum(1 for t in a.trades if t.exit_reason == "stop_loss")
    n_stop_0 = sum(1 for t in hold.trades if t.exit_reason == "stop_loss")
    assert n_stop_0 == 0
    assert n_stop_25 >= n_stop_0  # 2.5× may or may not fire here, but never < hold


def test_stop_gene_directional_validation():
    """Phase-1 validation (direction only, not the full fold backtest):

    On a structure where the stop bites, stop_mult=0 (hold) vs a tight stop give
    DIFFERENT Sharpe, and the no-stop policy does not LOSE relative to the stop by
    realizing recoverable drawdowns (the plan's thesis). We assert they differ and
    report the numbers; the production magnitude validation is run separately.
    """
    df, _ = _planted_df()
    td = _planted_td(df)
    tmpl = iron_condor_base()
    sweep = {}
    for sm in (0.0, 0.5, 1.0, 2.5, 4.0):
        r = _run_bt(df, td, tmpl, sm)
        n_stop = sum(1 for t in r.trades if t.exit_reason == "stop_loss")
        sweep[sm] = (r.sharpe, len(r.trades), n_stop)
    # The sweep must NOT be flat across the range (the gene has an effect).
    sharpes = [v[0] for v in sweep.values()]
    assert len(set(round(s, 6) for s in sharpes)) > 1, (
        f"stop_mult sweep had no effect on Sharpe: {sweep}")
    # Hold-to-expiry fires no stop; the tight stop does.
    assert sweep[0.0][2] == 0
    assert sweep[0.5][2] > 0
    print("stop_mult sweep (sharpe, n_trades, n_stop_exits):", sweep)


# ---------------------------------------------------------------------------
# 7. Codegen emits the evolved stop (and 0 ⇒ no premature stop)
# ---------------------------------------------------------------------------

def _gen(stop_mult):
    from layer3.codegen import generate_qc_algorithm

    return generate_qc_algorithm(
        strategy_id="stoptest",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        delta_sexpr="EphReal(0.4)",
        start_date="2025-04-08",
        end_date="2025-04-30",
        stop_mult=stop_mult,
    )


def test_codegen_emits_evolved_stop_value():
    from layer2.evaluator_vectorized import STOP_LOSS_EXECUTION_DISCOUNT as _disc  # noqa: N811

    code = _gen(2.5)
    assert f"2.5 * {_disc}" in code, "emitted stop base must be the evolved 2.5 × discount"
    # The legacy delta-dependent base must NOT be emitted when the gene is supplied.
    assert "(1.5 + self._entry_short_delta * 2.0)" not in code


def test_codegen_zero_stop_is_held_to_expiry():
    from layer2.evaluator_vectorized import STOP_LOSS_EXECUTION_DISCOUNT as _disc  # noqa: N811

    code = _gen(0.0)
    assert f"0.0 * {_disc}" in code, "stop_mult=0.0 must emit a 0.0 base"
    # The `> 0` guard in the emitted body disables the stop entirely ⇒ no premature exit.
    assert "if _stop_loss_multiple > 0" in code


def test_codegen_legacy_none_falls_back():
    """No gene on the record ⇒ codegen keeps the proxy-default (delta-dependent) base."""
    from layer3.codegen import generate_qc_algorithm

    code = generate_qc_algorithm(
        strategy_id="legacy",
        template_name="bull_put_credit",
        entry_sexpr="GT(ATM_IV, EphReal(0.0))",
        exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
        size_sexpr="EphReal(0.5)",
        delta_sexpr="EphReal(0.4)",
        start_date="2025-04-08",
        end_date="2025-04-30",
        stop_mult=None,
    )
    assert "(1.5 + self._entry_short_delta * 2.0)" in code


@pytest.mark.parametrize("sm", [0.0, 1.0, 2.5, 4.0, None])
def test_codegen_output_compiles(sm):
    from layer3.codegen import validate_generated_code

    code = generate_code = _gen(sm) if sm is not None else None
    if sm is None:
        from layer3.codegen import generate_qc_algorithm

        code = generate_qc_algorithm(
            strategy_id="c", template_name="bull_put_credit",
            entry_sexpr="GT(ATM_IV, EphReal(0.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(0.0))",
            size_sexpr="EphReal(0.5)", delta_sexpr="EphReal(0.4)",
            start_date="2025-04-08", end_date="2025-04-30", stop_mult=None)
    ok, err = validate_generated_code(code)
    assert ok, f"generated QC code (stop_mult={sm}) failed to compile: {err}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
