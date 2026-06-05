"""Silent-collapse guard (gp_engine.evolve) — fail-fast when the population is
100% infeasible.

Background (root-caused 2026-06-02): an all-sentinel population gives NSGA-III
nothing to select on, so a run silently returns an arbitrary churn front instead
of erroring — the exact failure that masqueraded as a "GP can't discover signal"
bug. The trigger reproduced here is warmup_bars >= bars/day: the PER-DAY
fire-rate / conditional-Sharpe gates then see 0 eligible bars on every day and
sentinel every tree (MEASURED: 48 bars/day + warmup 60 -> 0/256 feasible).

These tests pin the guard's contract:
  1. RAISES (within _FEAS_GUARD_CONSECUTIVE_GENS) on a genuine total collapse.
  2. does NOT fire when a feasible population exists (no false positive).
  3. respects the GP_DISABLE_FEASIBILITY_GUARD escape hatch.
  4. is exempt for tiny (<_FEAS_GUARD_MIN_POP) test populations.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))  # allow sibling-test import
from test_gp_positive_control import build_planted_edge_fixture  # noqa: E402

from layer2.io import split_by_date  # noqa: E402
from layer2.evaluator_vectorized import prepare_terminal_data  # noqa: E402
from layer2.terminal_stats import compute_norm_stats_from_arrays  # noqa: E402
from layer2.grammar import (  # noqa: E402
    Grammar, SCALAR_ONLY_FUNCTIONS, build_scalar_only_terminal_set,
)
from layer2.gp_engine import (  # noqa: E402
    evolve, EvolutionConfig, _FEAS_GUARD_CONSECUTIVE_GENS, _FEAS_GUARD_MIN_POP,
)
from layer2.fitness import VectorizedFitnessEvaluator  # noqa: E402
from layer2.templates import bull_put_credit_standard  # noqa: E402


def _build_fe(bars_per_date: int, warmup_bars: int, n_days: int = 40):
    """Build (template, grammar, fe) over a planted-edge fixture at a given
    bar-density / warmup. warmup_bars >= bars_per_date drives the total collapse;
    warmup_bars < bars_per_date leaves a feasible population."""
    df, _ = build_planted_edge_fixture(n_days=n_days, bars_per_date=bars_per_date, seed=7)
    splits = split_by_date(df, train_end="2024-01-29", val_end="2024-02-10",
                           embargo_days=0)
    train = splits["train"]
    fold_norm = compute_norm_stats_from_arrays(
        prepare_terminal_data(train, normalize_terminals=False))
    train_td = prepare_terminal_data(train, norm_stats_override=fold_norm)
    tmpl = bull_put_credit_standard()
    grammar = Grammar(functions=SCALAR_ONLY_FUNCTIONS,
                      terminals=build_scalar_only_terminal_set(),
                      max_depth=5, max_nodes=15)
    fe = VectorizedFitnessEvaluator(
        template=tmpl, data=train, terminal_data=train_td,
        min_trades=8, warmup_bars=warmup_bars, regime_gate_enabled=True)
    return tmpl, grammar, fe


def _evo_config(pop_size: int, n_generations: int) -> EvolutionConfig:
    return EvolutionConfig(pop_size=pop_size, n_generations=n_generations,
                           seed=123, seed_fraction=0.0, n_partitions=4)


def test_guard_raises_on_total_collapse():
    """48 bars/day + warmup=60 -> 0 per-day-eligible bars -> every tree sentineled.
    The guard must RAISE (not silently return a churn front) within the consecutive
    threshold, with an actionable warmup-vs-bars/day diagnostic. Also asserts the
    in-flight lock is released on the raise path (a leaked lock would deadlock the
    next evolve() call in-process)."""
    from layer2.gp_engine import _EVOLVE_INFLIGHT_LOCK
    tmpl, grammar, fe = _build_fe(bars_per_date=48, warmup_bars=60)
    cfg = _evo_config(pop_size=64, n_generations=_FEAS_GUARD_CONSECUTIVE_GENS + 3)
    with pytest.raises(RuntimeError, match="feasibility guard"):
        evolve(tmpl, fe, grammar, cfg)
    assert not _EVOLVE_INFLIGHT_LOCK.locked(), (
        "evolve() must release _EVOLVE_INFLIGHT_LOCK on the guard-raise path")


def test_guard_message_names_warmup_cause():
    """The diagnostic must point at the warmup_bars >= bars/day root cause so an
    operator can act without re-deriving it."""
    tmpl, grammar, fe = _build_fe(bars_per_date=48, warmup_bars=60)
    cfg = _evo_config(pop_size=64, n_generations=_FEAS_GUARD_CONSECUTIVE_GENS + 2)
    with pytest.raises(RuntimeError) as ei:
        evolve(tmpl, fe, grammar, cfg)
    msg = str(ei.value)
    assert "warmup_bars" in msg and "bars-per-day" in msg
    assert "0/64" in msg  # all-infeasible count surfaced


def test_guard_silent_when_feasible():
    """120 bars/day + warmup=15 -> feasible population exists -> evolve completes
    normally and the guard never fires."""
    tmpl, grammar, fe = _build_fe(bars_per_date=120, warmup_bars=15)
    cfg = _evo_config(pop_size=64, n_generations=_FEAS_GUARD_CONSECUTIVE_GENS + 1)
    final_pop, metrics, _ = evolve(tmpl, fe, grammar, cfg)
    assert len(final_pop) == 64
    # at least one feasible individual is the whole point of "not collapsed"
    from layer2.fitness import FAILED_FITNESS_SENTINEL
    n_feas = sum(1 for i in final_pop
                 if i.fitness is not None and float(i.fitness[0]) < FAILED_FITNESS_SENTINEL * 0.5)
    assert n_feas > 0, "feasible fixture must leave feasible individuals"


def test_guard_respects_env_override(monkeypatch):
    """GP_DISABLE_FEASIBILITY_GUARD=1 disables the abort (for deliberate
    infeasible-path experiments). evolve completes instead of raising."""
    monkeypatch.setenv("GP_DISABLE_FEASIBILITY_GUARD", "1")
    tmpl, grammar, fe = _build_fe(bars_per_date=48, warmup_bars=60)
    cfg = _evo_config(pop_size=64, n_generations=_FEAS_GUARD_CONSECUTIVE_GENS + 1)
    final_pop, _, _ = evolve(tmpl, fe, grammar, cfg)  # must NOT raise
    assert len(final_pop) == 64


def test_guard_exempts_small_pop():
    """Populations below _FEAS_GUARD_MIN_POP can legitimately sit infeasible for a
    few generations in tiny unit tests; the guard must not fire on them."""
    assert _FEAS_GUARD_MIN_POP > 16
    tmpl, grammar, fe = _build_fe(bars_per_date=48, warmup_bars=60)
    cfg = _evo_config(pop_size=16, n_generations=_FEAS_GUARD_CONSECUTIVE_GENS + 1)
    final_pop, _, _ = evolve(tmpl, fe, grammar, cfg)  # must NOT raise
    assert len(final_pop) == 16
