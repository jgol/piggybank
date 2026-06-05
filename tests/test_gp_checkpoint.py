"""Generation-level checkpoint / resume tests for layer2.gp_engine.evolve().

Covers the SIGHUP/SIGTERM-safe checkpoint added so a multi-hour template-fold
evolution survives a closed terminal / dropped SSH:

  (i)   a resumed run CONTINUES from the saved generation (does not restart);
  (ii)  determinism ACROSS the checkpoint boundary — a run interrupted at gen G
        and resumed produces the SAME final population (same trees + same
        fitness) as an uninterrupted run with the same seed;
  (iii) a fingerprint-MISMATCHED checkpoint (config/grammar drift) is rejected
        and the run starts fresh instead of restoring stale state;
  (iv)  the atomic-write + load round-trip preserves population, RNG streams,
        ramped evaluator knobs, and the inner-val tracking dict;
  (v)    the checkpoint is DELETED on clean completion.

These exercise the real evolve() loop (small pop, few generations) — not a mock.
"""
import os
import random

import numpy as np
import pandas as pd
import pytest

from layer2.fitness import FitnessEvaluator
from layer2.gp_engine import (
    EvolutionConfig, Individual, evolve, pareto_front,
    _CHECKPOINT_FORMAT_VERSION,
    _CheckpointAndExit, _checkpoint_path, _load_checkpoint,
    _run_fingerprint, _serialize_individual, _deserialize_individual,
    _write_checkpoint_atomic,
)
from layer2.grammar import Grammar, GType, to_str
from layer2.io import (
    PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor


def _make_data(n_dates=1, bars_per_date=15) -> pd.DataFrame:
    """Tiny deterministic dataset (mirrors test_gp_engine._make_data)."""
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


def _fresh_evaluator():
    return FitnessEvaluator(
        template=iron_condor(), data=_make_data(),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )


def _pop_signatures(pop):
    """Order-independent fingerprint of a population: (signature, fitness)."""
    out = []
    for ind in pop:
        out.append((
            ind.signature(),
            None if ind.fitness is None else tuple(round(float(x), 9)
                                                   for x in ind.fitness),
            ind.age,
        ))
    return sorted(out)


# ---------------------------------------------------------------------------
# (iv) atomic write + load round-trip and serialization fidelity
# ---------------------------------------------------------------------------

def test_individual_serialization_round_trip():
    """to_str/from_sexpr round-trip restores trees, fitness, age, NSGA state."""
    template = iron_condor()
    grammar = Grammar(max_depth=4, max_nodes=20)
    random.seed(1)
    ind = Individual(
        entry_tree=grammar.grow(GType.BOOL, max_d=4),
        exit_tree=grammar.grow(GType.BOOL, max_d=4),
        size_tree=grammar.grow(GType.REAL, max_d=3),
        template_name=template.name,
        delta_tree=grammar.grow(GType.REAL, max_d=3),
    )
    ind.fitness = np.array([-1.23, 0.5, -0.1])
    ind.age = 7
    ind._nds_rank = 2
    ind._crowding_dist = 1.5

    restored = _deserialize_individual(_serialize_individual(ind))
    assert to_str(restored.entry_tree) == to_str(ind.entry_tree)
    assert to_str(restored.exit_tree) == to_str(ind.exit_tree)
    assert to_str(restored.size_tree) == to_str(ind.size_tree)
    assert to_str(restored.delta_tree) == to_str(ind.delta_tree)
    assert restored.template_name == ind.template_name
    np.testing.assert_array_equal(restored.fitness, ind.fitness)
    assert restored.age == 7
    assert restored._nds_rank == 2
    assert restored._crowding_dist == 1.5


def test_checkpoint_atomic_write_and_load(tmp_path):
    """_write_checkpoint_atomic writes via temp+replace; _load_checkpoint
    restores only when the fingerprint matches."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)
    fe = _fresh_evaluator()
    fp = _run_fingerprint(template, grammar, config, fe)

    state = {
        # Use the LIVE format version so this test exercises the fingerprint /
        # round-trip path, not the format-version gate (which has its own test).
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fp, "template_name": template.name,
        "gen": 2, "population": [], "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "niching_rng_state": np.random.default_rng(0).bit_generator.state,
        "min_trades": 5, "random_entry_margin": 0.1,
        "metrics_log": [], "val_tracking": {},
    }
    _write_checkpoint_atomic(tmp_path, template.name, state)
    assert _checkpoint_path(tmp_path, template.name).exists()
    # No leftover temp files.
    assert not list(tmp_path.glob("*.tmp.*"))

    loaded = _load_checkpoint(tmp_path, template.name, fp)
    assert loaded is not None
    assert loaded["gen"] == 2
    assert loaded["min_trades"] == 5


# ---------------------------------------------------------------------------
# (iii) fingerprint mismatch → rejected
# ---------------------------------------------------------------------------

def test_fingerprint_mismatch_rejected(tmp_path):
    """A checkpoint written under one config must NOT restore under a different
    config — _load_checkpoint returns None (and warns) on fingerprint drift."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    config_a = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)
    fe = _fresh_evaluator()
    fp_a = _run_fingerprint(template, grammar, config_a, fe)

    state = {
        "format_version": _CHECKPOINT_FORMAT_VERSION,
        "fingerprint": fp_a, "template_name": template.name,
        "gen": 1, "population": [], "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "niching_rng_state": np.random.default_rng(0).bit_generator.state,
        "min_trades": 1, "random_entry_margin": 0.0,
        "metrics_log": [], "val_tracking": {},
    }
    _write_checkpoint_atomic(tmp_path, template.name, state)

    # Different seed → different fingerprint → reject.
    config_b = EvolutionConfig(pop_size=8, n_generations=3, seed=99, n_partitions=4)
    fp_b = _run_fingerprint(template, grammar, config_b, fe)
    assert fp_a != fp_b
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        assert _load_checkpoint(tmp_path, template.name, fp_b) is None

    # The matching fingerprint still loads.
    assert _load_checkpoint(tmp_path, template.name, fp_a) is not None


def test_fingerprint_changes_with_grammar(tmp_path):
    """Mutating the grammar (function/terminal set) must change the
    fingerprint so a stale checkpoint is rejected."""
    template = iron_condor()
    config = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)
    fe = _fresh_evaluator()
    g_full = Grammar(max_depth=3, max_nodes=15)
    g_small = Grammar(
        functions=g_full.functions[:5], terminals=g_full.terminals,
        max_depth=3, max_nodes=15,
    )
    assert (_run_fingerprint(template, g_full, config, fe)
            != _run_fingerprint(template, g_small, config, fe))


def test_format_version_mismatch_rejected(tmp_path):
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)
    fe = _fresh_evaluator()
    fp = _run_fingerprint(template, grammar, config, fe)
    state = {
        "format_version": 999, "fingerprint": fp, "template_name": template.name,
        "gen": 1, "population": [], "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "niching_rng_state": np.random.default_rng(0).bit_generator.state,
        "min_trades": 1, "random_entry_margin": 0.0,
        "metrics_log": [], "val_tracking": {},
    }
    _write_checkpoint_atomic(tmp_path, template.name, state)
    with pytest.warns(UserWarning, match="format_version"):
        assert _load_checkpoint(tmp_path, template.name, fp) is None


# ---------------------------------------------------------------------------
# (i) + (ii) determinism ACROSS the checkpoint boundary
# ---------------------------------------------------------------------------

class _AbortAfterGen(Exception):
    """Test-only: raised from on_generation to simulate a SIGHUP mid-run AFTER
    the clean-boundary checkpoint for the next generation is on disk."""


def _run_uninterrupted(tmp_path, seed, n_gen):
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=n_gen, seed=seed,
                             n_partitions=4)
    fe = _fresh_evaluator()
    out_dir = tmp_path / "gold"
    final_pop, metrics, _vt = evolve(
        iron_condor(), fe, grammar, config,
        checkpoint_dir=str(out_dir), checkpoint_every=2,
    )
    return final_pop, metrics


def test_resume_continues_from_saved_generation(tmp_path):
    """(i) After an interrupt at gen 2, the checkpoint says resume@2 and the
    resumed run only runs the REMAINING generations (2,3,4 for n_gen=5)."""
    seed, n_gen = 7, 5
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=n_gen, seed=seed,
                             n_partitions=4)
    out_dir = tmp_path / "run"

    # Interrupt at the START of generation index 2. By then the end-of-gen-1
    # clean-boundary checkpoint (resume@2) is already on disk.
    def _cb(m, _pop):
        if m.generation == 2:
            raise _AbortAfterGen()

    fe = _fresh_evaluator()
    with pytest.raises(_AbortAfterGen):
        evolve(iron_condor(), fe, grammar, config,
               on_generation=_cb,
               checkpoint_dir=str(out_dir), checkpoint_every=2)

    ckpt = _checkpoint_path(out_dir, "iron_condor_standard")
    assert ckpt.exists(), "interrupt must leave a resumable checkpoint"
    import pickle
    with open(ckpt, "rb") as fh:
        saved = pickle.load(fh)
    assert saved["gen"] == 2, f"expected resume@2, got {saved['gen']}"

    # Resume: a fresh evaluator + same config; evolve must restore and only run
    # generations 2,3,4. The returned metrics_log covers ALL n_gen generations
    # (restored history gens 0-1 + fresh gens 2-4).
    seen = []
    def _cb2(m, _pop):
        seen.append(m.generation)
    fe2 = _fresh_evaluator()
    final_pop, metrics, _vt = evolve(
        iron_condor(), fe2, grammar, config,
        on_generation=_cb2,
        checkpoint_dir=str(out_dir), checkpoint_every=2,
    )
    # Only gens 2,3,4 were freshly executed (on_generation fires per fresh gen).
    assert seen == [2, 3, 4], f"resume executed gens {seen}, expected [2,3,4]"
    assert len(metrics) == n_gen
    # Clean completion deletes the checkpoint.
    assert not ckpt.exists(), "checkpoint must be deleted on clean completion"


def test_determinism_across_checkpoint_boundary(tmp_path):
    """(ii) THE core guarantee: interrupt-and-resume yields the SAME final
    population (trees + fitness + age) as an uninterrupted run with the same
    seed. The clean-boundary checkpoint snapshots all three RNG streams at the
    generation boundary, so resume is bit-identical."""
    seed, n_gen = 11, 5

    # --- Gold: fully uninterrupted run (its own checkpoint dir, deleted on
    # clean completion). ---
    gold_pop, gold_metrics = _run_uninterrupted(tmp_path, seed, n_gen)
    gold_sig = _pop_signatures(gold_pop)
    gold_front = sorted(ind.signature() for ind in pareto_front(gold_pop))

    # --- Interrupted + resumed run. ---
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=n_gen, seed=seed,
                             n_partitions=4)
    out_dir = tmp_path / "interrupted"

    def _cb(m, _pop):
        if m.generation == 2:
            raise _AbortAfterGen()

    fe = _fresh_evaluator()
    with pytest.raises(_AbortAfterGen):
        evolve(iron_condor(), fe, grammar, config, on_generation=_cb,
               checkpoint_dir=str(out_dir), checkpoint_every=2)
    assert _checkpoint_path(out_dir, "iron_condor_standard").exists()

    fe2 = _fresh_evaluator()
    resumed_pop, resumed_metrics = (lambda r: (r[0], r[1]))(
        evolve(iron_condor(), fe2, grammar, config,
               checkpoint_dir=str(out_dir), checkpoint_every=2)
    )
    resumed_sig = _pop_signatures(resumed_pop)
    resumed_front = sorted(ind.signature() for ind in pareto_front(resumed_pop))

    # Final populations must be identical across the checkpoint boundary.
    assert resumed_sig == gold_sig, (
        "resumed final population differs from uninterrupted run — "
        "RNG / population state did not restore deterministically"
    )
    assert resumed_front == gold_front
    # Per-generation best-objective trajectory must match too (gens 2..4).
    for g in range(2, n_gen):
        np.testing.assert_allclose(
            resumed_metrics[g].best_per_objective,
            gold_metrics[g].best_per_objective,
            rtol=0, atol=1e-9,
            err_msg=f"gen {g} best_per_objective diverged after resume",
        )


def test_signal_handler_flush_then_resume(tmp_path):
    """SIGTERM path: the handler flushes a checkpoint synchronously. We exercise
    it by raising _CheckpointAndExit through the same mechanism the real handler
    uses (evolve catches it and sys.exit(0)). After 'kill', a checkpoint exists
    and a resume restores it. (On-signal resume re-does the in-flight gen, so we
    assert continuation + restore, not bit-identity.)"""
    seed, n_gen = 5, 6
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=n_gen, seed=seed,
                             n_partitions=4)
    out_dir = tmp_path / "sig"

    # Drive a real SIGTERM into our own process from the on_generation callback
    # once gen 3 starts. The installed handler flushes the checkpoint and raises
    # _CheckpointAndExit, which evolve() turns into a clean SystemExit(0).
    def _cb(m, _pop):
        if m.generation == 3:
            os.kill(os.getpid(), __import__("signal").SIGTERM)

    fe = _fresh_evaluator()
    with pytest.raises(SystemExit) as ei:
        evolve(iron_condor(), fe, grammar, config, on_generation=_cb,
               checkpoint_dir=str(out_dir), checkpoint_every=100)  # K large: only signal writes
    assert ei.value.code == 0
    ckpt = _checkpoint_path(out_dir, "iron_condor_standard")
    assert ckpt.exists(), "SIGTERM must leave a checkpoint"
    import pickle
    with open(ckpt, "rb") as fh:
        saved = pickle.load(fh)
    # Handler resumes AT the in-flight generation (3).
    assert saved["gen"] == 3

    # Resume runs to completion and deletes the checkpoint.
    fe2 = _fresh_evaluator()
    final_pop, metrics, _vt = evolve(
        iron_condor(), fe2, grammar, config,
        checkpoint_dir=str(out_dir), checkpoint_every=100,
    )
    assert len(metrics) == n_gen
    assert all(ind.fitness is not None for ind in final_pop)
    assert not ckpt.exists()


def test_env_var_enables_checkpointing(tmp_path, monkeypatch):
    """The launch script enables checkpointing WITHOUT editing experiment.py by
    exporting GP_CHECKPOINT_DIR (+ GP_CHECKPOINT_EVERY). evolve() must honor
    both even when the caller passes neither argument — this is the only path
    the real walk-forward run uses."""
    monkeypatch.setenv("GP_CHECKPOINT_DIR", str(tmp_path / "envdir"))
    monkeypatch.setenv("GP_CHECKPOINT_EVERY", "1")  # checkpoint every gen
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=4, seed=3, n_partitions=4)

    seen = []
    def _cb(m, _pop):
        # By the time gen 2 starts, the resume@2 checkpoint exists (every-gen).
        ck = _checkpoint_path(tmp_path / "envdir", "iron_condor_standard")
        if m.generation == 1:
            seen.append(ck.exists())  # written at end of gen 0 (K=1)

    fe = _fresh_evaluator()
    # No checkpoint_dir / checkpoint_every args — purely env-driven.
    final_pop, metrics, _vt = evolve(iron_condor(), fe, grammar, config,
                                     on_generation=_cb)
    assert seen and seen[0] is True, "env GP_CHECKPOINT_DIR did not enable checkpointing"
    assert len(metrics) == 4
    # Clean completion still deletes the checkpoint.
    assert not _checkpoint_path(tmp_path / "envdir", "iron_condor_standard").exists()


def test_data_fingerprint_makes_checkpoint_fold_specific():
    """Walk-forward evolves the SAME template+config across folds with different
    train windows into the SAME checkpoint filename. The evaluator data
    fingerprint must enter the run fingerprint so a fold-2 checkpoint is
    rejected when fold-1 restarts."""
    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=3, seed=42, n_partitions=4)

    fe_fold1 = FitnessEvaluator(
        template=template, data=_make_data(n_dates=1, bars_per_date=15),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    fe_fold2 = FitnessEvaluator(
        template=template, data=_make_data(n_dates=2, bars_per_date=15),  # diff window
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    fp1 = _run_fingerprint(template, grammar, config, fe_fold1)
    fp2 = _run_fingerprint(template, grammar, config, fe_fold2)
    assert fp1 != fp2, (
        "different train windows produced the same fingerprint — a stale-fold "
        "checkpoint could be wrongly resumed"
    )


# ---------------------------------------------------------------------------
# (H2) gen-0 checkpoint hole — a checkpoint must exist BEFORE gen-0 evaluation
# completes, so a SIGTERM/SIGHUP landing mid-gen-0 (which can take minutes for a
# large pop) leaves a resumable population instead of losing the whole unit.
# ---------------------------------------------------------------------------

class _AbortDuringGen0Eval(Exception):
    """Test-only: raised from the evaluator on the FIRST evaluate() call, i.e.
    while gen 0 is still being scored — BEFORE gen 0 (and thus the first
    end-of-generation checkpoint) completes."""


class _AbortOnFirstEvalEvaluator(FitnessEvaluator):
    """Real evaluator that raises _AbortDuringGen0Eval the first time evaluate()
    is called. Simulates an interrupt arriving in the middle of the gen-0
    full-population scoring loop, before any generation has finished."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_calls = 0

    def evaluate(self, *args, **kwargs):
        self._eval_calls += 1
        if self._eval_calls == 1:
            raise _AbortDuringGen0Eval()
        return super().evaluate(*args, **kwargs)


def test_gen0_checkpoint_written_before_eval_completes(tmp_path):
    """(H2) An interrupt DURING gen-0 evaluation must still leave a resumable
    checkpoint. evolve() writes an initial resume@0 checkpoint right after
    initialize_population (before the gen-0 evaluate loop), so even though the
    evaluator blows up on its first call — before gen 0 finishes — the
    checkpoint exists, loads under the matching fingerprint, says gen 0, and
    carries a full not-yet-evaluated population.
    """
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=5, seed=7, n_partitions=4)
    out_dir = tmp_path / "gen0"

    fe = _AbortOnFirstEvalEvaluator(
        template=iron_condor(), data=_make_data(),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    # The interrupt fires in the FIRST evaluate() call — gen 0 never completes,
    # so the every-K (and end-of-gen) checkpoint never runs. Only the new
    # initial gen-0 write can have produced a checkpoint.
    with pytest.raises(_AbortDuringGen0Eval):
        evolve(iron_condor(), fe, grammar, config,
               checkpoint_dir=str(out_dir), checkpoint_every=2)
    assert fe._eval_calls == 1, "evaluator did not abort on the first eval call"

    ckpt = _checkpoint_path(out_dir, "iron_condor_standard")
    assert ckpt.exists(), (
        "no checkpoint after a gen-0 interrupt — the gen-0 hole is not closed"
    )

    # It must be resume-VALID: matching fingerprint, resume@0, full population,
    # and (since gen 0 never finished) NOT-yet-evaluated individuals.
    fp = _run_fingerprint(iron_condor(), grammar, config, fe)
    loaded = _load_checkpoint(out_dir, "iron_condor_standard", fp)
    assert loaded is not None, "initial gen-0 checkpoint failed fingerprint check"
    assert loaded["gen"] == 0, f"expected resume@0, got {loaded['gen']}"
    assert len(loaded["population"]) == config.pop_size, (
        "initial checkpoint did not snapshot the full initialized population"
    )
    # Population was captured pre-evaluation → fitness is None for every member.
    assert all(d["fitness"] is None for d in loaded["population"]), (
        "gen-0 checkpoint should hold the un-evaluated initial population"
    )


def test_gen0_checkpoint_resumes_and_completes(tmp_path):
    """(H2) The gen-0 checkpoint is not just present — it RESUMES cleanly. A
    fresh evaluator restores resume@0 and runs all generations to completion,
    deleting the checkpoint at the end."""
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=4, seed=7, n_partitions=4)
    out_dir = tmp_path / "gen0resume"

    fe = _AbortOnFirstEvalEvaluator(
        template=iron_condor(), data=_make_data(),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
    )
    with pytest.raises(_AbortDuringGen0Eval):
        evolve(iron_condor(), fe, grammar, config,
               checkpoint_dir=str(out_dir), checkpoint_every=2)
    ckpt = _checkpoint_path(out_dir, "iron_condor_standard")
    assert ckpt.exists()

    # Resume with a healthy evaluator: restores resume@0, runs gens 0..3.
    seen = []
    def _cb(m, _pop):
        seen.append(m.generation)
    fe2 = _fresh_evaluator()
    final_pop, metrics, _vt = evolve(
        iron_condor(), fe2, grammar, config,
        on_generation=_cb,
        checkpoint_dir=str(out_dir), checkpoint_every=2,
    )
    assert seen == [0, 1, 2, 3], (
        f"resume@0 did not re-run generation 0 from the top: executed {seen}"
    )
    assert len(metrics) == config.n_generations
    assert all(ind.fitness is not None for ind in final_pop)
    assert not ckpt.exists(), "checkpoint must be deleted on clean completion"


def test_gen0_checkpoint_not_rewritten_on_resume(tmp_path):
    """(H2) The initial gen-0 write must fire ONLY on a fresh start, never when
    resuming an existing checkpoint — otherwise a resume would clobber the
    restored (possibly mid-run) checkpoint back to gen 0. We seed a resume@2
    checkpoint, then confirm a resume keeps gen >= 2 throughout (the initial
    write is suppressed)."""
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=5, seed=7, n_partitions=4)
    out_dir = tmp_path / "noclobber"

    # Produce a genuine resume@2 checkpoint by interrupting an uninterrupted run
    # at gen 2 (end-of-gen-1 every-K=2 boundary is on disk).
    def _abort_at_2(m, _pop):
        if m.generation == 2:
            raise _AbortAfterGen()
    fe = _fresh_evaluator()
    with pytest.raises(_AbortAfterGen):
        evolve(iron_condor(), fe, grammar, config, on_generation=_abort_at_2,
               checkpoint_dir=str(out_dir), checkpoint_every=2)
    ckpt = _checkpoint_path(out_dir, "iron_condor_standard")
    import pickle
    with open(ckpt, "rb") as fh:
        assert pickle.load(fh)["gen"] == 2

    # Now resume: the run must NOT reset the checkpoint to gen 0. Capture the gen
    # recorded on disk at the first fresh generation boundary.
    seen = []
    def _cb(m, _pop):
        seen.append(m.generation)
    fe2 = _fresh_evaluator()
    evolve(iron_condor(), fe2, grammar, config, on_generation=_cb,
           checkpoint_dir=str(out_dir), checkpoint_every=2)
    # Resume executed only gens 2,3,4 — gen 0/1 were NOT re-run, proving the
    # initial gen-0 write was suppressed on the resume path.
    assert seen == [2, 3, 4], (
        f"resume re-ran early generations {seen} — the initial gen-0 write "
        f"clobbered the resumed checkpoint"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
