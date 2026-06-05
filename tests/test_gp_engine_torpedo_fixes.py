"""B1 torpedo-fix tests for layer2.gp_engine.evolve() inner-validation early-stop.

The inner-validation overfit detector (evolve(), the
``gen % val_checkpoint_interval == 0`` block) was truncating EVERY walk-forward
run to ~gen 31 of 100. Two coupled bugs:

  (a) apples-to-oranges comparator — it compared the single BEST (max) train
      Sharpe over the WHOLE population against the MEAN val Sharpe over the
      Pareto FRONT, so the gap was inflated by construction;
  (b) it fired on LOSING folds where there is no in-sample edge to overfit
      (best train Sharpe <= 0, val even worse → large positive gap read as
      "overfitting").

The fix:
  (a) compute the train side the SAME way the val side is computed — mean of
      (-fitness[0]) over the SAME Pareto front, same sentinel filter; and
  (b) require a POSITIVE in-sample edge (train front Sharpe > 0) as a
      precondition before the gap may increment the overfit counter.

These tests drive the real evolve() loop (small pop, few generations) with a
controlled ``val_evaluator`` so the ONLY variable governing whether early-stop
fires is the sign of the in-sample edge — exactly the precondition under test.

  B1-losing : negative in-sample edge + forced large gap → must NOT early-stop
              (run completes all generations).
  B1-overfit: positive in-sample edge + forced large gap → MUST early-stop
              (genuine overfit is still caught).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from layer2.fitness import FitnessEvaluator
from layer2.gp_engine import EvolutionConfig, evolve
from layer2.grammar import Grammar
from layer2.io import (
    PRICE_COLUMN, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)
from layer2.templates import iron_condor


def _make_data(n_dates=1, bars_per_date=15) -> pd.DataFrame:
    """Tiny deterministic dataset (mirrors test_gp_checkpoint._make_data)."""
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


class _FixedSharpeEvaluator(FitnessEvaluator):
    """Faithful real evaluator whose only override is a DETERMINISTIC positive
    in-sample Sharpe, so the Pareto-front train-mean is a known positive value.

    fitness[0] is neg_sharpe (NSGA-III minimizes), so a positive Sharpe S means
    fitness[0] = -S. Objectives 1..n are set to 0.0 (well inside the feasible
    band, never the <=-1e5 sentinel) so the whole population is a single front
    and the front-mean train Sharpe equals exactly ``sharpe``.
    """

    def __init__(self, *args, sharpe: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_sharpe = float(sharpe)

    def evaluate(self, entry_tree, exit_tree, size_tree, delta_tree=None,
                 stop_mult=None):
        # stop_mult accepted to mirror the real evaluator signature (the GP stop
        # gene); ignored here since this mock returns a fixed Sharpe.
        vec = np.zeros(len(self.objectives), dtype=np.float64)
        vec[0] = -self._fixed_sharpe  # neg_sharpe → Sharpe = +sharpe
        return vec


def _train_front_mean(front):
    """Recompute the train-front mean Sharpe EXACTLY as evolve()/the val
    evaluator do: mean of (-fitness[0]) over the front, sentinel-filtered."""
    fs = [
        -float(i.fitness[0]) for i in front
        if i.fitness is not None and -float(i.fitness[0]) > -1e5
    ]
    return float(np.mean(fs)) if fs else 0.0


# ---------------------------------------------------------------------------
# B1 (b): losing fold — NO positive edge → early-stop must NOT fire.
# ---------------------------------------------------------------------------

def test_losing_fold_does_not_early_stop():
    """A LOSING fold (negative in-sample edge) with a large, persistent
    train-val gap must NOT be early-stopped: ordinary OOS degradation is not
    overfitting. The run must complete ALL generations.

    The default tiny-data evaluator yields a strongly NEGATIVE front Sharpe, so
    the val stub forces a gap of +2.0 on EVERY checkpoint (would trip the old
    3-checkpoint counter). With the fix the negative-edge precondition keeps the
    counter at 0 and the run runs to completion.
    """
    grammar = Grammar(max_depth=3, max_nodes=15)
    # 6 generations + interval 1 ⇒ checkpoints at gens 1..5 = 5 checks
    # (>= the 3 needed to trip the old detector).
    config = EvolutionConfig(pop_size=8, n_generations=6, seed=7, n_partitions=4)
    fe = _fresh_evaluator()

    observed_train_means = []

    def _val_stub(front):
        m = _train_front_mean(front)
        observed_train_means.append(m)
        return m - 2.0  # force gap = +2.0 every checkpoint

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _pop, metrics, _vt = evolve(
            iron_condor(), fe, grammar, config,
            val_evaluator=_val_stub, val_checkpoint_interval=1,
        )

    # Precondition sanity: the in-sample edge really is non-positive here.
    assert observed_train_means, "val stub never ran — no checkpoint occurred"
    assert all(m <= 0.0 for m in observed_train_means), (
        f"test premise broken: expected non-positive train edge, got "
        f"{observed_train_means}"
    )
    # The run completed every generation (no early-stop break).
    assert len(metrics) == config.n_generations, (
        f"losing fold was early-stopped at {len(metrics)} gens — the negative "
        f"in-sample-edge precondition failed to suppress the overfit gate"
    )
    # And no early-stop warning was emitted.
    assert not any("Early stop at gen" in str(w.message) for w in caught), (
        "early-stop warning fired on a losing fold with no in-sample edge"
    )


# ---------------------------------------------------------------------------
# B1 (a)+(b): genuine overfit — positive edge + much worse val → MUST fire.
# ---------------------------------------------------------------------------

def test_genuine_overfit_does_early_stop():
    """A genuine-overfit fold (POSITIVE in-sample edge, val much worse) must
    STILL be early-stopped after 3 consecutive over-gap checkpoints. This
    guards against the fix over-correcting and disabling early-stop entirely.

    The evaluator is forced to a deterministic positive front Sharpe (+1.5);
    the val stub returns train_mean - 2.0 (gap +2.0 > 1.0) every checkpoint, so
    the counter reaches 3 and the loop breaks.
    """
    grammar = Grammar(max_depth=3, max_nodes=15)
    # interval 1 ⇒ checks at gens 1,2,3,... → 3rd consecutive over-gap at gen 3.
    config = EvolutionConfig(pop_size=8, n_generations=10, seed=7, n_partitions=4)
    fe = _FixedSharpeEvaluator(
        template=iron_condor(), data=_make_data(),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        sharpe=1.5,
    )

    observed_train_means = []

    def _val_stub(front):
        m = _train_front_mean(front)
        observed_train_means.append(m)
        return m - 2.0  # gap = +2.0 with POSITIVE edge → genuine overfit

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _pop, metrics, _vt = evolve(
            iron_condor(), fe, grammar, config,
            val_evaluator=_val_stub, val_checkpoint_interval=1,
        )

    # Premise sanity: the in-sample edge really is positive.
    assert observed_train_means, "val stub never ran — no checkpoint occurred"
    assert observed_train_means[0] == pytest.approx(1.5), (
        f"forced positive edge not observed: {observed_train_means[:3]}"
    )
    # Early-stop fired: the run stopped BEFORE exhausting all generations.
    assert len(metrics) < config.n_generations, (
        f"genuine overfit was NOT early-stopped (ran all "
        f"{config.n_generations} gens) — the fix disabled early-stop entirely"
    )
    # It stopped at gen 3 (3rd consecutive over-gap checkpoint at interval 1).
    assert len(metrics) == 4, (  # gens 0,1,2,3 executed → break at gen 3
        f"expected early-stop at gen 3 (4 gens executed), got {len(metrics)}"
    )
    # And the early-stop warning was emitted.
    assert any("Early stop at gen" in str(w.message) for w in caught), (
        "genuine overfit did not emit the early-stop warning"
    )


def test_positive_edge_small_gap_does_not_early_stop():
    """Control: a POSITIVE in-sample edge with only a SMALL train-val gap
    (<= 1.0) must NOT early-stop — the gate keys on the GAP, not merely on the
    presence of an edge. Confirms the comparator threshold still governs."""
    grammar = Grammar(max_depth=3, max_nodes=15)
    config = EvolutionConfig(pop_size=8, n_generations=6, seed=7, n_partitions=4)
    fe = _FixedSharpeEvaluator(
        template=iron_condor(), data=_make_data(),
        backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        sharpe=1.5,
    )

    def _val_stub(front):
        # Gap of only 0.5 (< 1.0): val tracks train closely → not overfitting.
        return _train_front_mean(front) - 0.5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _pop, metrics, _vt = evolve(
            iron_condor(), fe, grammar, config,
            val_evaluator=_val_stub, val_checkpoint_interval=1,
        )

    assert len(metrics) == config.n_generations, (
        f"positive edge with a SMALL gap was early-stopped at {len(metrics)} "
        f"gens — the gap threshold is being ignored"
    )
    assert not any("Early stop at gen" in str(w.message) for w in caught)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
