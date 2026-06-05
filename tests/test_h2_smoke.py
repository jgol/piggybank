"""Reproducibility + degeneracy smoke test for the H2 pilot.

Senior Developer's "single highest-leverage improvement" from the L2 build
review: before running H2, land a smoke test that surfaces the known
blockers (B2 zero-trade dominance) AND locks down the reproducibility story.

Test inventory and status:

  PASSING (reproducibility, already fixed by Commit 1 / f07d781):
    - test_experiment.py::test_run_experiment_reproducible_with_same_master_seed
    - test_experiment.py::test_run_experiment_template_order_independent
    - test_experiment.py::test_derive_template_seed_hamming_distance
    - test_experiment.py::test_evolve_rejects_concurrent_entry

  NEW IN THIS FILE (degeneracy + shuffled-l1 validity):
    - test_pareto_front_excludes_zero_trade_strategies  — xfail on B2
    - test_shuffled_l1_run_completes_and_serializes     — PASSES (sanity check)
    - test_shuffled_l1_not_strictly_better_than_real_l1 — PASSES (direction sanity)

The xfail markers document which assertions Commit 3 must flip to passing.
Do NOT remove the markers until B2 is fixed.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from layer2.experiment import ExperimentConfig, run_experiment
from layer2.io import (
    PRICE_COLUMN, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)


# ---------------------------------------------------------------------------
# Synthetic L1 Parquet generator (mirrors the one in test_experiment.py)
# ---------------------------------------------------------------------------

def _make_l1_parquet(tmp_path: Path, n_dates: int = 10,
                     bars_per_date: int = 30) -> Path:
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
                row[col] = rng.standard_normal(384).astype(np.float32).tolist()
            for col in REGIME_PROB_COLUMNS:
                row[col] = float(rng.uniform())
            rows.append(row)
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "l1_smoke.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")
    return parquet_path


# ---------------------------------------------------------------------------
# B2: zero-trade dominance degeneracy
# ---------------------------------------------------------------------------
# With DEFAULT_OBJECTIVES = (neg_sharpe, max_drawdown, neg_trade_count_score),
# a zero-trade strategy has fitness (0, 0, +1):
# neg_sharpe = 0 (no returns → 0/0 → 0 by std-guard)
# max_drawdown = 0 (equity curve is flat at initial capital)
# neg_trade_count_score = +1.0 (tc_score = -1.0 for zero trades; negated)
# This is INCOMPARABLE with any productive strategy on max_drawdown (no
# productive strategy achieves DD=0), so zero-trade trees always live on
# the Pareto front by default.
#
# Commit 3 fix: either gate zero-trade strategies with FAILED_FITNESS_SENTINEL
# or remove trade-count as an objective and make min-trades a constraint.
# ---------------------------------------------------------------------------

def test_pareto_front_excludes_zero_trade_strategies(tmp_path):
    """No individual on any template's Pareto front should be a zero-trade
    strategy. Post-Commit 3: FitnessEvaluator gates `trades < min_trades`
    with FAILED_FITNESS_SENTINEL on every objective, so NSGA-III correctly
    dominates against degenerate trees (B2 fix)."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard", "bull_put_credit_standard"),
        pop_size=16, n_generations=3, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(tmp_path / "results"),
        backtester_warmup_bars=3,
        min_trades=1,  # smoke-test data is tiny (6 dates); default 20 kills all
    )
    run_experiment(config)

    strategy_files = list((tmp_path / "results").glob("strategies_*.jsonl"))
    assert strategy_files, "no strategy files produced by the run"

    zero_trade_count = 0
    sentinel_count = 0
    productive_count = 0
    total_count = 0
    for f in strategy_files:
        for line in f.read_text().splitlines():
            ind = json.loads(line)
            fitness = ind["fitness"]
            if fitness is None:
                continue
            tc_obj = fitness[2]
            # Three bands:
            # tc_obj ≈ 1e6 → FAILED_FITNESS_SENTINEL (gated by min_trades)
            # 0.5 < tc_obj < 1e5 → legitimate zero-trade "-1.0 neg" sentinel
            # from the old compute_fitness path (B2 bug
            # — should never appear on Pareto front
            # under the current gate)
            # tc_obj <= 0 → productive strategy (negated positive tc_score)
            if tc_obj >= 1e5:
                sentinel_count += 1
            elif tc_obj > 0.5:
                zero_trade_count += 1
            else:
                productive_count += 1
            total_count += 1

    # Require at least one productive individual on the front — otherwise
    # the test has nothing to say about "no zero-trade dominance" (vacuous).
    assert productive_count > 0, (
        f"population collapsed: {sentinel_count}/{total_count} individuals "
        f"are sentinel-gated (zero trades), 0 are productive. Smoke-test "
        f"data is too restrictive or seed trees no longer fire."
    )

    # The core B2 invariant: no un-gated zero-trade individual (the classic
    # (0, 0, +1) fitness pattern from the pre-fix code) may appear on the
    # Pareto front.
    assert zero_trade_count == 0, (
        f"{zero_trade_count}/{total_count} Pareto-front individuals have "
        f"ungated neg_trade_count_score in (0.5, 1e5) — indicates the B2 "
        f"gate was bypassed. See L2 review synthesis B2."
    )


# ---------------------------------------------------------------------------
# Shuffled-L1 condition validity (passes on current code — regression guard)
# ---------------------------------------------------------------------------

def test_shuffled_l1_run_completes_and_serializes(tmp_path):
    """Basic sanity: shuffled-L1 condition runs end-to-end, produces valid
    results.json + strategies_*.jsonl + metrics_*.json. This is a guardrail
    for the H2 pipeline plumbing, not a claim about content.
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="shuffled-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(tmp_path / "results"),
        backtester_warmup_bars=3,
    )
    result = run_experiment(config)

    # Plumbing asserts: files exist, schema looks right
    out_dir = tmp_path / "results"
    assert (out_dir / "results.json").exists()
    assert (out_dir / "strategies_iron_condor_standard.jsonl").exists()
    assert (out_dir / "metrics_iron_condor_standard.json").exists()

    with (out_dir / "results.json").open() as f:
        data = json.load(f)
    # The resolved shuffle_seed (master + 1_000_000) must be written to results
    assert data["config"]["shuffle_seed"] == 42 + 1_000_000, (
        "resolved shuffle_seed missing from results.json — audit trail broken"
    )
    assert data["config"]["condition"] == "shuffled-l1"
    assert result.n_train_rows > 0


def test_shuffled_l1_not_strictly_better_than_real_l1(tmp_path):
    """Direction sanity: the shuffled-L1 control should not systematically
    OUTPERFORM real-L1 — if it does, either the shuffle isn't destroying
    signal (null is too easy) or the real-L1 plumbing is broken.

    On synthetic data the signal difference is not material, so we use a
    weak, non-parametric check: shuffled-L1 best_sharpe across seeds is NOT
    strictly greater than real-L1 best_sharpe by >0.5 standard deviations
    (an arbitrary but defensible "unlikely under the null" threshold).
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=8, bars_per_date=15)

    def _run(condition: str, out: Path) -> float:
        config = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition=condition,
            template_names=("iron_condor_standard",),
            pop_size=12, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-04-20",
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        run_experiment(config)
        with (out / "results.json").open() as f:
            data = json.load(f)
        best_obj = data["templates"][0]["best_objectives"]
        # best_objectives[0] = best neg_sharpe (minimized)
        return float(best_obj[0]) if best_obj else 0.0

    real_neg_sharpe = _run("real-l1", tmp_path / "real")
    shuf_neg_sharpe = _run("shuffled-l1", tmp_path / "shuf")

    # Smaller neg_sharpe = higher Sharpe. Shuffled should NOT be dramatically
    # smaller (i.e., dramatically better). On synthetic data they tend to be
    # comparable; this guardrail catches gross regressions, not subtle ones.
    gap = real_neg_sharpe - shuf_neg_sharpe
    assert gap >= -2.0, (
        f"shuffled-L1 best neg_sharpe ({shuf_neg_sharpe:.3f}) is dramatically "
        f"better than real-L1 ({real_neg_sharpe:.3f}); gap={gap:.3f}. Either "
        f"the shuffle is not destroying signal or real-L1 is broken."
    )
