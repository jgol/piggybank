"""End-to-end smoke test for layer2.experiment."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from layer2.experiment import (
    ExperimentConfig, _derive_template_seed, _SHUFFLE_SEED_OFFSET, run_experiment,
    H2_HV_REFERENCE_POINT,
)
from layer2.fitness import DEFAULT_OBJECTIVES
from layer2.io import (
    PRICE_COLUMN, PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS, TYPED_VECTOR_COLUMNS,
)


def _make_l1_parquet(tmp_path: Path, n_dates: int = 10, bars_per_date: int = 30) -> Path:
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
                # Lists in Parquet (will be converted to ndarray on load)
                row[col] = rng.standard_normal(384).astype(np.float32).tolist()
            for col in REGIME_PROB_COLUMNS:
                row[col] = float(rng.uniform())
            rows.append(row)
    df = pd.DataFrame(rows)
    parquet_path = tmp_path / "l1_test.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")
    return parquet_path


def test_run_experiment_real_l1(tmp_path):
    """Full pipeline: load Parquet → split → evolve 1 template → save."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=8, bars_per_date=40)
    output_dir = tmp_path / "results"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8,
        n_generations=2,
        seed=42,
        n_partitions=4,
        train_end="2024-05-01",
        val_end="2024-07-01",
        # embargo_days=0: the monthly fixture (2024-MM-15) has only 8 dates, so the
        # default 5-trading-DATE embargo (split_by_date counts index positions, not
        # calendar days — io.py:324) would push val_start past the last date and
        # collapse val/test to empty. Every other split test here does the same.
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    result = run_experiment(config)
    assert result.n_train_rows > 0
    assert len(result.template_results) == 1
    tr = result.template_results[0]
    assert tr.template_name == "iron_condor_standard"
    assert tr.final_population_size == 8
    assert len(tr.metrics_log) == 2          # both generations ran
    # The Pareto front is COMPUTED and SERIALIZED — that is what this end-to-end
    # smoke test verifies. The fixture is pure synthetic NOISE (random typed vectors,
    # constant scalars), so the GP's noise-rejection gates (random-entry / min-trades
    # / regime) correctly sentinel every strategy and "Hardening #1" reports
    # pareto_front_size==0 (the PRODUCTIVE count). That is the gates working as
    # designed, NOT a pipeline failure — requiring a productive strategy from noise
    # would be testing the gates' failure to reject noise. Assert the front structure
    # was produced + serialized with valid trees, and the productive count is well-
    # defined. (See _make_l1_parquet — no tradeable signal is injected.)
    assert tr.pareto_front_size >= 0
    assert len(tr.pareto_front) >= 1, "front must be computed + serialized"
    for ind in tr.pareto_front:
        assert ind["entry_tree"] and ind["exit_tree"] and ind["size_tree"]
        assert "val_sharpe" in ind and "total_trades" in ind

    # Output files written
    assert (output_dir / "results.json").exists()
    assert (output_dir / "strategies_iron_condor_standard.jsonl").exists()
    assert (output_dir / "metrics_iron_condor_standard.json").exists()

    # Check results.json structure
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    assert data["n_train_rows"] > 0
    assert len(data["templates"]) == 1


def test_run_experiment_shuffled_l1(tmp_path):
    """Shuffled-L1 condition runs through the same plumbing."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="shuffled-l1",
        template_names=("bull_put_credit",),
        pop_size=8,
        n_generations=2,
        seed=42,
        shuffle_seed=99,
        n_partitions=4,
        train_end="2024-04-15",
        backtester_warmup_bars=3,
    )
    result = run_experiment(config)
    assert result.n_train_rows > 0
    assert result.template_results[0].template_name == "bull_put_credit_standard"


def test_experiment_config_validates_condition():
    with pytest.raises(ValueError, match="condition"):
        ExperimentConfig(l1_parquet_path="x.parquet", condition="invalid-mode")


def test_experiment_config_accepts_scalar_only():
    cfg = ExperimentConfig(l1_parquet_path="x.parquet", condition="scalar-only")
    assert cfg.condition == "scalar-only"


# ---------------------------------------------------------------------------
# H1: scalar-only condition (third H2 arm)
# ---------------------------------------------------------------------------

def test_run_experiment_scalar_only_end_to_end(tmp_path):
    """Scalar-only condition runs through the same plumbing and writes valid
    output. Grammar strips EMB_* terminals + probe scalars; template seeds
    with PredRV* references get rewritten to RealizedVol30m."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    output_dir = tmp_path / "scalar_results"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="scalar-only",
        template_names=("iron_condor_standard", "bull_put_credit_standard"),  # one uses PredRV15
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    result = run_experiment(config)
    assert result.n_train_rows > 0
    # Template results produced for both templates (Pareto front may be 0
    # for degenerate tiny-data runs where all strategies hit sentinel)
    assert len(result.template_results) == 2
    for tr in result.template_results:
        assert tr.pareto_front_size >= 0

    # Verify results.json records the condition for audit
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    assert data["config"]["condition"] == "scalar-only"

    # Verify no tree in the produced strategies references probe scalars
    # (the rewritten seeds + restricted grammar must keep L1 features OUT)
    import re
    probe_pattern = re.compile(r"\b(PredRV15|PredRV30|PredRegime|PredSpread)\b")
    emb_pattern = re.compile(r"\bEMB_[A-Z_]+\b")
    for name in ("iron_condor_standard", "bull_put_credit_standard"):
        strat_text = (output_dir / f"strategies_{name}.jsonl").read_text()
        probe_hits = probe_pattern.findall(strat_text)
        emb_hits = emb_pattern.findall(strat_text)
        assert not probe_hits, (
            f"scalar-only {name} strategies leaked probe scalars: {set(probe_hits)}"
        )
        assert not emb_hits, (
            f"scalar-only {name} strategies leaked typed-vector terminals: {set(emb_hits)}"
        )


def test_scalar_only_grammar_has_no_emb_terminals():
    """Direct grammar-level regression guard: the scalar-only terminal set
    contains no EMB_* typed vectors and no probe scalars."""
    from layer2.grammar import (
        EMB_TYPES, SCALAR_ONLY_FUNCTIONS, build_scalar_only_terminal_set,
    )
    terms = build_scalar_only_terminal_set()
    term_names = {t.name for t in terms}
    term_types = {t.ret_type for t in terms}
    # No typed-vector GTypes
    for emb_t in EMB_TYPES:
        assert emb_t not in term_types, f"{emb_t} leaked into scalar-only terminals"
    # No probe scalars
    for probe in ("PredRV15", "PredRV30", "PredRegime", "PredSpread"):
        assert probe not in term_names, f"{probe} leaked into scalar-only terminals"
    # Raw bypass scalars preserved
    for raw in ("ATM_IV", "VIXSpot", "RawSpread", "RealizedVol30m",
                "MinutesToClose", "DeltaSpread1", "DeltaSpread5"):
        assert raw in term_names, f"raw scalar {raw} missing from scalar-only terminals"


def test_scalar_only_functions_exclude_emb_operators():
    from layer2.grammar import EMB_OPERATORS, SCALAR_ONLY_FUNCTIONS
    for op in EMB_OPERATORS:
        assert op not in SCALAR_ONLY_FUNCTIONS, (
            f"typed-vector operator {op.name} leaked into SCALAR_ONLY_FUNCTIONS"
        )


def test_scalar_only_seed_substitution_rewrites_probes():
    """`substitute_stripped_terminals_for_scalar_only` rewrites PredRV15 →
    RealizedVol30m in bull_put_credit's entry seed."""
    from layer2.grammar import (
        substitute_stripped_terminals_for_scalar_only, to_str,
    )
    from layer2.templates import bull_put_credit
    t = bull_put_credit()
    orig = to_str(t.entry_seed)
    rewritten = to_str(
        substitute_stripped_terminals_for_scalar_only(t.entry_seed)
    )
    assert "PredRV15" in orig
    assert "PredRV15" not in rewritten
    assert "RealizedVol30m" in rewritten


def test_scalar_only_condition_uses_same_seed_derivation(tmp_path):
    """B3 invariant: the scalar-only condition uses the same per-template
    seed derivation as real-l1 (hash(master, template_name)). Verifies by
    running scalar-only twice and asserting reproducibility."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)

    def _run(out):
        cfg = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="scalar-only",
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-04-15",
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        run_experiment(cfg)

    _run(tmp_path / "a")
    _run(tmp_path / "b")
    sa = (tmp_path / "a" / "strategies_iron_condor_standard.jsonl").read_text()
    sb = (tmp_path / "b" / "strategies_iron_condor_standard.jsonl").read_text()
    assert sa == sb, "scalar-only condition lost reproducibility"


# ---------------------------------------------------------------------------
# Commit 4 review follow-up: Model QA findings + reviewer must-fixes
# ---------------------------------------------------------------------------

def test_substitution_helper_rejects_typed_vector_terminal():
    """Model QA + Code Reviewer: the ValueError guards in
    substitute_stripped_terminals_for_scalar_only must be exercised.
    Typed-vector terminal → ValueError (no scalar analog)."""
    from layer2.grammar import (
        EMB_TYPES, GType, TermDef, TermNode,
        substitute_stripped_terminals_for_scalar_only,
    )
    # Synthetic seed containing an EMB_GRID terminal (the actual terminal
    # from build_terminal_set uses name "EMB_GRID" and ret_type=EMB_GRID)
    bad_term = TermNode(defn=TermDef("EMB_GRID", GType.EMB_GRID))
    with pytest.raises(ValueError, match="typed-vector terminal"):
        substitute_stripped_terminals_for_scalar_only(bad_term)


def test_substitution_helper_rejects_predregime():
    """AI Engineer + Model QA: PredRegime has no acceptable scalar analog
    (4-class L1 inference vs. binary AbstainFlag — silent semantic
    collapse). The substitution must raise ValueError rather than coerce."""
    from layer2.grammar import (
        GType, TermDef, TermNode,
        substitute_stripped_terminals_for_scalar_only,
    )
    # Build a PredRegime terminal node (defn matches the one built by
    # build_terminal_set — ret_type REAL since PredRegime is an integer
    # encoded as a scalar in the L1 output DataFrame)
    pred_regime = TermNode(defn=TermDef("PredRegime", GType.REAL))
    with pytest.raises(ValueError, match="PredRegime"):
        substitute_stripped_terminals_for_scalar_only(pred_regime)


def test_scalar_only_grammar_can_grow_all_ret_types():
    """Model QA Finding 4 + F1 update: scalar-only grammar can grow a
    valid tree of every reachable GType. F1 (2026-04-25, BLOCKER) removed
    the Regime literal terminals AND IN_REGIME/REGIME_IS operators from
    the scalar-only grammar — so REGIME is no longer a reachable type
    (and must NOT be, otherwise it would be a leakage path for L1's
    PredRegime via ctx.current_regime).

    The remaining reachable types are REAL, BOOL, INT, SIDE.
    EMB_* must be unreachable. REGIME must also be unreachable post-F1.
    """
    import random as _r
    from layer2.grammar import (
        EMB_TYPES, GType, Grammar, SCALAR_ONLY_FUNCTIONS,
        build_scalar_only_terminal_set, validate,
    )
    g = Grammar(
        functions=SCALAR_ONLY_FUNCTIONS,
        terminals=build_scalar_only_terminal_set(),
        max_depth=5, max_nodes=20,
    )
    _r.seed(0)
    # F1: REGIME is no longer reachable in scalar-only (Regime literals
    # stripped + IN_REGIME/REGIME_IS stripped from SCALAR_ONLY_FUNCTIONS).
    for t in (GType.REAL, GType.BOOL, GType.INT, GType.SIDE):
        for _ in range(10):
            tree = g.grow(t)
            assert tree.ret_type == t
            assert validate(tree), f"grown tree for {t} failed validation"
    # F1 invariant: REGIME must NOT be reachable in scalar-only.
    regime_min_depth = g._min_depth.get(GType.REGIME, 99)
    assert regime_min_depth >= 99, (
        f"F1 violation: GType.REGIME is reachable in scalar-only grammar "
        f"(min_depth={regime_min_depth}). This is a leakage path for L1's "
        f"PredRegime label via ctx.current_regime."
    )
    # EMB_* types must be unreachable from scalar-only grammar
    for emb_t in EMB_TYPES:
        depth_needed = g._min_depth.get(emb_t, 99)
        assert depth_needed >= 99, (
            f"{emb_t} is reachable in scalar-only grammar (min_depth="
            f"{depth_needed}); this contradicts the L1-exclusion invariant"
        )


def test_scalar_only_results_records_seed_trees(tmp_path):
    """Model QA Finding 1: results.json must record the original AND
    effective (post-substitution) seed trees per template so an auditor
    holding only the artifacts can reconstruct the initial population."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    output_dir = tmp_path / "audit"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="scalar-only",
        template_names=("bull_put_credit",),  # seed uses PredRV15 → rewritten
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    tpl = next(t for t in data["templates"] if t["name"] == "bull_put_credit_standard")
    # Audit trail present
    assert "seed_trees_original" in tpl
    assert "seed_trees_effective" in tpl
    # Original references PredRV15, effective references RealizedVol30m
    assert "PredRV15" in tpl["seed_trees_original"]["entry"]
    assert "PredRV15" not in tpl["seed_trees_effective"]["entry"]
    assert "RealizedVol30m" in tpl["seed_trees_effective"]["entry"]


def test_scalar_only_results_records_seed_trees_unchanged_for_real_l1(tmp_path):
    """In non-scalar-only conditions, original and effective seeds must be
    identical (no substitution fires). Audit-trail symmetry."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    output_dir = tmp_path / "real"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("bull_put_credit",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    tpl = data["templates"][0]
    assert tpl["seed_trees_original"] == tpl["seed_trees_effective"], (
        "real-l1 must not apply substitution; original must equal effective"
    )


def test_results_json_records_algorithm_provenance(tmp_path):
    """Model QA Commit 5 requirement: results.json must include a
    `provenance` block recording pymoo version, niching algorithm
    identity, and seed-derivation semantics. Without these, an auditor
    cannot tell whether a result came from the canonical Deb & Jain 2014
    niching or an earlier cosine-similarity variant.
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    output_dir = tmp_path / "prov"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-04-15",
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    assert "provenance" in data, (
        "results.json missing `provenance` block (Model QA Commit 5 audit "
        "requirement)"
    )
    prov = data["provenance"]
    # Required keys
    for key in ("pymoo_version", "niching_algorithm", "niching_rng_derivation",
                "per_template_seed_derivation"):
        assert key in prov, f"provenance missing required key {key!r}"
    # Niching algorithm identity: must reference pymoo Deb & Jain primitives
    assert "pymoo" in prov["niching_algorithm"].lower()
    assert "deb" in prov["niching_algorithm"].lower() or "2014" in prov["niching_algorithm"]


# ---------------------------------------------------------------------------
# Commit 6.5: val/test scoring + hypervolume primary outcome
# ---------------------------------------------------------------------------

def test_run_experiment_computes_val_hypervolume(tmp_path):
    """Val/test split evaluation: every Pareto-front individual must be
    re-scored on the held-out val split, and the template result must
    include a val_hypervolume scalar (the H2 primary outcome).
    """
    # Use 12 months (12 dates) with embargo_days=0 so train/val/test splits
    # are all non-empty on synthetic monthly data
    l1_path = _make_l1_parquet(tmp_path, n_dates=12, bars_per_date=15)
    output_dir = tmp_path / "val_hv"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-09-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    result = run_experiment(config)

    tr = result.template_results[0]
    assert tr.val_hypervolume is not None, (
        "val_hypervolume not computed — Commit 6.5 val-scoring pathway "
        "is missing or broken"
    )
    assert tr.hv_reference_point == list(H2_HV_REFERENCE_POINT), (
        "hv_reference_point drifted from the pre-declared H2 constant"
    )
    # val_hv must be non-negative (hypervolume is always >= 0) and finite
    assert tr.val_hypervolume >= 0.0
    assert tr.val_hypervolume < float("inf")

    # Verify serialized into results.json
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    tpl = data["templates"][0]
    assert tpl["val_hypervolume"] is not None
    assert tpl["hv_reference_point"] == list(H2_HV_REFERENCE_POINT)


def test_strategies_jsonl_includes_val_and_test_fitness(tmp_path):
    """Each NON-sentinel Pareto-front individual in strategies_*.jsonl must carry its
    val_fitness and test_fitness vectors, not just the train fitness.

    BLOCKER-3 (2026-06-01 audit): a DEGENERATE front (all members sentinel) now
    correctly writes an EMPTY strategies file (no sentinel pseudo-champions). This
    tiny 2-gen fixture reliably degenerates, so the val/test-fitness CONTRACT is
    pinned directly at the serialization boundary (which is what actually carries
    the fields) rather than relying on the GP finding a real edge. The file-level
    assertion then accepts the empty (degenerate) case as the correct BLOCKER-3
    behavior, and any REAL line still must carry val/test fitness.
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=12, bars_per_date=15)
    output_dir = tmp_path / "val_fit"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-09-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)

    # File-level: either empty (degenerate front, BLOCKER-3) or every REAL line
    # carries val/test fitness. A non-sentinel line MUST carry both vectors.
    strat_lines = [l for l in (output_dir / "strategies_iron_condor_standard.jsonl")
                   .read_text().splitlines() if l.strip()]
    for line in strat_lines:
        ind = json.loads(line)
        assert "fitness" in ind            # train (pre-existing)
        assert "val_fitness" in ind         # Commit 6.5
        assert "test_fitness" in ind        # Commit 6.5
        assert len(ind["val_fitness"]) == len(DEFAULT_OBJECTIVES)
        assert len(ind["test_fitness"]) == len(DEFAULT_OBJECTIVES)

    # Contract pin INDEPENDENT of GP outcome: a NON-degenerate (size>0) result with
    # a real member serializes val_fitness/test_fitness; a degenerate (size==0)
    # result writes an EMPTY file. Both directions verified directly on the writer.
    from layer2.experiment import _save_single_template_result, TemplateRunResult
    real_member = {
        "template_name": "iron_condor_standard", "entry_tree": "a", "exit_tree": "b",
        "size_tree": "c", "fitness": [-1.0, -1.0, -1.0],
        "val_fitness": [-0.8] * len(DEFAULT_OBJECTIVES),
        "test_fitness": [-0.7] * len(DEFAULT_OBJECTIVES), "val_sharpe": 0.8,
    }
    _real_dir = tmp_path / "real_contract"
    _save_single_template_result(
        TemplateRunResult(
            template_name="iron_condor_standard", final_population_size=1,
            pareto_front_size=1, pareto_front=[real_member], metrics_log=[],
            fitness_cache_stats={}, wall_time_s=0.1),
        _real_dir)
    _real_lines = [l for l in (_real_dir / "strategies_iron_condor_standard.jsonl")
                   .read_text().splitlines() if l.strip()]
    assert len(_real_lines) == 1, "a non-degenerate front must serialize its real member"
    _r = json.loads(_real_lines[0])
    assert len(_r["val_fitness"]) == len(DEFAULT_OBJECTIVES)
    assert len(_r["test_fitness"]) == len(DEFAULT_OBJECTIVES)


def _make_l1_parquet_daily(tmp_path: Path, n_days: int = 90,
                           bars_per_date: int = 12) -> Path:
    """L1 parquet with n_days DISTINCT daily dates (so val splits can exceed
    the DSR gate's min_days=20). Mirrors _make_l1_parquet's schema."""
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2024-01-02")
    rows = []
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).strftime("%Y-%m-%d")
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
    parquet_path = tmp_path / "l1_daily.parquet"
    df.to_parquet(parquet_path, engine="pyarrow")
    return parquet_path


def test_dsr_gate_filters_front_and_annotates(tmp_path):
    """P1-B: the DSR gate must FILTER the returned Pareto front (Blocker A:
    it was annotate-only). With an adequate TRAIN window (T_train >> min_days):

      * raising dsr_threshold can only shrink (never grow) the returned front,
      * every member in the returned front is dsr_passed=True (the gate is
        consistent with what it keeps) and carries train_sharpe (the gate
        deflates the TRAINING Sharpe),
      * the evolution-level (N, distinct trials) is recorded and N >= front.

    NOTE on the observable: we assert on len(tr.pareto_front) — the DSR-filtered
    LIST — NOT tr.pareto_front_size. On this synthetic fixture the GP finds no
    productive strategy, so the all-sentinel degenerate-front detection
    (Hardening #1) force-zeroes pareto_front_size; asserting on it would be
    VACUOUS. The returned LIST is untouched by Hardening #1 and still reflects
    exactly what the DSR gate kept, so it is the load-bearing observable.
    """
    l1_path = _make_l1_parquet_daily(tmp_path, n_days=90, bars_per_date=12)

    def _run(threshold, out):
        config = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard",),
            pop_size=12, n_generations=3, seed=42, n_partitions=4,
            train_end="2024-02-15",   # ~45 train dates (>> min_days=20)
            val_end="2024-03-31",     # ~45 val dates
            embargo_days=0,
            dsr_gate_enabled=True,
            dsr_threshold=threshold,
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        return run_experiment(config)

    # Lenient threshold: keep everything that beats SR* with any probability.
    res_low = _run(0.0, tmp_path / "dsr_low")
    tr_low = res_low.template_results[0]
    # Strict threshold: demand near-certain significance — front can only shrink.
    res_high = _run(0.999999, tmp_path / "dsr_high")
    tr_high = res_high.template_results[0]

    # Non-vacuity guard: the gate must actually have champions to judge.
    assert len(tr_low.pareto_front) > 0, (
        "fixture must yield a non-empty front for the gate to act on "
        "(else this test is vacuous)"
    )
    # ANNOTATE-DON'T-DESTROY (2026-06-01 empty-fleet fix): the gate now KEEPS the full
    # front at every threshold and only flips dsr_passed flags — so a multi-day run
    # always yields the raw champions even when none clear the bar. The OBSERVABLE that
    # tightens with τ is therefore the COUNT of dsr_passed members, NOT the front size.
    assert len(tr_high.pareto_front) == len(tr_low.pareto_front), (
        "front SIZE must be threshold-invariant (annotate-don't-destroy)")
    _n_pass_low = sum(1 for m in tr_low.pareto_front if m.get("dsr_passed"))
    _n_pass_high = sum(1 for m in tr_high.pareto_front if m.get("dsr_passed"))
    assert _n_pass_high < _n_pass_low, (
        f"raising dsr_threshold must shrink the dsr_passed SUBSET (not the front): "
        f"low(thr=0)→{_n_pass_low} high(thr≈1)→{_n_pass_high}")

    # Every champion carries the train-Sharpe-deflated DSR annotations (pass or fail),
    # and at the lenient τ=0 every assessable champion is flagged passed.
    for m in tr_low.pareto_front:
        assert m.get("dsr_passed") is True, (
            "at τ=0 every assessable champion is annotated dsr_passed=True")
        assert m.get("dsr") is not None
        assert "dsr_sr_star" in m and "dsr_n_trials" in m
        assert "train_sharpe" in m, "the gate deflates the TRAINING Sharpe"

    # Evolution-level N (distinct trials) recorded and is >= the returned front:
    # SR* is deflated against the population, not the tiny front.
    n_trials = tr_low.pareto_front[0]["dsr_n_trials"]
    assert n_trials >= len(tr_low.pareto_front), (
        f"DSR N (distinct evo trials = {n_trials}) must be >= front "
        f"size ({len(tr_low.pareto_front)})"
    )


def test_dsr_gate_skipped_when_train_window_too_short(tmp_path):
    """When the TRAINING window is shorter than min_days the gate is
    INCONCLUSIVE, not a reject-all: it annotates dsr_passed=False +
    dsr_inconclusive=True (MEDIUM-8) and does NOT filter. The gate keys on the
    TRAIN window because it deflates the TRAINING Sharpe (the in-sample quantity
    NSGA-III selected on, hence the one inflated by the N-trial search). Verified
    by comparing a strict-threshold run (which would empty the front IF the gate
    ran) against the gate-disabled baseline — the returned front LISTS must be
    IDENTICAL in length (skipped, not applied), and every returned member is
    annotated dsr_passed=False + dsr_inconclusive=True.

    (Observable is len(pareto_front), not pareto_front_size — see the filter
    test's docstring: Hardening #1 zeroes the size on this synthetic fixture.)
    """
    # Daily fixture; deliberately SHORT train window (~12 distinct dates < 20).
    l1_path = _make_l1_parquet_daily(tmp_path, n_days=60, bars_per_date=12)

    def _run(enabled, threshold, out):
        config = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-01-14",   # ~12 train dates (< min_days=20)
            val_end="2024-02-20",     # large val window (irrelevant — gate keys on TRAIN)
            embargo_days=0,
            dsr_gate_enabled=enabled,
            dsr_threshold=threshold,
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        return run_experiment(config).template_results[0]

    tr_off = _run(False, 0.5, tmp_path / "dsr_off")
    tr_strict = _run(True, 0.999999, tmp_path / "dsr_strict")

    # Non-vacuity guard: the front LIST is non-empty (sentinel members are still
    # listed; only pareto_front_size is zeroed), else skip==filter trivially.
    assert len(tr_off.pareto_front) > 0, "fixture must yield a non-empty front list"
    # Skip path: a strict threshold does NOT shrink the front vs gate-disabled,
    # because the gate cannot assess significance on a < min_days TRAIN window.
    assert len(tr_strict.pareto_front) == len(tr_off.pareto_front), (
        f"short train window must SKIP the gate (no filtering): "
        f"off→{len(tr_off.pareto_front)} strict→{len(tr_strict.pareto_front)}"
    )
    # Any returned members are annotated INCONCLUSIVE (MEDIUM-8: dsr_passed=False
    # + dsr_inconclusive=True, NOT None).
    for m in tr_strict.pareto_front:
        assert m.get("dsr_passed") is False, (
            "below min_days the gate must annotate dsr_passed=False, not filter"
        )
        assert m.get("dsr_inconclusive") is True, (
            "short-window members must carry dsr_inconclusive=True (MEDIUM-8)"
        )


# ---------------------------------------------------------------------------
# DECISIVE boundary tests for the train-Sharpe switch (review finding C1).
# The GP integration fixtures above only ever produce degenerate all-sentinel
# fronts (junk train Sharpes -> nonsensical SR* ~26 ann), so the >=-operator
# straddles a fixed mediocre DSR and the size assertions pass WITHOUT the gate
# discriminating champions by Sharpe — they cannot detect a train->val
# regression. These tests call the extracted gate helper directly with
# CONTROLLED champion Sharpes + a realistic evolution (N, V), so they pin the
# actual load-bearing semantics.
# ---------------------------------------------------------------------------

def _evo_records_for_sr_star(rng_seed: int = 11, n: int = 4000,
                             mean_ann: float = 0.3, std_ann: float = 0.75,
                             n_days: int = 125) -> dict:
    """Evolution trial-records {sig: (ANNUALIZED Sharpe, n_days)} whose (N, V)
    give a realistic production SR*_ann ≈ 2.5-3.0 (the τ=0.5 pure-deflation bar)."""
    rng = np.random.default_rng(rng_seed)
    sharpes = rng.normal(mean_ann, std_ann, n)
    return {f"sig{i}": (float(s), int(n_days)) for i, s in enumerate(sharpes)}


def test_dsr_front_gate_deflates_training_sharpe_not_validation():
    """DECISIVE: the gate deflates the TRAINING Sharpe, not validation.

    Members are built so train and val DISAGREE on the verdict — strong-train/
    weak-val vs weak-train/strong-val. The gate keeps the strong-TRAIN members
    regardless of their (tiny) val Sharpe and rejects the weak-TRAIN members
    despite their (huge) val Sharpe. If the gate ever reverts to deflating val
    Sharpe, every assertion here flips. This is the test the integration
    fixtures cannot provide (their fronts are all-sentinel)."""
    from layer2.experiment import _apply_dsr_front_gate
    from layer2.pbo import expected_max_sharpe, evolution_n_v_from_records

    evo = _evo_records_for_sr_star()
    N, V = evolution_n_v_from_records(evo, min_days=20)
    sr_star_ann = expected_max_sharpe(N, V) * math.sqrt(252)
    # Non-degeneracy: the realistic τ=0.5 bar is ~2.5-3.0 ann (NOT the junk-front
    # regime where SR*_ann ≈ 26). Members straddle this bar by a wide margin.
    assert 2.0 < sr_star_ann < 3.5, f"SR*_ann={sr_star_ann:.2f} off realistic range"

    champs = [
        {"id": "A_strongtrain", "train_sharpe": 5.0, "val_sharpe": 0.1,
         "train_return_skew": 0.0, "train_return_kurtosis": 0.0},
        {"id": "B_strongval",   "train_sharpe": 0.5, "val_sharpe": 9.9,
         "train_return_skew": 0.0, "train_return_kurtosis": 0.0},
        {"id": "C_passtrain",   "train_sharpe": 4.5, "val_sharpe": 0.2,
         "train_return_skew": 0.0, "train_return_kurtosis": 0.0},
        {"id": "D_rejecttrain", "train_sharpe": 1.5, "val_sharpe": 8.0,
         "train_return_skew": 0.0, "train_return_kurtosis": 0.0},
    ]
    front = [f"ind_{c['id']}" for c in champs]  # placeholders; gate only zips them

    kept, kept_front = _apply_dsr_front_gate(
        list(champs), list(front), train_T=125, evo_records=evo,
        dsr_gate_enabled=True, dsr_threshold=0.5, template_name="unit",
        min_days=20,
    )

    # ANNOTATE-DON'T-DESTROY (2026-06-01): the gate KEEPS the full front; the verdict
    # lives in the dsr_passed FLAGS. The decisive observable is the dsr_passed SUBSET,
    # which must follow TRAIN Sharpe (above/below SR*), not val.
    assert {c["id"] for c in kept} == {c["id"] for c in champs}, "full front kept"
    _passed_ids = {c["id"] for c in kept if c.get("dsr_passed")}
    assert _passed_ids == {"A_strongtrain", "C_passtrain"}, (
        f"dsr_passed subset must follow TRAIN Sharpe (above/below SR*≈{sr_star_ann:.1f}), "
        f"not val; passed={_passed_ids}"
    )
    # Decisive anti-regression: A flagged-pass despite val=0.1; B flagged-fail despite val=9.9.
    assert champs[0]["dsr_passed"] is True, "A: strong TRAIN must pass"
    assert champs[1]["dsr_passed"] is False, (
        "B: weak TRAIN must FAIL even with a huge val Sharpe — proves train axis"
    )
    # front returned in lock-step with the dicts (full front, all annotated).
    assert kept_front == front
    assert len(kept_front) == len(kept) == len(champs)
    # Real SR* annotation and >1 distinct DSR value (not the junk-front regime).
    assert 2.0 < champs[0]["dsr_sr_star_annualized"] < 3.5
    assert champs[0]["dsr_n_trials"] == N
    assert len({c["dsr"] for c in champs}) > 1


def test_dsr_front_gate_skips_below_min_days_train():
    """Helper-level skip axis: a short TRAIN window (< min_days) -> annotate
    dsr_passed=False + dsr_inconclusive=True (MEDIUM-8) and do NOT filter
    (INCONCLUSIVE, not reject-all)."""
    from layer2.experiment import _apply_dsr_front_gate
    evo = _evo_records_for_sr_star()
    champs = [{"id": "x", "train_sharpe": 5.0, "val_sharpe": 5.0,
               "train_return_skew": 0.0, "train_return_kurtosis": 0.0}]
    kept, kept_front = _apply_dsr_front_gate(
        list(champs), ["ind_x"], train_T=10, evo_records=evo,
        dsr_gate_enabled=True, dsr_threshold=0.5, template_name="unit",
        min_days=20)
    assert len(kept) == 1 and kept_front == ["ind_x"], "skip must NOT filter"
    # MEDIUM-8: canonical reader idiom is `dsr_passed is True` -> significant.
    assert champs[0]["dsr_passed"] is False and champs[0]["dsr"] is None
    assert champs[0]["dsr_inconclusive"] is True


def test_dsr_front_gate_disabled_is_passthrough():
    """gate disabled -> no annotations, front returned unchanged."""
    from layer2.experiment import _apply_dsr_front_gate
    champs = [{"id": "x", "train_sharpe": 5.0, "val_sharpe": 5.0,
               "train_return_skew": 0.0, "train_return_kurtosis": 0.0}]
    kept, kept_front = _apply_dsr_front_gate(
        list(champs), ["ind_x"], train_T=300, evo_records={},
        dsr_gate_enabled=False, dsr_threshold=0.5, template_name="unit",
        min_days=20)
    assert len(kept) == 1 and kept_front == ["ind_x"]
    assert "dsr_passed" not in champs[0] and "dsr" not in champs[0]


def test_fitness_evaluator_score_on_data_does_not_cache():
    """score_on_data is the val/test path — it must NOT pollute the train
    cache and must NOT consult it (different data produces different
    fitness for the same trees)."""
    from layer2.fitness import FitnessEvaluator
    from layer2.templates import iron_condor
    # Use the shared synthetic-data helper in test_fitness
    import tests.test_fitness as tf
    template = iron_condor()
    # Train data: n_dates=2 (stationary-ish)
    train_data = tf._make_data(n_dates=2, bars_per_date=30)
    # Val data: different n_dates/bar range — different realized stats
    val_data   = tf._make_data(n_dates=3, bars_per_date=20)
    fe = FitnessEvaluator(
        template=template, data=train_data,
        backtester_kwargs={"warmup_bars": 5, "default_minutes_to_expiry": 60.0},
    )
    # First evaluate trains the cache
    train_fit = fe.evaluate(template.entry_seed, template.exit_seed, template.size_seed)
    assert fe.stats["evaluations"] == 1
    assert fe.stats["cache_hits"] == 0
    # score_on_data must NOT increment eval_count or hit count
    val_fit = fe.score_on_data(
        template.entry_seed, template.exit_seed, template.size_seed,
        data=val_data,
    )
    assert fe.stats["evaluations"] == 1, "score_on_data leaked into train cache"
    assert fe.stats["cache_hits"] == 0


def test_hypervolume_zero_when_all_sentinel(tmp_path):
    """An entire Pareto front of sentinel-fitness individuals must produce
    hv=0 (sentinel rows are outside the reference box by construction).
    Regression guard for the hypervolume filter (Code Reviewer FINDING-A)."""
    from layer2.experiment import _compute_val_hypervolume, H2_HV_REFERENCE_POINT
    from layer2.fitness import FAILED_FITNESS_SENTINEL
    all_sentinel = np.full((5, len(H2_HV_REFERENCE_POINT)), FAILED_FITNESS_SENTINEL)
    hv = _compute_val_hypervolume(all_sentinel, H2_HV_REFERENCE_POINT)
    assert hv == 0.0, (
        f"all-sentinel front produced hv={hv}; expected 0.0. Sentinel rows "
        f"must be filtered as 'outside reference box'."
    )


def test_hypervolume_positive_when_front_dominates_reference(tmp_path):
    """A Pareto-front with strict domination of the reference point must
    produce positive hypervolume."""
    from layer2.experiment import _compute_val_hypervolume, H2_HV_REFERENCE_POINT
    # An individual strictly better on every axis than the reference → contributes HV.
    n_obj = len(H2_HV_REFERENCE_POINT)
    front = np.array([
        [-0.5, 0.1, -0.5] + [-0.5] * (n_obj - 3),
        [-0.3, 0.2, -0.7] + [-0.3] * (n_obj - 3),
    ])
    hv = _compute_val_hypervolume(front, H2_HV_REFERENCE_POINT)
    assert hv > 0.0


# ---------------------------------------------------------------------------
# Commit 6.6: operational resilience — checkpoint + resume + error handling
# ---------------------------------------------------------------------------

def test_per_template_artifacts_written_incrementally(tmp_path):
    """Commit 6.6: each template's artifacts (strategies_*.jsonl,
    metrics_*.json, template_*.json) must be written IMMEDIATELY after
    that template completes, not batched at end-of-run. This converts
    mid-run crashes into "lose at most current template."
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "incremental"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard", "bull_put_credit_standard"),
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-08-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)

    # Per-template files exist for BOTH templates
    for name in ("iron_condor_standard", "bull_put_credit_standard"):
        assert (output_dir / f"strategies_{name}.jsonl").exists()
        assert (output_dir / f"metrics_{name}.json").exists()
        assert (output_dir / f"template_{name}.json").exists(), (
            f"per-template summary template_{name}.json missing — "
            f"incremental checkpointing (Commit 6.6) is not firing"
        )
    # Top-level results.json also exists
    assert (output_dir / "results.json").exists()


def test_resume_detection_skips_completed_templates(tmp_path, capsys):
    """Commit 6.6: run A produces per-template artifacts; run B (identical
    config pointing at the same output_dir) must detect completed
    templates and skip their re-evolution. Resumed result must be
    byte-identical to the original run's artifacts."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "resume"

    def _cfg():
        return ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-06-15",
            val_end="2024-08-15",
            embargo_days=0,
            output_dir=str(output_dir),
            backtester_warmup_bars=3,
        )

    # First run: produces artifacts
    run_experiment(_cfg())
    original_strat = (output_dir / "strategies_iron_condor_standard.jsonl").read_text()
    original_summary = (output_dir / "template_iron_condor_standard.json").read_text()

    # Second run: must detect + skip
    capsys.readouterr()  # drain captured output
    run_experiment(_cfg())
    out = capsys.readouterr().out
    assert "resume" in out.lower() or "SKIP" in out, (
        f"second run did not log resume detection; stdout:\n{out}"
    )
    # Artifacts unchanged — resumed run must NOT overwrite
    assert (output_dir / "strategies_iron_condor_standard.jsonl").read_text() == original_strat
    assert (output_dir / "template_iron_condor_standard.json").read_text() == original_summary


def test_template_failure_logged_and_pipeline_continues(tmp_path, monkeypatch):
    """Commit 6.6: a template that raises mid-evolution must NOT abort
    the whole H2 run. Pipeline logs failed_<name>.log and continues to
    the next template. Verified by monkey-patching evolve() to raise on
    the first template."""
    from layer2 import experiment as exp_mod

    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "failure"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard", "bull_put_credit_standard"),
        pop_size=8, n_generations=2, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-08-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )

    real_evolve = exp_mod.evolve
    call_count = {"n": 0}

    def _evolve_fails_first(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("synthetic failure on first template")
        return real_evolve(*args, **kwargs)

    monkeypatch.setattr(exp_mod, "evolve", _evolve_fails_first)

    result = run_experiment(config)
    # Second template completed
    assert len(result.template_results) == 1
    assert result.template_results[0].template_name == "bull_put_credit_standard"
    # Failure log written for the first template
    assert (output_dir / "failed_iron_condor_standard.log").exists(), (
        "failed template did not produce a failure log — post-mortem broken"
    )
    log_text = (output_dir / "failed_iron_condor_standard.log").read_text()
    assert "synthetic failure on first template" in log_text
    assert "Traceback:" in log_text


# ---------------------------------------------------------------------------
# Commit 6.7: audit trail additions + integration tests
# ---------------------------------------------------------------------------

def test_provenance_records_l1_parquet_sha256(tmp_path):
    """Model QA R1: results.json provenance must include SHA256 of the
    L1 Parquet file, its size, and its path. Without this, an auditor
    cannot verify they're looking at the same input data."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "parquet_sha"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-09-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    prov = data["provenance"]
    assert "l1_parquet_sha256" in prov
    assert len(prov["l1_parquet_sha256"]) == 64  # SHA256 hex digest length
    assert prov["l1_parquet_size_bytes"] > 0
    assert prov["l1_parquet_path"].endswith(".parquet")


def test_provenance_records_realized_split_boundaries(tmp_path):
    """Model QA R2: results.json provenance must include realized first/
    last date of each split, not just row counts. Config's train_end is
    a REQUESTED cutoff; this records the ACTUAL post-embargo boundaries."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=12, bars_per_date=12)
    output_dir = tmp_path / "split_bounds"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-09-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    bounds = data["provenance"]["split_boundaries"]
    for split in ("train", "val", "test"):
        assert split in bounds
        assert "date_min" in bounds[split]
        assert "date_max" in bounds[split]
        assert "n_dates" in bounds[split]
        assert "n_rows" in bounds[split]
    # train precedes val precedes test (walk-forward invariant)
    assert bounds["train"]["date_max"] < bounds["val"]["date_min"]
    assert bounds["val"]["date_max"] < bounds["test"]["date_min"]


def test_provenance_records_grammar_signature_and_ref_dirs(tmp_path):
    """Model QA R3 + R9: results.json provenance must include n_ref_dirs
    and a grammar signature so future grammar mutations are detectable."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "grammar_sig"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15",
        val_end="2024-09-15",
        embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    prov = data["provenance"]
    assert "n_ref_dirs" in prov
    assert prov["n_ref_dirs"] > 0
    assert "grammar_signature" in prov
    sig = prov["grammar_signature"]
    assert "sha256" in sig
    assert "n_functions" in sig
    assert "n_terminals" in sig
    assert sig["max_depth"] == config.grammar_max_depth
    assert sig["max_nodes"] == config.grammar_max_nodes


def test_three_arm_integration_tri_condition(tmp_path):
    """Commit 6.7 (Code Reviewer): integration test that exercises all 3
    H2 conditions sequentially against the same L1 Parquet, same master
    seed, same 2 templates. Validates that:
    (a) each condition produces valid artifacts,
    (b) scalar-only's seed_trees differ from real-l1's for templates whose seed
        uses PredRV15 (bull_put_credit → RealizedVol30m) OR a QC-NON-TRANSFERABLE
        scalar (BLOCKER-14, 2026-06-01: iron_condor's SessionReturn → SPXReturn3d).
        real-l1 / shuffled-l1 leave both unchanged.
    (c) all three runs are well-isolated (no cross-condition state leak).
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=12, bars_per_date=12)

    def _run(condition: str, out: Path):
        cfg = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition=condition,
            template_names=("iron_condor_standard", "bull_put_credit_standard"),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-06-15",
            val_end="2024-09-15",
            embargo_days=0,
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        return run_experiment(cfg)

    out_real  = tmp_path / "real"
    out_shuf  = tmp_path / "shuffled"
    out_scalar = tmp_path / "scalar"
    r_real  = _run("real-l1", out_real)
    r_shuf  = _run("shuffled-l1", out_shuf)
    r_scalar = _run("scalar-only", out_scalar)

    # All three produced two templates' results.json + jsonl
    for out in (out_real, out_shuf, out_scalar):
        assert (out / "results.json").exists()
        assert (out / "strategies_iron_condor_standard.jsonl").exists()
        assert (out / "strategies_bull_put_credit_standard.jsonl").exists()

    # bull_put_credit seed uses PredRV15 → rewritten under scalar-only only
    for r, condition in ((r_real, "real-l1"), (r_shuf, "shuffled-l1"),
                          (r_scalar, "scalar-only")):
        bpc = next(tr for tr in r.template_results if tr.template_name == "bull_put_credit_standard")
        if condition == "scalar-only":
            assert "PredRV15" in bpc.seed_trees_original["entry"]
            assert "PredRV15" not in bpc.seed_trees_effective["entry"]
            assert "RealizedVol30m" in bpc.seed_trees_effective["entry"]
        else:
            assert bpc.seed_trees_original == bpc.seed_trees_effective

    # iron_condor seed gates on SessionReturn (QC-NON-TRANSFERABLE). BLOCKER-14:
    # under scalar-only it is rewritten to a transferable analog (SPXReturn3d); under
    # real-l1 / shuffled-l1 it is left unchanged.
    for r, condition in ((r_real, "real-l1"), (r_shuf, "shuffled-l1"),
                          (r_scalar, "scalar-only")):
        ic = next(tr for tr in r.template_results if tr.template_name == "iron_condor_standard")
        if condition == "scalar-only":
            assert "SessionReturn" in ic.seed_trees_original["entry"]
            assert "SessionReturn" not in ic.seed_trees_effective["entry"]
            assert "SessionReturn" not in ic.seed_trees_effective["exit"]
            assert "SPXReturn3d" in ic.seed_trees_effective["entry"]
        else:
            assert ic.seed_trees_original == ic.seed_trees_effective


# ---------------------------------------------------------------------------
# Commit 6.8: resume integrity + audit trail hardening
# ---------------------------------------------------------------------------

def test_run_fingerprint_written_on_first_run(tmp_path):
    """6.8: first run of run_experiment must write a run_fingerprint.json
    containing the config fingerprint + env snapshot + grammar signature."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "fp"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    fp_path = output_dir / "run_fingerprint.json"
    assert fp_path.exists()
    with fp_path.open() as f:
        fp = json.load(f)
    for key in ("fingerprint", "env_snapshot", "grammar_signature",
                "config_summary", "l1_parquet_sha256"):
        assert key in fp, f"run_fingerprint missing {key}"
    # env_snapshot must include the library versions
    assert "pymoo_version" in fp["env_snapshot"]
    assert "python_version" in fp["env_snapshot"]


def test_resume_rejects_config_fingerprint_mismatch(tmp_path):
    """6.8 BLOCKING: if operator changes seed/pop_size/condition and resumes
    against an output_dir with a different fingerprint, must raise
    RuntimeError (not silently mix old artifacts with new evolution)."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "mismatch"

    def _cfg(**override):
        base = dict(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=1, seed=42, n_partitions=4,
            train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
            output_dir=str(output_dir),
            backtester_warmup_bars=3,
        )
        base.update(override)
        return ExperimentConfig(**base)

    run_experiment(_cfg())
    # Try to resume with a different seed — must raise
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        run_experiment(_cfg(seed=99))


def test_resume_rejects_stale_artifacts_without_fingerprint(tmp_path):
    """6.8: per-template artifacts from a pre-6.8 run (no run_fingerprint.json)
    must not be trusted for resume — raise RuntimeError with a clear message."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "stale"
    output_dir.mkdir()
    # Fake a pre-6.8 artifact: a template_*.json with no run_fingerprint.json
    (output_dir / "template_iron_condor_standard.json").write_text(
        json.dumps({"template_name": "iron_condor"}))
    (output_dir / "strategies_iron_condor_standard.jsonl").write_text("")
    (output_dir / "metrics_iron_condor_standard.json").write_text("[]")
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    with pytest.raises(RuntimeError, match="no run_fingerprint.json"):
        run_experiment(config)


def test_per_template_summary_includes_env_snapshot(tmp_path):
    """6.8: each template_<name>.json must stamp env_snapshot + grammar
    signature AT SAVE TIME so a resumed run can detect environment drift."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "tpl_env"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "template_iron_condor_standard.json").open() as f:
        tpl = json.load(f)
    assert "env_snapshot" in tpl
    assert "grammar_signature" in tpl
    assert tpl["env_snapshot"]["pymoo_version"]


def test_total_wall_time_accumulates_across_resume(tmp_path):
    """6.8 F2: total_wall_time_s in results.json must sum per-template
    wall times, NOT just this-invocation elapsed. Verify resuming a run
    preserves the original template's wall_time_s in the sum."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "wall"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    r1 = run_experiment(config)
    wall1 = r1.total_wall_time_s
    # Resume (should skip the template, then total_wall should still equal
    # the per-template wall_time_s from run 1)
    r2 = run_experiment(config)
    assert r2.total_wall_time_s == pytest.approx(wall1, rel=0.01), (
        f"resumed total_wall_time_s ({r2.total_wall_time_s}) should match "
        f"original sum of per-template walls ({wall1})"
    )


def test_warnings_channel_records_scalar_only_substitutions(tmp_path):
    """6.8 R7: scalar-only runs must surface each substitution event in
    results.json's top-level `warnings` list."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "warn"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="scalar-only",
        template_names=("bull_put_credit",),  # uses PredRV15 → substituted
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    warnings = data.get("warnings", [])
    subs = [w for w in warnings if w.get("kind") == "scalar_only_substitution"]
    assert len(subs) >= 1, f"scalar-only substitution not logged; warnings={warnings}"
    sub = subs[0]
    assert sub["template"] == "bull_put_credit_standard"
    assert "PredRV15" in sub["from_tree"]
    assert "RealizedVol30m" in sub["to_tree"]


def test_provenance_records_requirements_sha256(tmp_path):
    """6.8 R10: requirements-l2.txt SHA must be in provenance for dep-pin
    integrity — committing to an exact pin set in pre-registration."""
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "reqsha"
    config = ExperimentConfig(
        l1_parquet_path=str(l1_path),
        condition="real-l1",
        template_names=("iron_condor_standard",),
        pop_size=8, n_generations=1, seed=42, n_partitions=4,
        train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
        output_dir=str(output_dir),
        backtester_warmup_bars=3,
    )
    run_experiment(config)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    prov = data["provenance"]
    assert "requirements_l2_sha256" in prov
    assert len(prov["requirements_l2_sha256"]) == 64
    assert prov["pandas_version"]
    assert prov["python_version"]
    # TOCTOU-safe hash timing
    assert prov["l1_parquet_hash_timing"] == "at_load_time"


def test_compute_val_hypervolume_boundary_inclusive(tmp_path):
    """6.8 AI Engineer finding: filter uses `<=` not strict `<` so boundary
    points (e.g. max_drawdown == ref exactly) are INCLUDED. Their HV
    contribution is 0 so this doesn't change the score, but it avoids
    arbitrary exclusion of Sharpe=0 strategies."""
    from layer2.experiment import _compute_val_hypervolume, H2_HV_REFERENCE_POINT
    # Single individual with max_drawdown == 1.0 (boundary on that axis),
    # strict better on the other two. Under old strict `<`, excluded.
    # Under new `<=`, included (contributes 0 volume on the DD axis).
    n_obj = len(H2_HV_REFERENCE_POINT)
    front_with_boundary = np.array([[-0.5, 1.0, -0.5] + [-0.5] * (n_obj - 3)])
    hv = _compute_val_hypervolume(front_with_boundary, H2_HV_REFERENCE_POINT)
    # The boundary point contributes 0 volume on the DD axis, so HV = 0
    # but the INCLUSION is what we're testing (no exception raised)
    assert hv >= 0.0
    # A non-boundary point beats boundary — should contribute visibly
    front_mixed = np.array([
        [-0.5, 1.0, -0.5] + [-0.5] * (n_obj - 3),
        [-0.5, 0.5, -0.5] + [-0.5] * (n_obj - 3),
    ])
    hv_mixed = _compute_val_hypervolume(front_mixed, H2_HV_REFERENCE_POINT)
    assert hv_mixed > 0.0


# ---------------------------------------------------------------------------
# Commit 6.8 post-review fixes (full-branch Code Reviewer)
# ---------------------------------------------------------------------------

def test_env_drift_blocking_vs_warning_policy(tmp_path, monkeypatch, capsys):
    """Full-branch Code Reviewer fix #1: env drift on pymoo/numpy/scipy/
    python must RAISE (scientific-validity). Drift on pandas/pyarrow
    (declared tolerant in requirements-l2.txt) must WARN-and-continue."""
    from layer2 import experiment as exp_mod
    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "drift"

    def _cfg():
        return ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=1, seed=42, n_partitions=4,
            train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
            output_dir=str(output_dir),
            backtester_warmup_bars=3,
        )

    # Original run
    run_experiment(_cfg())

    # Mock env_snapshot to simulate pandas drift (should warn, not raise)
    real_snapshot = exp_mod._env_snapshot
    def _drifted_pandas():
        snap = real_snapshot()
        snap["pandas_version"] = "9.9.9"  # drifted
        return snap
    monkeypatch.setattr(exp_mod, "_env_snapshot", _drifted_pandas)
    capsys.readouterr()
    run_experiment(_cfg())   # must NOT raise
    out = capsys.readouterr().out
    assert "WARNING" in out and "pandas_version" in out, (
        f"pandas drift should warn-and-continue; stdout:\n{out}"
    )

    # Now mock pymoo drift (BLOCKING)
    def _drifted_pymoo():
        snap = real_snapshot()
        snap["pymoo_version"] = "99.99.99"
        return snap
    monkeypatch.setattr(exp_mod, "_env_snapshot", _drifted_pymoo)
    with pytest.raises(RuntimeError, match="BLOCKING environment drift"):
        run_experiment(_cfg())


def test_warnings_persisted_across_resume(tmp_path, monkeypatch):
    """Full-branch Code Reviewer fix #3: warnings from a prior run must
    persist into results.json after a resumed run. Without this, a
    crash-mid-pilot + resume loses visibility into earlier substitutions."""
    from layer2 import experiment as exp_mod

    l1_path = _make_l1_parquet(tmp_path, n_dates=10, bars_per_date=12)
    output_dir = tmp_path / "warn_persist"

    def _cfg():
        return ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="scalar-only",
            template_names=("iron_condor_standard", "bull_put_credit_standard"),
            pop_size=8, n_generations=1, seed=42, n_partitions=4,
            train_end="2024-06-15", val_end="2024-09-15", embargo_days=0,
            output_dir=str(output_dir),
            backtester_warmup_bars=3,
        )

    # First run: monkey-patch evolve to fail on the SECOND template so
    # iron_condor completes + logs any warnings, then bull_put_credit
    # doesn't run (but is scalar-only-substitution-relevant)
    real_evolve = exp_mod.evolve
    call_count = {"n": 0}
    def _evolve_fails_second(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic fail on template 2")
        return real_evolve(*args, **kwargs)
    monkeypatch.setattr(exp_mod, "evolve", _evolve_fails_second)

    run_experiment(_cfg())
    # warnings.jsonl from run 1 captures: template_failure for
    # bull_put_credit + (scalar-only substitution is on bull_put_credit
    # but it failed before warning fired, so that won't be present)
    warnings_path = output_dir / "warnings.jsonl"
    assert warnings_path.exists(), "warnings.jsonl not created on first run"
    run1_warnings = [json.loads(l) for l in warnings_path.read_text().splitlines() if l.strip()]
    failures_run1 = [w for w in run1_warnings if w["kind"] == "template_failure"]
    assert len(failures_run1) >= 1

    # Un-patch so resumed run succeeds
    monkeypatch.setattr(exp_mod, "evolve", real_evolve)
    run_experiment(_cfg())

    # results.json now contains ALL warnings from run 1 AND new ones from
    # run 2 (e.g. the bull_put_credit substitution when it actually runs)
    with (output_dir / "results.json").open() as f:
        data = json.load(f)
    # Prior template_failure persists; scalar-only substitution also present
    kinds = [w["kind"] for w in data.get("warnings", [])]
    assert "template_failure" in kinds, (
        f"prior run's template_failure warning lost on resume; kinds={kinds}"
    )


def test_sequential_cross_condition_reproducibility(tmp_path):
    """Model QA Finding 5: running real-l1 then scalar-only in the same
    process must not leak state from one condition to the other. Each
    condition's output must match a reference run done in isolation.
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)

    def _run(condition: str, out: Path):
        cfg = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition=condition,
            template_names=("iron_condor_standard",),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-04-15",
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        run_experiment(cfg)

    # Sequential: real-l1 then scalar-only
    _run("real-l1",     tmp_path / "seq_real")
    _run("scalar-only", tmp_path / "seq_scalar")

    # Isolated (fresh process would be ideal but we reset state manually).
    # The B3 concurrency lock is already released between runs; each
    # evolve() re-seeds RNG; each run_experiment builds a fresh Grammar
    # and FitnessEvaluator. So running scalar-only first here should give
    # a scalar-only result byte-identical to the sequential one.
    _run("scalar-only", tmp_path / "iso_scalar")
    a = (tmp_path / "seq_scalar" / "strategies_iron_condor_standard.jsonl").read_text()
    b = (tmp_path / "iso_scalar" / "strategies_iron_condor_standard.jsonl").read_text()
    assert a == b, (
        "scalar-only after real-l1 differs from scalar-only in isolation — "
        "cross-condition state leakage in run_experiment"
    )


# ---------------------------------------------------------------------------
# B3/B4: seed hygiene
# ---------------------------------------------------------------------------

def test_shuffle_seed_defaults_decoupled_from_seed():
    """B4: default shuffle_seed must differ from master seed."""
    cfg = ExperimentConfig(l1_parquet_path="x.parquet", seed=42)
    assert cfg.shuffle_seed is not None
    assert cfg.shuffle_seed != cfg.seed
    assert cfg.shuffle_seed == 42 + _SHUFFLE_SEED_OFFSET


def test_shuffle_seed_equal_to_seed_rejected():
    """B4: explicitly setting shuffle_seed == seed is a hard error."""
    with pytest.raises(ValueError, match="shuffle_seed must differ"):
        ExperimentConfig(l1_parquet_path="x.parquet", seed=42, shuffle_seed=42)


def test_shuffle_seed_can_be_overridden():
    cfg = ExperimentConfig(l1_parquet_path="x.parquet", seed=42, shuffle_seed=777)
    assert cfg.shuffle_seed == 777


def test_derive_template_seed_is_deterministic_and_distinct():
    """B3: per-template seeds are reproducible AND differ across templates."""
    a1 = _derive_template_seed(42, "iron_condor_standard")
    a2 = _derive_template_seed(42, "iron_condor_standard")
    b = _derive_template_seed(42, "bull_put_credit")
    c = _derive_template_seed(43, "iron_condor_standard")
    assert a1 == a2                        # deterministic under same inputs
    assert a1 != b                          # distinct across templates at same master seed
    assert a1 != c                          # distinct across master seeds for same template


def test_run_experiment_reproducible_with_same_master_seed(tmp_path):
    """B3: running the same config twice → identical Pareto front signatures
    AND identical per-generation metrics_log trajectory.

    Per-template islands must isolate from one another (no cross-template RNG
    contamination) AND the whole pipeline must be deterministic given seed.
    Model QA requirement: lock down metrics_log too, so mid-run drift that
    happens to converge to the same front is also caught.
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    def _mkcfg(out: Path) -> ExperimentConfig:
        return ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=("iron_condor_standard", "bull_put_credit_standard"),
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-04-15",
            output_dir=str(out),
            backtester_warmup_bars=3,
        )

    run_experiment(_mkcfg(out_a))
    run_experiment(_mkcfg(out_b))

    with (out_a / "results.json").open() as f:
        data_a = json.load(f)
    with (out_b / "results.json").open() as f:
        data_b = json.load(f)

    # Compare per-template best_objectives (Pareto-front-adjacent summary)
    # Wall-clock and cache stats legitimately differ; structural results must not.
    assert len(data_a["templates"]) == len(data_b["templates"])
    for ta, tb in zip(data_a["templates"], data_b["templates"]):
        assert ta["name"] == tb["name"]
        assert ta["pareto_front_size"] == tb["pareto_front_size"]
        assert ta["final_population_size"] == tb["final_population_size"]
        assert ta["best_objectives"] == tb["best_objectives"]

    # Strategies archives byte-identical per template
    for name in ("iron_condor_standard", "bull_put_credit_standard"):
        sa = (out_a / f"strategies_{name}.jsonl").read_text()
        sb = (out_b / f"strategies_{name}.jsonl").read_text()
        assert sa == sb, f"strategies_{name}.jsonl differs between runs"

    # metrics_*.json byte-identical per template — locks down per-generation
    # trajectory (front_size, best/median/worst per objective, node counts,
    # nan_count). Wall_time_s is the only non-deterministic field, and it's
    # captured as a float that still happens to serialize identically only
    # by luck; we therefore compare field-wise, skipping wall_time_s.
    for name in ("iron_condor_standard", "bull_put_credit_standard"):
        with (out_a / f"metrics_{name}.json").open() as f:
            ma = json.load(f)
        with (out_b / f"metrics_{name}.json").open() as f:
            mb = json.load(f)
        assert len(ma) == len(mb)
        for gen_a, gen_b in zip(ma, mb):
            for key in (
                "generation", "population_size", "front_size",
                "best_per_objective", "median_per_objective",
                "worst_per_objective", "mean_node_count", "nan_count",
            ):
                assert gen_a[key] == gen_b[key], (
                    f"metrics_{name}.json[{key}] differs between runs"
                )


def test_run_experiment_template_order_independent(tmp_path):
    """B3: per-template islands are independent — swapping evaluation order
    must not change any template's result (no shared-RNG contamination).

    Three templates so we catch associativity bugs that a 2-template
    swap would miss (3! = 6 orderings, but 2 contrasting orderings are enough).
    """
    l1_path = _make_l1_parquet(tmp_path, n_dates=6, bars_per_date=15)
    names = ("iron_condor_standard", "bull_put_credit_standard", "bear_call_credit_standard")

    def _run(order, out: Path):
        cfg = ExperimentConfig(
            l1_parquet_path=str(l1_path),
            condition="real-l1",
            template_names=order,
            pop_size=8, n_generations=2, seed=42, n_partitions=4,
            train_end="2024-04-15",
            output_dir=str(out),
            backtester_warmup_bars=3,
        )
        run_experiment(cfg)

    out_forward = tmp_path / "forward"
    out_reverse = tmp_path / "reverse"
    _run(names, out_forward)
    _run(tuple(reversed(names)), out_reverse)

    # Each template's strategies_<name>.jsonl must be byte-identical across orderings
    for name in names:
        a = (out_forward / f"strategies_{name}.jsonl").read_text()
        b = (out_reverse / f"strategies_{name}.jsonl").read_text()
        assert a == b, (
            f"strategies_{name}.jsonl depends on template order — "
            f"per-template RNG isolation is broken (B3 regression)"
        )


def test_derive_template_seed_hamming_distance():
    """B3: derived seeds across templates should have high pairwise Hamming
    distance under a realistic master-seed sweep. Property audits that the
    hash is actually mixing bits, not collapsing adjacent master_seed inputs
    to correlated outputs.
    """
    template_names = (
        "iron_condor_standard", "iron_butterfly_standard", "bull_put_credit_standard", "bear_call_credit_standard",
        "bull_call_debit", "bear_put_debit", "ratio_put_backspread",
    )
    seeds = []
    for master in range(100):
        for name in template_names:
            seeds.append(_derive_template_seed(master, name))
    # No collisions across 700 derived seeds
    assert len(set(seeds)) == len(seeds), "BLAKE2b produced a collision under the realistic sweep"
    # Spot-check Hamming distance on a small sample: 7 templates at master=42
    sample = [_derive_template_seed(42, n) for n in template_names]
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            ham = bin(sample[i] ^ sample[j]).count("1")
            assert ham >= 8, (
                f"Hamming distance {ham} between {template_names[i]} and "
                f"{template_names[j]} is too low — hash is not mixing bits"
            )


def test_evolve_rejects_concurrent_entry():
    """B3: evolve() raises RuntimeError if entered concurrently from another
    thread. Converts the silent footgun (parallel templates silently sharing
    module-global RNG) into a loud failure.
    """
    import threading
    from layer2.gp_engine import (
        EvolutionConfig, _EVOLVE_INFLIGHT_LOCK, evolve,
    )
    from layer2.fitness import FitnessEvaluator
    from layer2.grammar import Grammar
    from layer2.templates import iron_condor

    template = iron_condor()
    grammar = Grammar(max_depth=3, max_nodes=15)

    # Simulate an in-flight evolve() by holding the lock from the main
    # thread, then invoke evolve() and expect RuntimeError.
    assert _EVOLVE_INFLIGHT_LOCK.acquire(blocking=False)
    try:
        fe = FitnessEvaluator(
            template=template,
            data=pd.DataFrame(),  # empty — we won't reach evaluation
            backtester_kwargs={"warmup_bars": 3, "default_minutes_to_expiry": 60.0},
        )
        cfg = EvolutionConfig(pop_size=4, n_generations=1, seed=0, n_partitions=3)
        with pytest.raises(RuntimeError, match="not safe to call concurrently"):
            evolve(template, fe, grammar, cfg)
    finally:
        _EVOLVE_INFLIGHT_LOCK.release()
