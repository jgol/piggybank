"""L2 experiment orchestration: load L1 Parquet → run NSGA-III evolution →
report Pareto front + metrics. CLI + Python entry points.

CLI usage:
    python -m layer2.experiment run \\
        --l1-parquet path/to/l1_output.parquet \\
        --condition real-l1 \\
        --templates iron_condor,bull_put_credit \\
        --pop-size 256 --generations 100 --seed 42 \\
        --train-end 2024-09-30 --val-end 2025-01-31 \\
        --output results/run_001/

Programmatic usage:
    from layer2.experiment import run_experiment, ExperimentConfig
    result = run_experiment(ExperimentConfig(...))
"""
from __future__ import annotations

# Cap BLAS threads at process startup — must happen BEFORE numpy import.
# Prevents GP evolution from saturating all CPU cores on shared machines.
# Shell-level env vars (OMP_NUM_THREADS=2 nohup ...) don't reliably
# propagate through macOS zsh + nohup chains.
import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_k, "4")
del _k

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.fitness import (
    DEFAULT_OBJECTIVES, FitnessEvaluator, VectorizedFitnessEvaluator,
)
from layer2.gp_engine import (
    EvolutionConfig, GenerationMetrics, Individual, evolve, pareto_front,
)
from layer2.grammar import (
    Grammar, SCALAR_ONLY_FUNCTIONS, PROBES_ONLY_FUNCTIONS, EMB_ONLY_FUNCTIONS,
    assert_grammar_matches_preregistration,
    build_scalar_only_terminal_set, build_probes_only_terminal_set,
    build_emb_only_terminal_set, to_str, canonical_key,
    substitute_stripped_terminals_for_scalar_only,
    substitute_for_probes_pure,
    substitute_for_emb_pure,
)
from layer2.io import load_l1_parquet, load_minute_parquet, split_by_date
from layer2.shuffle import shuffle_l1
from layer2.templates import (
    Template, all_templates, base_template_by_name, template_by_name,
)


# ---------------------------------------------------------------------------
# Seed derivation (B3/B4 fix)
# ---------------------------------------------------------------------------

# Offset between master seed and shuffle seed when the caller does not set
# shuffle_seed explicitly. Keeps the shuffled-L1 permutation RNG stream
# decoupled from the GP RNG stream under the same master seed.
_SHUFFLE_SEED_OFFSET = 1_000_000


def _env_snapshot() -> Dict[str, str]:
    """Snapshot of library versions + Python version at call time.
    Used both in results.json provenance AND per-template summaries
    (so resumed runs can detect environment drift — Model QA finding)."""
    import sys
    import pymoo
    import numpy as _np
    import scipy as _sp
    snap = {
        "python_version": sys.version.split()[0],
        "pymoo_version":  pymoo.__version__,
        "numpy_version":  _np.__version__,
        "scipy_version":  _sp.__version__,
        "pandas_version": pd.__version__,
    }
    try:
        import pyarrow
        snap["pyarrow_version"] = pyarrow.__version__
    except ImportError:
        snap["pyarrow_version"] = None
    return snap


# Environment keys whose drift BLOCKS a resumed run (scientific-validity
# critical — affects evolution math, fitness computation, random-stream
# semantics). Patch/minor drift on these invalidates the pilot.
_ENV_DRIFT_BLOCKING_KEYS: Tuple[str, ...] = (
    "pymoo_version",    # niching / non-dominated sort semantics
    "numpy_version",    # random stream (default_rng) semantics
    "scipy_version",    # pymoo uses scipy in hypervolume calc
    "python_version",   # dict ordering / f-string semantics historically drifted
)

# Environment keys whose drift generates a WARNING but does not block.
# requirements-l2.txt tolerates minor-version drift on these (pyarrow
# is declared `>=14.0.0`) because they affect I/O format stability but
# not the evolution math.
_ENV_DRIFT_WARNING_KEYS: Tuple[str, ...] = (
    "pandas_version",
    "pyarrow_version",
)


# Config fields that define the EXPERIMENT IDENTITY. Changes to any of these
# make artifacts from a prior run incompatible for resume. Fields NOT in this
# list (e.g. output_dir, l1_parquet_path — which we check separately via SHA)
# can change freely between runs without invalidating resume.
_FINGERPRINT_FIELDS: Tuple[str, ...] = (
    "condition", "seed", "shuffle_seed", "pop_size", "n_generations",
    "n_partitions", "crossover_rate", "mutation_rate", "seed_fraction",
    "train_end", "val_end", "embargo_days",
    "train_start", "test_end",  # v8: OOR-pilot date range hooks
    "backtester_warmup_bars", "backtester_minutes_to_expiry",
    "grammar_max_depth", "grammar_max_nodes", "min_trades",
    "cost_multiplier",  # Level B: cost sensitivity sweep
    "use_vectorized",  # vectorized vs per-bar evaluator produces different results
    "level_b",  # Level B: base templates with delta_tree vs V1 fixed templates
    "per_fold_seeds",  # P1-A: per-fold seed re-derivation changes the seeded population
    "probe_bundle_sha256",  # per-fold probe refit: content hash, not path
    "regime_gate_enabled",  # per-regime Sharpe gate changes fitness values
    "regime_gate_margin_k",  # noise-aware per-regime gate margin (k·combined SE)
    "norm_mode", "trailing_window",  # plan #4: trailing-rolling vs expanding norm
)


def _config_fingerprint(config: ExperimentConfig, l1_sha256: str) -> str:
    """Deterministic hash of experiment identity for resume safety.

    Includes all _FINGERPRINT_FIELDS from the config PLUS the L1 Parquet's
    SHA256. Two runs with matching fingerprints produce byte-identical
    artifacts (modulo platform float drift); mismatching fingerprints
    mean the old artifacts are not valid for resume.
    """
    import hashlib as _hashlib
    parts = [f"{fld}={getattr(config, fld)!r}" for fld in _FINGERPRINT_FIELDS]
    parts.append(f"l1_sha256={l1_sha256}")
    return _hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _derive_template_seed(base_seed: int, template_name: str) -> int:
    """Deterministic per-template seed from a master seed + template name.

    Uses BLAKE2b truncated to 32 bits so the result fits a uint32 (np/random
    seed range). Sequential template evolution under the same master_seed
    produces distinct but reproducible RNG streams per template — a template's
    evolution does not depend on which other templates ran before it.
    """
    key = f"{base_seed}|{template_name}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big")


# ---------------------------------------------------------------------------
# H2 primary outcome — hypervolume on validation split (Commit 6.5)
# ---------------------------------------------------------------------------

# Fixed reference point for hypervolume computation. Pre-declared as a
# 3 objectives: neg_sharpe, neg_sortino, neg_trade_count_score.
# 0: neg_sharpe — ref=2.0 (admits Sharpe down to -2.0)
# 1: neg_sortino — ref=2.0 (admits Sortino down to -2.0)
# 2: neg_trade_count_score — ref=-0.3 (requires tc_score > 0.3)
# Drawdown is a SOFT PENALTY on neg_sharpe/neg_sortino (fitness.py), NOT a 4th
# objective — so the hypervolume stays 3-D and no drawdown attractor is created.
H2_HV_REFERENCE_POINT: Tuple[float, ...] = (2.0, 2.0, -0.3)

assert len(H2_HV_REFERENCE_POINT) == len(DEFAULT_OBJECTIVES), (
    f"HV ref point dim ({len(H2_HV_REFERENCE_POINT)}) != "
    f"objectives dim ({len(DEFAULT_OBJECTIVES)})"
)


def _compute_val_hypervolume(front_val_fitness: np.ndarray,
                             ref_point: Tuple[float, ...]) -> float:
    """Hypervolume dominated by the front's validation fitness under the
    fixed reference point. Returns 0.0 if the front is empty.

    front_val_fitness: (n_front, n_obj) array of val-split fitness vectors.

    v4 amendment: switch from "filter out rows with any axis > ref" to
    "clip each row to ref on every axis". Motivation — a strategy that
    is inside the box on 2/3 objectives but blown up on one axis (e.g.
    `neg_sharpe = 50` from a single catastrophic leg) previously was
    dropped entirely from the HV; the arm then lost credit for the
    in-box axes. Clipping projects such rows onto the box boundary:
    axes outside the box contribute 0 volume (boundary = 0), axes
    inside contribute their real dominated volume. Sentinel rows
    (1e6 everywhere) clip to the ref itself and contribute 0 volume
    — equivalent to exclusion. No information is lost; partial signal
    is recovered.

    Commit 6.8 convention (retained): boundary inclusions use `<=`. The
    clipped path implements this by construction.
    """
    from pymoo.indicators.hv import HV
    if front_val_fitness.size == 0:
        return 0.0
    ref = np.asarray(ref_point, dtype=np.float64)
    F_clipped = np.minimum(front_val_fitness, ref)
    hv = HV(ref_point=ref)
    return float(hv(F_clipped))


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single GP-run configuration. One condition × one master seed.

    Conditions (H2 three-arm design):
      - 'real-l1':     full grammar (typed vectors + probes + raw scalars)
      - 'shuffled-l1': full grammar, L1-derived columns permuted (null control)
      - 'scalar-only': grammar with EMB_* terminals + probe scalars removed;
                       only raw bypass scalars + primitives remain. Distinguishes
                       "typed embeddings add value" from "L1 has any signal".

    Seed model (B3/B4 fix):
      - `seed` is the MASTER seed. All other RNG streams derive from it unless
        explicitly overridden:
          * per-template GP seed = hash(seed, template_name)  (decouples islands)
          * shuffle seed          = seed + 1_000_000 if shuffle_seed is None
      - Set `shuffle_seed` explicitly only when you want a shuffle permutation
        independent of the master seed (e.g., multiple shuffle replicates with
        the same GP seed, or a shuffle control across seeds).
    """
    l1_parquet_path: str
    condition: str = "real-l1"            # 'real-l1', 'shuffled-l1', 'scalar-only', 'probes-only', 'emb-only'
    template_names: Tuple[str, ...] = ()  # empty = all templates
    pop_size: int = 256
    n_generations: int = 100                 # CLI default 100; raised from 75 to match nursery ramp (40% of gens)
    seed: int = 42                         # Master seed (GP + shuffle derive from this)
    shuffle_seed: Optional[int] = None     # If None: derived as seed + 1_000_000
    n_partitions: int = 6                  # NSGA-III Das-Dennis partitions.
    # With 3 objectives, n_part=6 gives C(8,2)=28 ref dirs → 256/28=9.1
    # per niche (strong within-niche selection pressure).
    # NOTE: CLI default is --n-partitions=12 (overrides this default).
    crossover_rate: float = 0.7
    # v8: 0.15 → 0.25. 3-seed smoke's C6 FAIL (51/72 template-runs with
    # >60% duplicate Pareto trees) attributed to insufficient exploration
    # pressure. 0.25 is mid-range of GP-literature typical.
    mutation_rate: float = 0.25
    # seed_fraction=0.15: 15% of population seeded with hand-crafted theta-decay
    # entry conditions to bootstrap evolution past the cold-start catastrophe.
    # At 0.0, 91% of random individuals produce sentinel fitness (too few trades),
    # destroying NSGA-III selection pressure for ~50 generations.
    # Ablation fairness: each condition has per-condition seed substitution
    # (grammar.py substitute_for_*) that maps unavailable terminals to
    # semantically appropriate analogs from that condition's terminal set.
    # : Condition C now has 7 REAL terminals (3 structural + 4 market-
    # structure), preventing seed degeneration to SessionReturn-only.
    seed_fraction: float = 0.15
    train_end: str = "2024-09-30"
    val_end: Optional[str] = "2025-01-31"
    # v8 OOR support: optional train_start floor + test_end ceiling.
    # Defaults preserve v6/v8 main-pilot behavior (use earliest date in
    # parquet for train_start; use latest date for test_end). OOR bear
    # pilot overrides both.
    train_start: Optional[str] = None
    test_end: Optional[str] = None
    embargo_days: int = 5
    output_dir: Optional[str] = None       # if set, write results.json + metrics.parquet
    backtester_warmup_bars: int = 30  # Must equal max(_LAG_CHOICES)=30; Lag/Delta/CrossAbove produce zero-fill artifacts for bars < warmup
    backtester_minutes_to_expiry: float = 60.0
    grammar_max_depth: int = 5  # raised from 4: shallow trees cause crossover
    # to fail >70% of the time (no valid swap points under depth+node caps),
    # degenerating to parent cloning. Depth 5 gives more recombination sites.
    grammar_max_nodes: int = 15  # raised from 12: depth 5 + 12 nodes = only
    # linear chains. 15 allows AND(cond1, AND(cond2, cond3)) patterns needed
    # for regime-conditional entries. Matches Level B default.
    # B2 gate threshold — strategies with fewer trades than this receive
    # FAILED_FITNESS_SENTINEL on every objective. Raised to 20 (2026-05-09)
    # G12 fix: lowered from 20 to 10. Selective strategies (trading 30% of days)
    # produce ~12 trades per 125-day fold — critical for 0DTE where abstaining
    # SE(Sharpe) ≈ 1/√N → at N=10 is ±0.32 (noise dominates), at N=30 is
    # ±0.18 (2× improvement). Raised from 10 to 30 after Condition A showed
    # systematic train→val collapse: strategies with 10-20 trades fit noise.
    # BCC fold 2 (the only strong OOS result) had 30-38 trades per strategy.
    # 2026-06-02 (user decision (b)): LOWERED 30→20 so genuinely SELECTIVE entry
    # signals (~0.05 trades/day, the GP's whole point) survive evolution instead
    # of being sentinel'd for under-trading. 20 is a deliberate compromise — ABOVE
    # the 10-20 noise zone the comment warns about, BELOW the 30-38 good-result
    # range — and the DSR + g3_passed annotations (now annotate-don't-destroy, not
    # filters) still flag the high-trade-count / significant subset for confirmatory
    # use. Trades the noise-control margin for selectivity reach; revisit if the
    # front fills with low-trade noise-fits.
    min_trades: int = 20
    # Level B: cost sensitivity sweep. Scales all entry/exit costs by this
    # factor. 1.0 = baseline, 1.5 = 150% costs (stress test).
    cost_multiplier: float = 1.0
    # Use vectorized evaluator (evaluator_vectorized.py) instead of the
    # per-bar MultiLegOptionsBacktester. Required for 1-minute resolution
    # (357K bars would be too slow with the recursive evaluator).
    # 2026-05-09: auto-upgraded to True for multi-leg templates because the
    # non-vectorized path uses stale Brenner-Subrahmanyam pricing (no skew,
    # no stop-loss, no credit haircut, no calibrated costs). Results from
    # the non-vectorized path are not comparable to the vectorized path.
    # Level B: use 5 base templates with delta_tree (4th GP tree for
    # continuous delta exploration). When True, template_names are resolved
    # via base_template_by_name() which returns templates with delta_seed,
    # enabling the GP to grow delta_tree during init.
    level_b: bool = False
    # P1-A: re-derive Level-B base-template entry seeds per fold on TRAIN data only
    # (the hard-coded seeds were grid-searched on the full dataset incl. test
    # windows -> look-ahead leak). Off by default (backward-compatible); set True
    # for the fresh H2 walk-forward run.
    per_fold_seeds: bool = False
    use_vectorized: bool = False
    # plan #4 (docs/trailing_rolling_normalization_spec_2026_06_03.md): the
    # normalization window for LEVEL terminals (ATM_IV/VIX/RV/spread levels).
    # "expanding" = per-fold robust median/IQR over the WHOLE train span (default,
    # backward-compatible). "trailing_rolling" = causal per-day trailing robust stats
    # so the IV-gate fires ~uniformly across regimes (98%/2022 + 2%/2024 → ~23% both),
    # fixing the multi-regime feasibility collapse. Non-level terminals + the warmup
    # region keep the expanding stats either way.
    norm_mode: str = "expanding"
    trailing_window: int = 20  # trading days; W-sweep knee (spec §4.2), MEASURED
    # Minimum training rows for vectorized evaluator (hardening #8).
    # Set to 0 in tests with tiny synthetic data.
    min_train_rows: int = 1000
    # Per-fold probe refit: path to probe_fitting_bundle.npz ( fix).
    # When set, probes are refitted per walk-forward fold using only that
    # fold's training data, eliminating probe look-ahead leak. Required
    # for Conditions B (probes-only), D (real-l1), and shuffled-l1 with
    # early folds.
    probe_bundle_path: Optional[str] = None
    # Content hash of the probe bundle file (computed at load time).
    # Used in fingerprint instead of path so renames don't break resume.
    probe_bundle_sha256: Optional[str] = None
    # Task #34: inner validation during evolution. When True, evaluates
    # the Pareto front on validation data every 10 generations to monitor
    # overfitting. Val fitness is OBSERVATION ONLY — not used for selection.
    inner_val_enabled: bool = True
    # Per-regime noise-aware gate (Ang & Bekaert 2002). When True, strategies
    # whose worst-regime Sharpe falls below -0.5 (with >= 10 trades in
    # that regime) receive FAILED_FITNESS_SENTINEL. Prevents strategies
    # from surviving by being profitable in one regime and catastrophically
    # losing in another.
    regime_gate_enabled: bool = True
    # Noise-aware per-regime gate margin (2026-06-04). Reject a strategy in a
    # regime only if its trade-level Sharpe is worse than the random-entry bar
    # by MORE than regime_gate_margin_k × the combined Lo(2002) sampling SE.
    # k=0 = old hard threshold (decided within sampling noise → spurious
    # iron_condor/iron_butterfly mid-fold collapses). See memory
    # project_per_regime_gate_noise_2026_06_04.
    regime_gate_margin_k: float = 1.0
    # P1-B (REDESIGNED 2026-05-30): Deflated-Sharpe-Ratio FILTER gate at
    # Pareto-front champion selection. SR*_N is computed ONCE from the WHOLE
    # evolution's distinct trials: N = count of distinct individuals evaluated
    # across all generations (deduped by canonical_key), V = variance of their
    # DAILY training Sharpes (over members with n_days >= min_days). Each
    # champion's TRAINING Sharpe is deflated against SR* (Bailey & Lopez de
    # Prado 2014) and champions failing DSR < dsr_threshold are REMOVED from the
    # returned front (this gate FILTERS — it is not annotate-only). Deflating the
    # TRAINING Sharpe (the quantity NSGA-III selected on, hence the one inflated
    # by the N-trial search) against the training-trial SR* keeps the units
    # consistent (in-sample on both sides); OOS robustness is separately enforced
    # by the cross-fold G1/G10 survival gates. Using evolution-level (N, V)
    # instead of the final front's own variance kills the variance-inversion bug
    # (an overfit spike no longer inflates and clears its own bar). Does NOT
    # change DEFAULT_OBJECTIVES, per-individual fitness, or result.psr. Pinnable
    # in the fresh pre-reg; reversible (dsr_gate_enabled).
    dsr_gate_enabled: bool = True
    # At τ=0.5 the gate is PURE deflation: a champion passes iff its
    # (de-annualized) TRAINING Sharpe beats the multiple-testing-deflated SR*
    # (DSR=0.5 ⟺ SR=SR*, T-invariant). For realistic (N≈3k-15k distinct trials,
    # daily-Sharpe std ≈0.75 ann → V≈0.0024) this implies an ANNUALIZED
    # champion-Sharpe bar of ~2.7-3.0 — a genuinely strong, defensible champion,
    # set by (N, V) not by τ. Because the deflated quantity is now the TRAINING
    # Sharpe, T is the (large) training window, so the DSR sigmoid is steep and
    # τ moves the bar less than over the short val fold (but is NOT τ-invariant —
    # higher τ adds a real margin). See docs/methodology . Pinnable in the
    # fresh pre-reg.
    # PILOT DEFAULT 0.01 (2026-05-31, user choice, was 0.5): near pass-through —
    # keeps every champion except those whose train Sharpe is significantly (99%)
    # BELOW SR*, annotating DSR on each, so a discovery pilot SEES the full front
    # instead of risking an empty one. τ=0.5 (pure deflation) remains the
    # CONFIRMATORY value to pin in the fresh pre-reg. ⚠ The gate deflates PROXY
    # train Sharpe, which the 2026-05-31 bcc_f2 parity test showed does NOT
    # transfer to QC (proxy +1.52 → QC −4.0) — so this threshold is a MINOR knob
    # vs the proxy↔QC signal divergence; see
    # docs/diagnostics/gp_holistic_audit_2026_05_31.md before relying on it.
    # MB1/MH1 (2026-06-01 holistic review): the FRONT-DSR gate (this τ, applied over
    # the whole search's (N, V)) is the ONLY multiple-testing control — the cross-fold
    # G1 is a near-vacuous secondary screen for small survivor pools and must NOT be
    # relied on for the N-trial correction. DSR = P(beat SR*), so HIGHER τ = STRICTER;
    # the significance bar is 0.95, "beats best-of-N-random" is 0.5, 0.05 is near
    # pass-through.
    # RUN VALUE 0.25 (user choice 2026-06-01): a "keep borderline candidates" bar —
    # stricter than near-pass-through, but BELOW the honest viability bar (0.5). Chosen
    # because an empty fleet under τ=0.5 could be an UNDISCOVERED BUG rather than
    # genuine no-edge, and the L3 pipeline still needs candidates to be tested against.
    # τ=0.25 carries NO viability/significance claim; a CONFIRMATORY viability run MUST
    # re-run at τ=0.5. Pin in the fresh pre-reg.
    dsr_threshold: float = 0.25
    # G11 transferability: when True, re-score each template's Pareto front under
    # calibrated terminal noise (mc_noise_robustness_gate) POST-evolution and attach
    # mc_median_sharpe; the cross-fold survival gate G11 then rejects strategies whose
    # edge does not survive the proxy↔QC option-chain noise gap. Default OFF — it adds
    # ~n_realizations backtests per front member, and a discovery pilot wants the raw
    # front first. Turn ON for a confirmatory/production run. The gate is a no-op when
    # this is OFF (mc_median_sharpe stays unset).
    mc_robustness_enabled: bool = False
    mc_robustness_realizations: int = 20

    def __post_init__(self):
        valid_conditions = ("real-l1", "shuffled-l1", "scalar-only", "probes-only", "emb-only")
        if self.condition not in valid_conditions:
            raise ValueError(
                f"condition must be one of {valid_conditions}, "
                f"got {self.condition!r}"
            )
        if self.shuffle_seed is None:
            self.shuffle_seed = self.seed + _SHUFFLE_SEED_OFFSET
        if self.shuffle_seed == self.seed:
            raise ValueError(
                f"shuffle_seed must differ from seed to keep the shuffled-L1 "
                f"permutation RNG decoupled from the GP RNG (both were {self.seed})"
            )


# ---------------------------------------------------------------------------
# Walk-forward fold definitions (SVP-001)
# ---------------------------------------------------------------------------

# 3 folds + 1 holdout. Anchored expanding training window.
# Each val period is ~125 trading days. 5-day embargo between splits.
# See (internal doc) for justification.
WALK_FORWARD_FOLDS = [
    {"fold_id": 1, "train_end": "2024-03-29", "val_end": "2024-09-27",
     "test_end": "2025-03-31", "label": "F1: val Apr-Sep 2024"},
    {"fold_id": 2, "train_end": "2024-09-27", "val_end": "2025-03-31",
     "test_end": "2025-09-30", "label": "F2: val Oct 2024-Mar 2025"},
    {"fold_id": 3, "train_end": "2025-03-31", "val_end": "2025-09-30",
     "test_end": "2026-03-31", "label": "F3: val Apr-Sep 2025"},
    # F4 TRUE HOLDOUT (2026-06-01, USER DECISION): the prior test_end=None made the
    # test window ~1 trading day (corpus ends 2026-04-09) → test_sharpe was noise. We
    # now carve a GENUINE >=20-trading-day untouched test holdout: val_end pulled back
    # to 2026-03-04 so [val_end + 5-day embargo, 2026-04-09] = 20 DISTINCT trading days
    # (test first day 2026-03-12), with val still 99 distinct trading days (Oct 2025–
    # 2026-03-04). Verified against raw_data/local_store/l1_minute_scalars.parquet via
    # the same split_by_date logic (5-day embargo on both train→val and val→test).
    {"fold_id": 4, "train_end": "2025-09-30", "val_end": "2026-03-04",
     "test_end": "2026-04-09", "label": "F4: val Oct 2025-Mar 2026 + test holdout to Apr 2026"},
]

# Harvey-Liu haircut threshold (static fallback for the cross-fold survival
# gate G1 when the evolution-level (N, V) is unavailable, e.g. legacy JSONL).
# The DSR math itself lives ONLY in layer2/pbo.py now (single implementation):
# the legacy in-module _expected_max_sr / deflated_sharpe_threshold /
# deflated_sharpe_ratio were DELETED 2026-05-30 (P1-B redesign, Blocker C) —
# they compared an ANNUALIZED Sharpe against a z-scale SR* (V≡1), which was
# silently over-strict (passed only at ann-Sharpe ≳ 3.74). All deflation now
# de-annualizes consistently via pbo.expected_max_sharpe / deflated_sharpe_ratio.
HAIRCUT_VAL_SHARPE_THRESHOLD = 0.30


# Survival gates (SVP-001, Section: Strategy Survival Gates)
SURVIVAL_GATES = {
    "G1_haircut_val_sharpe": 0.30,     # min val_sharpe after multiple-testing correction
    "G2_min_folds_positive": 2,        # positive val Sharpe in at least N of 3 folds
                                       # (NOTE: G2 is ANNOTATE-ONLY — this threshold
                                       # now sets the g2_persisted flag, not a filter)
    # MEDIUM-7 (2026-06-01 audit): lowered 100→40 to AGREE with the optimizer's
    # trade-count objective (trade_count_score full-reward plateau STARTS at
    # 0.3/day; over a ~125-day val window that is ~37 trades). At G3=100 (~0.8/day)
    # a SEARCH-OPTIMAL 0.3/day champion — fully rewarded by the objective — would be
    # rejected by the survival gate, a direct search↔gate contradiction. We align
    # the GATE to the SEARCH (not vice-versa) so we do not distort the grounded
    # 0DTE-frequency objective; 40 val trades still supports a meaningful Sharpe
    # estimate. (G3 remains a statistical-POWER floor, just one consistent with the
    # objective's lower edge rather than its ~0.8/day midpoint.)
    "G3_min_trades_val": 20,           # 2026-06-02 (b): 40→20, the val-trade analog of
                                       # the relaxed min_trades=20 so a selective signal
                                       # can be tagged g3_passed/survival_passed (annotate-
                                       # only; the high-confidence subset is still flagged)
    "G4_max_drawdown": 0.15,           # max drawdown < 15% of notional
    "G5_min_regime_sharpe": -0.3,      # no regime with Sharpe below this (tightened from -1.0)
    "G6_max_proxy_qc_gap": 0.50,       # proxy-QC gap tolerance
    "G7_min_qc_test_sharpe": 0.0,      # QC test Sharpe > 0
    "G8_min_profit_factor": 1.1,       # profit factor (Kelly-positive with margin)
    "G11_min_mc_median_sharpe": 0.0,   # transferability: median Sharpe under calibrated
                                       # terminal noise must stay > 0 (edge survives the
                                       # proxy↔QC option-chain noise gap). Annotated by
                                       # mc_noise_robustness_gate; gate is a no-op until
                                       # mc_robustness_enabled attaches the value.
}

# Template correlation groups: templates sharing structural risk exposure.
# IC = BPC + BCC (same legs combined). Selecting strategies from correlated
# templates creates concentrated portfolio risk. The portfolio gate (G9)
# limits survivors to at most 1 strategy per correlation group.
TEMPLATE_CORRELATION_GROUPS = {
    "neutral_credit": {
        "iron_condor_narrow", "iron_condor_standard", "iron_condor_wide",
        "iron_butterfly_narrow", "iron_butterfly_standard",
        # Backward-compatible old names (also used by Level B base templates)
        "iron_condor", "iron_butterfly",
    },
    "bull_credit": {
        "bull_put_credit_narrow", "bull_put_credit_standard", "bull_put_credit_wide",
        "bull_put_credit",
    },
    "bear_credit": {
        "bear_call_credit_narrow", "bear_call_credit_standard", "bear_call_credit_wide",
        "bear_call_credit",
    },
    "bull_debit":     {"bull_call_debit"},
    "bear_debit":     {"bear_put_debit"},
    "bear_backspread": {"ratio_put_backspread"},
}


def _template_group(template_name: str) -> str:
    """Return the correlation group for a template."""
    for group, members in TEMPLATE_CORRELATION_GROUPS.items():
        if template_name in members:
            return group
    return template_name  # ungrouped = own group


# ---------------------------------------------------------------------------
# Per-fold probe refit ( look-ahead fix)
# ---------------------------------------------------------------------------

def _refit_probes_for_fold(
    base_df: "pd.DataFrame",
    probe_bundle: dict,
    train_end: str,
) -> "pd.DataFrame":
    """Refit probes using only data <= train_end and replace probe columns.

    Args:
        base_df: Minute-resolution DataFrame (enriched Parquet).
        probe_bundle: Loaded NPZ with emb_fine, window_dates, bar_positions,
                      target_* and valid_* arrays.
        train_end: Fold's training cutoff date (YYYY-MM-DD).

    Returns:
        Copy of base_df with probe columns overwritten using per-fold probes.
    """
    from layer1.inference.batch_forecast import (
        fit_probes_standalone, predict_probes_standalone,
    )

    emb_fine = probe_bundle["emb_fine"]             # (N, 3072)
    window_dates = probe_bundle["window_dates"]     # (N,) str
    bar_positions = probe_bundle["bar_positions"]   # (N,) int32

    # 1. Build train mask for this fold
    train_mask = window_dates <= train_end
    n_train = int(train_mask.sum())
    print(f"    [probe refit] train_end={train_end}: {n_train}/{len(emb_fine)} "
          f"windows for probe fitting")

    # 2. Build targets dict for training subset
    # Bundle stores targets as target_rv_15 etc, validity as valid_rv_15 etc.
    _TARGET_KEYS = {
        "rv_15": "target_rv_15", "rv_30": "target_rv_30",
        "regime": "target_regime", "spread": "target_spread",
        "gamma_accel": "target_gamma_accel",
        "smile_convex": "target_smile_convex",
        "flow_tox": "target_flow_tox",
        "jump_30": "target_jump_30",
    }
    # Remap bundle target_spread → fit key "spread" (not spread_5)
    fit_targets = {}
    for fit_key, bundle_key in _TARGET_KEYS.items():
        if bundle_key in probe_bundle:
            arr = probe_bundle[bundle_key][train_mask]
            # Apply validity mask: set invalid rows to NaN so standalone
            # functions' NaN filtering handles them correctly.
            # Integer targets (e.g. regime int64) can't hold NaN —
            # cast to float first.
            valid_key = f"valid_{fit_key}"
            if valid_key in probe_bundle:
                valid = probe_bundle[valid_key][train_mask]
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float64)
                else:
                    arr = arr.copy()
                arr[~valid] = np.nan
            else:
                arr = arr.copy()
            fit_targets[fit_key] = arr

    # For core probes (rv_15, rv_30, regime, spread), drop NaN rows
    # (standalone functions expect no NaNs for these 4).
    core_valid = np.ones(n_train, dtype=bool)
    for k in ("rv_15", "rv_30", "regime", "spread"):
        if k in fit_targets:
            core_valid &= ~np.isnan(fit_targets[k])
    n_valid = int(core_valid.sum())
    for k in ("rv_15", "rv_30", "regime", "spread"):
        if k in fit_targets:
            fit_targets[k] = fit_targets[k][core_valid]
    emb_fine_train = emb_fine[train_mask][core_valid]

    # probes keep NaN (standalone handles it via internal filtering)
    # but they must be indexed to the same core_valid subset for emb_fine alignment.
    for k in ("gamma_accel", "smile_convex", "flow_tox", "jump_30"):
        if k in fit_targets:
            fit_targets[k] = fit_targets[k][core_valid]

    # 3. Fit probes
    probes = fit_probes_standalone(emb_fine_train, fit_targets)
    print(f"    [probe refit] fitted {len(probes)} probes on "
          f"{n_valid}/{n_train} valid windows")

    # 4. Predict on ALL windows
    preds = predict_probes_standalone(probes, emb_fine)

    # 5. Build 5-min prediction DataFrame
    pred_5min = {
        "date": window_dates.astype(str),
        "bar_position": bar_positions.astype(np.int64),
    }
    _PRED_MAP = {
        "PredRV15": "rv_15", "PredRV30": "rv_30",
        "PredSpread": "spread", "PredRegime": "predicted_regime",
    }
    for col, key in _PRED_MAP.items():
        if key in preds:
            pred_5min[col] = preds[key]
    # probes
    _PRED_MAP_V2 = {
        "PredGammaAccel": "gamma_accel",
        "PredSmileConvexity": "smile_convex",
        "PredJump": "jump_proba",
        "PredFlowToxicity": "flow_tox",
    }
    for col, key in _PRED_MAP_V2.items():
        if key in preds:
            pred_5min[col] = preds[key]
    # Regime probabilities
    if "regime_proba" in preds:
        for k in range(preds["regime_proba"].shape[1]):
            pred_5min[f"RegimeProb{k}"] = preds["regime_proba"][:, k]

    pred_df = pd.DataFrame(pred_5min)

    # 6. Forward-fill from 5-min to 1-min
    return _forward_fill_probes(base_df, pred_df)


def _forward_fill_probes(
    minute_df: "pd.DataFrame",
    pred_5min_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Replace probe columns in minute_df with forward-filled 5-min predictions.

    Uses per-date merge_asof — same approach as
    generate_enriched_minute_parquet.py. Each date is merged independently
    because merge_asof requires globally monotonic keys on the 'on' column,
    which bar_position is NOT across multiple dates.
    """
    result = minute_df.copy()

    probe_cols = [c for c in pred_5min_df.columns
                  if c not in ("date", "bar_position")]

    # Drop stale probe columns
    for col in probe_cols:
        if col in result.columns:
            result.drop(columns=[col], inplace=True)

    # Ensure date types match
    result["date"] = result["date"].astype(str)
    pred_5min_df = pred_5min_df.copy()
    pred_5min_df["date"] = pred_5min_df["date"].astype(str)

    # Per-date merge_asof
    merged_parts = []
    for date in sorted(result["date"].unique()):
        left = result[result["date"] == date].sort_values(
            "bar_position").reset_index(drop=True)
        right = pred_5min_df[pred_5min_df["date"] == date].sort_values(
            "bar_position").reset_index(drop=True)
        if len(right) == 0:
            for col in probe_cols:
                left[col] = 0.0
            merged_parts.append(left)
        else:
            part = pd.merge_asof(
                left,
                right[["bar_position"] + probe_cols],
                on="bar_position", direction="backward",
            )
            merged_parts.append(part)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Bars before first window get NaN → fill with 0.0
    for col in probe_cols:
        merged[col] = merged[col].fillna(0.0)

    assert len(merged) == len(minute_df), (
        f"Forward-fill changed row count: {len(minute_df)} → {len(merged)}")

    return merged


def _null_out_invalid_test_fields(pareto_front: List[Dict]) -> List[Dict]:
    """MEDIUM-11 (2026-06-01 audit): mark a VAL-ONLY fold's test_* as invalid.

    Fold 4's holdout has ``test_end=None`` — any test split over the post-val
    remainder collapses to ~1 trading day, so ``test_sharpe`` / ``test_sortino`` /
    ``n_trades_test`` are NOISE. NULL them out and set ``test_invalid=True`` so a
    downstream PBO/handoff reader cannot mistake 1-day garbage for OOS evidence.
    Mutates each member in place (and returns the list for convenience).
    """
    for _ind in pareto_front:
        _ind["test_sharpe"] = None
        _ind["test_sortino"] = None
        _ind["n_trades_test"] = None
        _ind["test_invalid"] = True
    return pareto_front


def run_walk_forward(
    base_config: ExperimentConfig,
    folds: Optional[List[Dict]] = None,
    output_base: Optional[str] = None,
) -> Dict:
    """Run walk-forward GP evolution across multiple folds.

    For each fold, creates an ExperimentConfig with the fold's train_end/val_end,
    runs GP evolution, and collects per-fold results. After all folds complete,
    applies cross-fold persistence scoring and survival gates.

    Args:
        base_config: Template config (pop_size, condition, templates, etc.)
        folds: List of fold dicts (default: WALK_FORWARD_FOLDS folds 1-3)
        output_base: Base output directory (each fold writes to output_base/fold_N/)

    Returns:
        Dict with per-fold results, cross-fold scores, and survival gate outcomes.
    """
    if folds is None:
        # Default: folds 1-3 (fold 4 = holdout, run separately)
        folds = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] <= 3]

    if output_base is None:
        output_base = f"results/wf_{base_config.condition}_{int(time.time())}"

    print(f"=== Walk-forward GP: {len(folds)} folds, condition={base_config.condition} ===\n")

    # Per-fold probe refit ( look-ahead fix)
    _probe_bundle = None
    _base_df = None
    _base_schema = None
    _base_l1_sha256 = None
    if base_config.probe_bundle_path is not None:
        _probe_bundle = dict(np.load(base_config.probe_bundle_path, allow_pickle=False))
        # Validate required keys
        _req_keys = {"emb_fine", "window_dates", "bar_positions"}
        _missing_keys = _req_keys - set(_probe_bundle.keys())
        if _missing_keys:
            raise ValueError(
                f"Probe bundle missing required keys: {_missing_keys}. "
                f"Expected at least: {sorted(_req_keys)}")
        # Coerce dates to str (NPZ may store as |S10 bytes depending on numpy version)
        _probe_bundle["window_dates"] = _probe_bundle["window_dates"].astype(str)
        # Compute content hash for fingerprint (not path — renames shouldn't break resume)
        _bundle_sha = _file_sha256(Path(base_config.probe_bundle_path).resolve())
        base_config.probe_bundle_sha256 = _bundle_sha
        # Validate encoder checkpoint SHA: probe bundle must match PLS bases
        # and L1 Parquet. A mismatched bundle produces garbage probes silently.
        _prov_path = Path(base_config.probe_bundle_path).with_suffix(".provenance.json")
        if _prov_path.exists():
            import json as _json
            _prov = _json.loads(_prov_path.read_text())
            _bundle_ckpt_sha = _prov.get("checkpoint_sha256", "")
            # Cross-check against PLS bases manifest if available
            _pls_manifest_path = Path("layer2/pca_bases_manifest.json")
            if _pls_manifest_path.exists():
                _pls_manifest = _json.loads(_pls_manifest_path.read_text())
                _pls_ckpt_sha = _pls_manifest.get("checkpoint_sha256", "")
                if _bundle_ckpt_sha and _pls_ckpt_sha and _bundle_ckpt_sha != _pls_ckpt_sha:
                    raise ValueError(
                        f"Probe bundle encoder SHA ({_bundle_ckpt_sha[:16]}...) "
                        f"!= PLS bases encoder SHA ({_pls_ckpt_sha[:16]}...). "
                        f"Bundle and PLS were fitted to different encoder checkpoints."
                    )
        print(f"  [probe refit] Loaded bundle: {base_config.probe_bundle_path}")
        print(f"  [probe refit] bundle_sha256: {_bundle_sha[:16]}...")
        _bd = _probe_bundle["window_dates"]
        print(f"  [probe refit] emb_fine: {_probe_bundle['emb_fine'].shape}, "
              f"dates: {_bd[0]} .. {_bd[-1]}")
        # Pre-load enriched Parquet once (it's ~1 GB, don't reload per fold)
        if not base_config.use_vectorized:
            raise ValueError(
                "Per-fold probe refit requires --minute (vectorized evaluator). "
                "The minute-resolution enriched Parquet is the only format that "
                "supports forward-filled probe replacement."
            )
        _base_df, _base_schema = load_minute_parquet(base_config.l1_parquet_path)
        _base_l1_sha256 = _file_sha256(Path(base_config.l1_parquet_path).resolve())
        print(f"  [probe refit] Pre-loaded minute Parquet: {len(_base_df)} rows")

    # / : Probe train-end alignment.
    # Only conditions that consume probes (B=probes-only, D=real-l1) are
    # affected by probe look-ahead. A (scalar-only) and C (emb-only) strip
    # probe terminals and are safe regardless.
    _PROBE_CONDITIONS = ("probes-only", "real-l1", "shuffled-l1")
    _PROBE_TRAIN_CUTOFF = "2024-09-30"
    if _probe_bundle is not None:
        print("  [probe refit] Per-fold probe refit active — "
              "skipping static cutoff assertion")
    elif base_config.condition in _PROBE_CONDITIONS:
        _leak_folds = [f for f in folds if f["train_end"] < _PROBE_TRAIN_CUTOFF]
        if _leak_folds:
            raise ValueError(
                f"Condition '{base_config.condition}' with folds that have "
                f"train_end before {_PROBE_TRAIN_CUTOFF} requires "
                f"--probe-bundle to prevent probe look-ahead leak. "
                f"Affected folds: {[f['fold_id'] for f in _leak_folds]}"
            )

    fold_results = {}
    for fold in folds:
        fold_id = fold["fold_id"]
        # Per-template subdirectory prevents race conditions when multiple
        # template processes run in parallel on the same output_base.
        # Previous: all templates shared fold_N/ → run_fingerprint.json
        # atomic rename race crashed IB repeatedly.
        _tpl_name = base_config.template_names[0] if len(base_config.template_names) == 1 else "multi"
        fold_dir = f"{output_base}/{_tpl_name}/fold_{fold_id}"
        print(f"\n{'='*60}")
        print(f"  Fold {fold_id}: {fold['label']}")
        print(f"  train_end={fold['train_end']}  val_end={fold['val_end']}")
        print(f"{'='*60}\n")

        fold_config = _dc_replace(
            base_config,
            train_end=fold["train_end"],
            val_end=fold["val_end"],
            test_end=fold.get("test_end"),
            output_dir=fold_dir,
        )

        if _probe_bundle is not None:
            # Per-fold probe refit: replace probe columns in a copy of base_df
            fold_df = _refit_probes_for_fold(
                _base_df, _probe_bundle, fold["train_end"],
            )
            result = run_experiment(
                fold_config,
                _preloaded_data=(fold_df, _base_schema),
                _precomputed_l1_sha256=_base_l1_sha256,
            )
        else:
            result = run_experiment(fold_config)
        fold_results[fold_id] = {
            "config": {"train_end": fold["train_end"], "val_end": fold["val_end"]},
            "label": fold["label"],
            "template_results": [],
        }
        # MEDIUM-11 (2026-06-01 audit): fold 4's holdout has test_end=None — a
        # test split over the post-val_end remainder collapses to ~1 trading day,
        # so test_sharpe/test_sortino/n_trades_test are NOISE, not OOS evidence.
        # NULL them out (and flag test_invalid) so downstream PBO/handoff cannot
        # read 1-day garbage as a holdout result. F4 is VAL-ONLY by design (the
        # val window IS its OOS evidence). Folds 1-3 (real test_end) keep test_*.
        _f4_test_invalid = fold.get("test_end") is None
        for tr in result.template_results:
            _pf = tr.pareto_front
            if _f4_test_invalid:
                _null_out_invalid_test_fields(_pf)
            fold_results[fold_id]["template_results"].append({
                "template_name": tr.template_name,
                "pareto_front_size": tr.pareto_front_size,
                "val_hypervolume": tr.val_hypervolume,
                "wall_time_s": tr.wall_time_s,
                "pareto_front": _pf,  # list of dicts with trees + fitness
            })

        # MEDIUM-10 (2026-06-01 audit): write an INCREMENTAL progress marker after
        # EVERY fold (not only the final walk_forward_summary.json, which lands
        # after the LAST fold). A ~12-day run that is interrupted mid-walk-forward
        # then leaves visible per-fold status instead of nothing.
        try:
            _prog_path = Path(output_base)
            _prog_path.mkdir(parents=True, exist_ok=True)
            _prog = {
                "completed_folds": sorted(fold_results.keys()),
                "n_folds_planned": len(folds),
                "last_fold_completed": fold_id,
                "condition": base_config.condition,
                "per_fold_template_front_sizes": {
                    fid: {tr["template_name"]: tr["pareto_front_size"]
                          for tr in fr["template_results"]}
                    for fid, fr in fold_results.items()
                },
                "timestamp": time.time(),
            }
            (_prog_path / "walk_forward_progress.json").write_text(
                json.dumps(_prog, indent=2, default=str))
        except Exception as _prog_exc:
            print(f"  [progress] could not write walk_forward_progress.json: "
                  f"{type(_prog_exc).__name__}: {_prog_exc}", file=sys.stderr)

    # Cross-fold persistence scoring
    print(f"\n{'='*60}")
    print(f"  Cross-fold persistence analysis")
    print(f"{'='*60}\n")

    persistence = _compute_cross_fold_persistence(fold_results)

    # Apply survival gates (G1-G4, G8 — G5-G7 require QC/regime data, applied later)
    #
    # n_trials = pop_size × n_templates is passed only as the per-strategy-vs-
    # static enable TOGGLE for G1 (n_trials > 0 → per-strategy DSR). G1's SR* is
    # NOT computed from this full-search N — that would double-count the
    # multiple-testing inflation the per-template FRONT gate already absorbed
    # (and pairs an inconsistent N with the candidate pool's V; review H1).
    # G1 instead uses N_eff = the candidate-pool size (see _apply_proxy_survival_
    # gates). Each individual in one generation is an independent lineage root;
    # templates are genuinely independent searches (different payoff structures);
    # folds do NOT multiply N (same strategy) and generations do NOT multiply N
    # (offspring share ~50% material via crossover — correlated, not independent).
    # Reference: Bailey & Lopez de Prado (2014), Section 3.2.
    all_templates = set()
    for fr in fold_results.values():
        for tr in fr["template_results"]:
            all_templates.add(tr["template_name"])
    n_templates = max(len(all_templates), 1)
    n_trials = base_config.pop_size * n_templates
    n_val_days = 125  # approximate; all folds have ~125 val days
    survivors = _apply_proxy_survival_gates(persistence, n_trials=n_trials,
                                            n_val_days=n_val_days)

    # PBO/CSCV: Probability of Backtest Overfitting (Bailey et al., 2017).
    # Walk-forward with 4 pre-defined folds is insufficient — 4 partitions
    # out of the combinatorial space of possible train/test splits. PBO
    # enumerates all C(N, k) partitions to give a probability, not a binary
    # pass/fail on hand-picked splits.
    #
    # Performance optimization (validated methodology):
    # - Top-10 strategies per template (PBO measures selection bias — if the
    # top-10 aren't overfit, lower-ranked ones are less relevant since they
    # wouldn't be deployed). Reduces backtests proportionally.
    # - Fold training dates only (not full 882-day dataset). Each fold's
    # strategies were evolved on that fold's training data, so PBO should
    # test overfitting within the same temporal scope. Reduces data per
    # backtest by ~40%.
    # - 8 groups, 2 test = C(8,2) = 28 CSCV combinations (Bailey et al.
    # 2017 use 6-16 groups in their examples).
    #
    # Expected runtime: ~4-6 hours for 5 templates × 4 folds (parallelizable).
    _PBO_MAX_STRATEGIES = 10
    pbo_results: Dict[str, Dict] = {}
    if base_config.use_vectorized:
        from layer2.pbo import compute_pbo_from_pareto_front
        from layer2.templates import base_template_by_name, template_by_name
        print("\n  [PBO] Computing Probability of Backtest Overfitting per template...")
        print(f"  [PBO] Top-{_PBO_MAX_STRATEGIES} strategies per template, fold training dates only")
        # Load data once (use load_minute_parquet for schema consistency)
        if _base_df is not None:
            _pbo_data = _base_df
        else:
            _pbo_data, _ = load_minute_parquet(base_config.l1_parquet_path)
        for fid, fr in fold_results.items():
            # Restrict data to this fold's training period
            fold_meta = fr.get("config", {}).get("train_end")
            if fold_meta:
                _fold_data = _pbo_data[_pbo_data["date"].astype(str) <= str(fold_meta)]
            else:
                _fold_data = _pbo_data
            for tr in fr["template_results"]:
                tname = tr["template_name"]
                pareto = tr.get("pareto_front", [])
                if len(pareto) < 3:
                    continue  # PBO needs >= 3 strategies
                # Top-N by val_sharpe (these are the candidates we'd deploy)
                pareto_sorted = sorted(pareto,
                                       key=lambda s: s.get("val_sharpe", 0),
                                       reverse=True)
                pareto_top = pareto_sorted[:_PBO_MAX_STRATEGIES]
                try:
                    tmpl = base_template_by_name(tname)
                except Exception:
                    try:
                        tmpl = template_by_name(tname)
                    except Exception:
                        continue
                try:
                    pbo_result = compute_pbo_from_pareto_front(
                        pareto_top, _fold_data, tmpl,
                        n_groups=8, n_test_groups=2,
                        embargo_days=base_config.embargo_days,
                    )
                    pbo_key = f"{tname}_fold{fid}"
                    pbo_results[pbo_key] = pbo_result
                    pbo_val = pbo_result["pbo"]
                    _interpretation = (
                        "LOW (robust)" if pbo_val < 0.10
                        else "MODERATE (caution)" if pbo_val < 0.40
                        else "HIGH (likely overfit)"
                    )
                    print(f"    {pbo_key}: PBO={pbo_val:.3f} [{_interpretation}] "
                          f"(n_strategies={pbo_result['n_strategies']}, "
                          f"n_combinations={pbo_result['n_combinations']})")
                    # Attach PBO to survivors matching this template AND fold
                    for s in survivors:
                        if (s.get("template_name") == tname
                                and str(s.get("fold_id", "")) == str(fid)):
                            s["pbo"] = pbo_val
                            s["pbo_key"] = pbo_key
                except Exception as _pbo_exc:
                    import sys
                    print(f"    {tname}_fold{fid}: PBO FAILED — "
                          f"{type(_pbo_exc).__name__}: {_pbo_exc}",
                          file=sys.stderr)

    # Save walk-forward summary
    summary = {
        "n_folds": len(folds),
        "folds": {fid: {k: v for k, v in fr.items() if k != "template_results"}
                  for fid, fr in fold_results.items()},
        "persistence_scores": persistence,
        "survivors": survivors,
        "survival_gates": SURVIVAL_GATES,
        # MEDIUM-10 / MEDIUM-13 (2026-06-01 audit): METHODOLOGY LIMITATIONS that
        # downstream consumers MUST honour — this summary cannot enforce them.
        "methodology_notes": {
            "G9_cross_template": (
                "G9 (portfolio diversification: ≤1 strategy per correlation group "
                "ACROSS templates) CANNOT run here when templates are evolved in "
                "SEPARATE per-template processes — each process's summary sees ONLY "
                "its own template, so the cross-template group de-duplication is a "
                "no-op. Cross-template G9 MUST be applied in a SEPARATE post-hoc "
                "aggregation over ALL per-template summaries. As of 2026-06-01 "
                "neither scripts/pilot_to_qc_handoff.py nor scripts/analyze_gp_"
                "results.py performs cross-template G9 — it is a REQUIRED MANUAL "
                "step before any multi-template portfolio claim. Do NOT treat the "
                "per-template survivor list as portfolio-diversified."
            ),
            "G2_annotate_only": (
                "G2 cross-fold persistence is ANNOTATE-ONLY (g2_persisted flag), not "
                "a filter — independently-evolved folds make byte-identical re-"
                "emergence measure-zero, so a hard G2 would empty the fleet."
            ),
            "anchored_expanding_non_independence": (
                "Folds are ANCHORED-EXPANDING (each fold trains on the prior fold's "
                "val window), so they share the val→train boundary and are NOT "
                "independent. Cross-fold persistence is therefore SUGGESTIVE, not a "
                "significance test. The OOS evidence is the front-DSR annotation "
                "(dsr_passed) + the F4 hold-out VAL window (F4 test_* is nulled — "
                "test_invalid — as it collapses to ~1 day)."
            ),
        },
        "pbo_results": {k: {kk: vv for kk, vv in v.items()
                            if kk != "logit_distribution"}
                        for k, v in pbo_results.items()} if pbo_results else {},
        "base_config": {k: str(v) for k, v in asdict(base_config).items()},
    }
    out_path = Path(output_base)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "walk_forward_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nSaved walk-forward summary to {out_path / 'walk_forward_summary.json'}")
    return summary


def _compute_cross_fold_persistence(fold_results: Dict) -> List[Dict]:
    """Score each strategy by how many folds it achieves positive val Sharpe.

    Two matching strategies:
    1. **Exact match**: identical entry+exit s-expressions across folds (seed-derived)
    2. **Behavioral match**: same template, and the BEST strategy per (template, fold)
       is compared across folds. This captures independently evolved strategies that
       converge to similar behavior without identical tree structure.

    For G2 (persistence gate), we use per-template behavioral matching: for each
    template, collect the best val_sharpe per fold. If the template has positive
    val_sharpe in 2+ folds (from ANY strategy in that template), the template
    passes G2. Individual strategies inherit their template's fold persistence.

    Returns list of dicts sorted by (folds_positive, mean_val_sharpe).
    """
    # --- SAME-KEY cross-fold persistence (MB2, 2026-06-01 holistic review) ---
    # A strategy "persists" only if the SAME canonical strategy (entry+exit+size+
    # delta key, commutative-normalized) is positive in 2+ folds. The prior per-fold-
    # BESTS matching credited a template even when a DIFFERENT tree won each fold;
    # over anchored/expanding walk-forward folds (F2 trains on F1's val) that is NOT
    # independent confirmation. Each individual inherits ITS OWN key's persistence.
    # (Structural similarity is now automatic — same key ⇒ identical structure — so
    # the prior Jaccard heuristic is retired.) NOTE: walk-forward folds are
    # autocorrelated, so even same-key persistence is SUGGESTIVE, not a significance
    # test.
    from layer2.grammar import from_sexpr, canonical_key

    # BLOCKER-3 (2026-06-01 audit): a degenerate all-sentinel front has
    # val_sharpe == -FAILED_FITNESS_SENTINEL (-1e6) on every member (it failed
    # the val-gate reconciliation). Such rows are NOT strategies — they must
    # not seed persistence keys, inflate the candidate pool, or be serialized as
    # "champions". Drop any val_sharpe <= -1e5 at the reader boundary so the
    # sentinel never propagates past cross-fold collection.
    _SENTINEL_VAL_SHARPE = -1e5

    def _is_sentinel(ind: Dict) -> bool:
        try:
            return float(ind.get("val_sharpe", 0.0)) <= _SENTINEL_VAL_SHARPE
        except (TypeError, ValueError):
            return False

    _key_cache: Dict[Tuple, str] = {}

    def _strategy_key(ind: Dict) -> str:
        # stop_mult is part of strategy identity for cross-fold persistence: a
        # hold-to-expiry (0.0) and a 2.5× variant of the same trees are DIFFERENT
        # strategies. Round to 1 dp to collapse Gaussian-mutation jitter (so the
        # "same ~2× stop" re-evolving across independent folds matches), matching
        # the EphReal _eph_round=2 jitter-collapse philosophy below.
        _sm = ind.get("stop_mult")
        _sm_key = "None" if _sm is None else f"{float(_sm):.1f}"
        _sig = (ind.get("entry_tree"), ind.get("exit_tree"),
                ind.get("size_tree"), ind.get("delta_tree"), _sm_key)
        if _sig in _key_cache:
            return _key_cache[_sig]
        _parts = []
        for _sx in _sig[:4]:
            if not _sx:
                _parts.append("None")
                continue
            try:
                # _eph_round=2 (campaign review HIGH-2): collapse EphReal JITTER
                # (e.g. 0.5740740740 → 0.57) so a structurally-identical strategy that
                # re-evolves across INDEPENDENT folds matches, while genuinely different
                # thresholds (0.50 vs 0.70 = different behavior) stay distinct. Without
                # rounding, full-precision constants made same-key matching measure-zero
                # across independent fold evolutions → G2 was structurally DEAD.
                _parts.append(canonical_key(from_sexpr(_sx), _eph_round=2))
            except Exception:
                _parts.append(str(_sx))  # robust fallback: raw sexpr (stricter, still same-key)
        _parts.append(f"SM:{_sm_key}")
        _k = "|".join(_parts)
        _key_cache[_sig] = _k
        return _k

    # (template, key) → {fold_id → best val Sharpe of THIS exact key in that fold}
    _key_rec: Dict[Tuple[str, str], Dict] = {}
    for fold_id, fr in fold_results.items():
        for tr in fr["template_results"]:
            template = tr["template_name"]
            for ind in tr["pareto_front"]:
                if _is_sentinel(ind):
                    continue  # BLOCKER-3: skip all-sentinel front members
                _kk = (template, _strategy_key(ind))
                _rec = _key_rec.setdefault(_kk, {"per_fold_val": {}, "per_fold_trades": {}})
                _vs = ind.get("val_sharpe", 0)
                if fold_id not in _rec["per_fold_val"] or _vs > _rec["per_fold_val"][fold_id]:
                    _rec["per_fold_val"][fold_id] = _vs
                    _rec["per_fold_trades"][fold_id] = ind.get("n_trades_val", 0)

    # per-KEY persistence
    key_persistence: Dict[Tuple[str, str], Dict] = {}
    for _kk, _rec in _key_rec.items():
        _pfv = _rec["per_fold_val"]
        _folds_positive = sum(1 for sh in _pfv.values() if sh > 0)
        key_persistence[_kk] = {
            "folds_seen": sorted(_pfv.keys()),
            "folds_positive": _folds_positive,
            "per_fold_val_sharpe": _pfv,
            "per_fold_trades": _rec["per_fold_trades"],
            "mean_jaccard": 1.0,           # same key ⇒ structurally identical
            "structurally_similar": True,  # by construction (same canonical tree)
            "n_folds_seen": len(_pfv),
        }

    print("  SAME-KEY cross-fold persistence (top 10 by folds_positive):")
    for (template, _k), tp in sorted(key_persistence.items(),
                                     key=lambda x: x[1]["folds_positive"], reverse=True)[:10]:
        if tp["folds_positive"] < 1:
            continue
        folds_str = ", ".join(f"F{fid}={sh:+.2f}"
                              for fid, sh in sorted(tp["per_fold_val_sharpe"].items()))
        print(f"    {template:20s} persist={tp['folds_positive']}/{tp['n_folds_seen']}  [{folds_str}]")

    # --- Collect all individual strategies, each carrying ITS OWN key's persistence ---
    results = []
    for fold_id, fr in fold_results.items():
        for tr in fr["template_results"]:
            template = tr["template_name"]
            for ind in tr["pareto_front"]:
                if _is_sentinel(ind):
                    continue  # BLOCKER-3: never carry a sentinel "champion" forward
                tp = key_persistence.get((template, _strategy_key(ind)), {})
                val_sh = ind.get("val_sharpe", 0)
                results.append({
                    "template": template,
                    "entry_tree": ind.get("entry_tree", ""),
                    "exit_tree": ind.get("exit_tree", ""),
                    "size_tree": ind.get("size_tree", ""),
                    "fold_id": fold_id,
                    "train_sharpe": ind.get("train_sharpe", 0.0),
                    "val_sharpe": val_sh,
                    # MEDIUM-11: a VAL-ONLY fold (F4) nulls test_* to None. Coerce to
                    # 0.0 / 0 HERE so the persistence record never carries None into a
                    # downstream `:+.3f` format or numeric comparison; the test_invalid
                    # flag (carried below) is the authoritative "ignore test" signal.
                    "test_sharpe": (ind.get("test_sharpe") if ind.get("test_sharpe")
                                    is not None else 0.0),
                    "n_trades_val": ind.get("n_trades_val", 0),
                    "n_trades_test": (ind.get("n_trades_test") if
                                      ind.get("n_trades_test") is not None else 0),
                    "total_trades": ind.get("total_trades", 0),
                    # Carry the VAL-ONLY-fold marker so readers can suppress test_*.
                    "test_invalid": bool(ind.get("test_invalid", False)),
                    # Template-level persistence (F4: behavioral matching)
                    "folds_seen": tp.get("folds_seen", []),
                    "folds_positive": tp.get("folds_positive", 0),
                    "per_fold_val_sharpe": tp.get("per_fold_val_sharpe", {}),
                    "per_fold_trades": tp.get("per_fold_trades", {}),
                    "mean_jaccard": tp.get("mean_jaccard", 0.0),
                    "structurally_similar": tp.get("structurally_similar", False),
                    "template_name": template,
                    # BLOCKER-2 fix (2026-06-01 audit): carry the DSR return moments
                    # and per-regime Sharpes into the survivor dict. Without these,
                    # _apply_proxy_survival_gates' G1 (which reads s.get("val_return_
                    # skew")/("val_return_kurtosis")) silently defaulted to skew=0 /
                    # excess-kurt=0 for EVERY cross-fold candidate — the real non-
                    # normality correction was never applied. regime_sharpes likewise
                    # fed G5 (downstream); absent ⇒ G5 always passed-by-default.
                    "val_return_skew": ind.get("val_return_skew", 0.0),
                    "val_return_kurtosis": ind.get("val_return_kurtosis", 0.0),
                    "regime_sharpes": ind.get("regime_sharpes"),
                    # BLOCKER fix (2026-06-01 audit): carry the G11 transferability
                    # annotation. Without it the cross-fold survival gate G11 always
                    # read None and silently no-op'd even with mc_robustness_enabled.
                    "mc_median_sharpe": ind.get("mc_median_sharpe"),
                    # delta_tree carried so downstream codegen sees Level-B structure.
                    "delta_tree": ind.get("delta_tree"),
                    # stop_mult carried so downstream L3 codegen emits the evolved
                    # stop (proxy↔QC parity). 0.0 = hold-to-expiry.
                    "stop_mult": ind.get("stop_mult"),
                    # HIGH-5 (2026-06-01 audit): val drawdown inputs for the
                    # drawdown-aware G9 champion sort. Without these the deploy
                    # selection sorted on raw val_sharpe, ignoring the drawdown the
                    # evolution penalty already shaped the front against.
                    "val_max_drawdown": ind.get("val_max_drawdown", 0.0),
                    "val_avg_position_size": ind.get("val_avg_position_size", 0.0),
                })

    # Sort: template persistence first, then individual val_sharpe
    results.sort(key=lambda x: (x["folds_positive"], x["val_sharpe"]), reverse=True)
    return results


def mc_noise_robustness_gate(
    strategies: List[Dict],
    data: "pd.DataFrame",
    template_map: Dict[str, "Template"],
    terminal_data: Dict[str, np.ndarray],
    n_realizations: int = 50,
    noise_scale: float = 1.0,
    warmup_bars: int = 30,
    cost_multiplier: float = 1.0,
) -> List[Dict]:
    """Post-GP Monte Carlo noise robustness gate.

    Re-scores each strategy on `n_realizations` independent noise
    realizations. Attaches `mc_median_sharpe` and `mc_sharpe_std` to
    each strategy dict. Does NOT filter — caller decides threshold.

    This replaces during-evolution noise injection, which destroyed
    NSGA-III convergence by changing the fitness landscape every
    generation. Post-hoc MC testing achieves the same robustness
    validation without interfering with the search process.

    Args:
        strategies: List of strategy dicts (must have entry_tree, exit_tree,
            size_tree, template_name, and optionally delta_tree as s-expressions).
        data: Training DataFrame for backtesting.
        template_map: Dict mapping template_name → Template object.
        terminal_data: Clean (noise-free) terminal data dict.
        n_realizations: Number of noise realizations per strategy.
        noise_scale: Multiplier on calibrated noise sigmas.
        warmup_bars: Backtester warmup.
        cost_multiplier: Cost scaling factor.

    Returns:
        Same list with mc_median_sharpe and mc_sharpe_std attached.
    """
    from layer2.evaluator_vectorized import vectorized_backtest, _add_terminal_noise
    from layer2.grammar import from_sexpr

    for s in strategies:
        tname = s.get("template_name", "")
        template = template_map.get(tname)
        if template is None:
            s["mc_median_sharpe"] = None
            s["mc_sharpe_std"] = None
            continue
        entry = from_sexpr(s["entry_tree"])
        exit_ = from_sexpr(s["exit_tree"])
        size = from_sexpr(s["size_tree"])
        delta = from_sexpr(s["delta_tree"]) if s.get("delta_tree") else None
        # MC re-score must use the strategy's evolved stop (else robustness Sharpe
        # diverges from the fitness it was selected on). Absent ⇒ backtester default.
        _stop_kw = ({"stop_loss_credit_multiple": float(s["stop_mult"])}
                    if s.get("stop_mult") is not None else {})

        sharpes = []
        for i in range(n_realizations):
            rng = np.random.RandomState(42 + i)
            noisy_td = _add_terminal_noise(terminal_data, rng, noise_scale)
            try:
                result = vectorized_backtest(
                    entry, exit_, size, data, template,
                    delta_tree=delta,
                    terminal_data=noisy_td,
                    warmup_bars=warmup_bars,
                    cost_multiplier=cost_multiplier,
                    **_stop_kw,
                )
                if result.sharpe > -1e5:
                    sharpes.append(result.sharpe)
            except Exception:
                pass

        if sharpes:
            s["mc_median_sharpe"] = round(float(np.median(sharpes)), 6)
            s["mc_sharpe_std"] = round(float(np.std(sharpes)), 6)
            s["mc_n_valid"] = len(sharpes)
        else:
            s["mc_median_sharpe"] = None
            s["mc_sharpe_std"] = None
            s["mc_n_valid"] = 0

    return strategies


def _apply_proxy_survival_gates(strategies: List[Dict],
                                n_trials: int = 0,
                                n_val_days: int = 125) -> List[Dict]:
    """Apply gates G1, G2, G3 that can be checked from proxy data alone.

    G4 (max drawdown), G5 (regime), G6 (proxy-QC gap), G7 (QC test Sharpe),
    G8 (profit factor) require per-trade PnL data or QC backtests and are
    applied downstream.

    Args:
        n_trials: total strategies evaluated across all templates and folds.
            Used for DSR-based G1 threshold. If 0, falls back to static 0.30.
        n_val_days: number of validation-period trading days.
    """
    # G1: cross-fold Deflated Sharpe Ratio gate (Bailey & Lopez de Prado 2014),
    # repointed to layer2/pbo.py (single DSR implementation, P1-B). Computes
    # P(SR > SR*) per strategy in DAILY units — the legacy in-module helpers
    # (deleted) compared an ANNUALIZED Sharpe against a z-scale SR* with V≡1,
    # which was units-buggy and silently over-strict. SR* here is the expected
    # maximum DAILY Sharpe of N trials with V = the population variance of the
    # candidates' DAILY val Sharpes (the trial-dispersion available at this
    # cross-fold stage). The per-template front gate (dsr_gate_evolution) is the
    # primary DSR filter; this is the secondary cross-fold survivor screen.
    # Keep-threshold semantics unchanged: keep when DSR >= 0.05.
    # Fallback: if N unknown (n_trials==0), use the static val_sharpe threshold.
    from layer2.pbo import (
        deflated_sharpe_ratio as _pbo_dsr,
        expected_max_sharpe as _pbo_emax,
        TRADING_DAYS_PER_YEAR as _TDY,
    )
    _DSR_SIGNIFICANCE = 0.05  # keep when P(SR>SR*) >= 0.05
    _sqrt_year = math.sqrt(_TDY)
    _use_per_strategy_dsr = n_trials > 0
    if _use_per_strategy_dsr:
        # V from the candidates' DAILY val Sharpes (ddof=0). < 2 candidates → 0.
        _daily_vals = np.array(
            [float(s.get("val_sharpe", 0.0)) / _sqrt_year for s in strategies],
            dtype=np.float64,
        )
        _g1_V = float(np.var(_daily_vals, ddof=0)) if len(_daily_vals) > 1 else 0.0
        # H1 fix (review): SR* = expected max of the SAME population whose
        # variance V we measured — the cross-fold candidate POOL — NOT the
        # full-search N = pop×templates. expected_max_sharpe(N, V) is only valid
        # when N and V describe one population; pairing N=1280 with the pool's
        # (compressed, post-selection) V mixed two populations and re-introduced
        # a milder variance-inversion. The full-search multiple-testing inflation
        # is ALREADY absorbed upstream by the per-template front DSR gate
        # (dsr_gate_evolution, N≈distinct evo trials); G1 is the SECONDARY screen
        # over the survivors it chooses among, so N_eff = the candidate-pool size
        # is the correct, consistent trial count here. (n_trials is retained only
        # as the per-strategy-vs-static enable toggle + provenance.)
        _n_eff = len(strategies)
        _sr_star = _pbo_emax(_n_eff, _g1_V)  # DAILY units, consistent (N_eff, V)
        print(f"  G1 DSR: per-strategy test (N_eff={_n_eff} cross-fold "
              f"candidates; full-search N={n_trials} handled upstream by the "
              f"front gate; T={n_val_days}, V={_g1_V:.4g}, "
              f"SR*_daily={_sr_star:.4f})")
    else:
        g1_threshold = SURVIVAL_GATES["G1_haircut_val_sharpe"]
        print(f"  G1 DSR: static threshold={g1_threshold:.3f} (N unknown)")

    # ANNOTATE-DON'T-DESTROY (2026-06-01, USER DECISION): the GP run must ALWAYS
    # yield the FULL Pareto front (champions) for inspection — never an empty fleet
    # from a hard survival filter. Each individual is TAGGED with which gates it
    # passes (positivity_passed / g3_passed / g10_passed / g11_passed + the existing
    # g2_persisted / val_dsr_passed) and a COMPOSITE survival_passed = the strict set
    # a CONFIRMATORY run would filter on. NOTHING is dropped here: `front` below is
    # the full input fleet, every member annotated. A GATE FUNNEL is printed so the
    # operator SEES how many individuals pass each gate without the fleet emptying
    # (mirrors the folds_positive histogram added for the annotate-only G2).
    front = []
    for s in strategies:
        # G1: Deflated Sharpe Ratio gate (DAILY units via pbo) — ANNOTATE-ONLY.
        if _use_per_strategy_dsr:
            _skew = float(s.get("val_return_skew", 0.0))
            _kurt = float(s.get("val_return_kurtosis", 0.0)) + 3.0  # excess→non-excess
            _dsr = _pbo_dsr(
                sharpe_daily=float(s["val_sharpe"]) / _sqrt_year,
                n_days=int(n_val_days),
                skew=_skew,
                kurtosis=_kurt,
                sr_star=_sr_star,
            )
            s["val_dsr"] = round(_dsr, 6)  # attach for downstream reporting
            # val_dsr_passed (NOT dsr_passed) — the cross-fold VAL DSR significance
            # flag, kept distinct from the per-template front gate's TRAIN-based
            # dsr_passed so the two annotations never clobber one another.
            s["val_dsr_passed"] = bool(_dsr >= _DSR_SIGNIFICANCE)
        # BLOCKER-2 (2026-06-01 audit): the cross-fold G1 DSR is ANNOTATE-ONLY
        # (attach val_dsr/val_dsr_passed), consistent with the per-template front DSR
        # gate. The remaining G1 signal is the POSITIVITY guard: a deflated-Sharpe
        # gate must never CLAIM a NEGATIVE observed Sharpe is a deployable edge. Under
        # annotate-don't-destroy this too becomes a TAG (positivity_passed), not a
        # drop — the full fleet is kept so the operator can inspect even the
        # negative-Sharpe candidates; survival_passed (below) carries the strict
        # filter a confirmatory run would apply. positivity_passed := val_sharpe > 0
        # in BOTH branches (the static g1_threshold is recorded separately for the
        # N-unknown fallback but no longer drops anyone).
        _val_sharpe = float(s.get("val_sharpe", 0.0))
        positivity_passed = bool(_val_sharpe > 0.0)
        if not _use_per_strategy_dsr:
            # N-unknown provenance: also record the static-haircut pass for reporting.
            s["g1_static_haircut_passed"] = bool(_val_sharpe >= g1_threshold)
        s["positivity_passed"] = positivity_passed

        # G2: cross-fold persistence — ANNOTATE-DON'T-DESTROY (BLOCKER-1, 2026-06-01
        # audit). "Persists" = the SAME canonical strategy key positive in ≥2 folds.
        # But folds are INDEPENDENTLY evolved, so a byte-identical tree-quad re-
        # emerging across folds is ~measure-zero → a HARD G2 filter rejects ~every
        # candidate and EMPTIES THE FLEET after a ~12-day run (the operator could
        # not then tell "no edge" from "undiscovered bug"). Worse, the anchored-
        # EXPANDING walk-forward folds share the val→train boundary (F2 trains on
        # F1's val window), so cross-fold persistence is NON-INDEPENDENT confirmation
        # — SUGGESTIVE, not a significance test (see _compute_cross_fold_persistence
        # ~:976-978). We therefore COMPUTE folds_positive and attach it as metadata
        # (g2_persisted) but do NOT drop individuals that fail it; the full fleet is
        # kept. A confirmatory/viability run can filter on g2_persisted post-hoc; the
        # OOS evidence is the front-DSR annotation + the F4 hold-out val. A
        # folds_positive histogram is printed into the summary so persistence is
        # VISIBLE without it emptying the fleet.
        s["folds_positive"] = int(s.get("folds_positive", 0))
        s["g2_persisted"] = bool(
            s["folds_positive"] >= SURVIVAL_GATES["G2_min_folds_positive"]
            and s.get("structurally_similar", False))

        # G3: This individual's val trades >= G3_min_trades_val — ANNOTATE-ONLY.
        g3_passed = bool(s["n_trades_val"] >= SURVIVAL_GATES["G3_min_trades_val"])
        s["g3_passed"] = g3_passed

        # G10: Walk-forward efficiency — val_sharpe / train_sharpe >= 0.3 — ANNOTATE-
        # ONLY. Rejects strategies that lose >70% of in-sample performance OOS.
        # Literature: Pardo (2008), WFE < 0.30 indicates severe overfitting.
        # g10_passed semantics PRESERVED from the prior hard filter:
        # * train_sharpe > 0.1 -> g10_passed = (val/train >= 0.3) [WFE applies]
        # * train_sharpe < 0.0 and val > 0 -> g10_passed = False [overfit-to-val]
        # * gray zone 0<=train<=0.1, or any other case -> g10_passed = True (N/A pass)
        # The `< 0.0` (NOT `<= 0.0`) boundary is deliberate: train_sharpe == 0.0 is the
        # absent-field / errored-backtest / breakeven sentinel, so it must NOT be
        # treated as overfit. _g10_applicable records whether WFE was the deciding
        # test (drives the "G10" tag in gates_passed, exactly as before).
        train_sharpe = s.get("train_sharpe", 0.0)
        _g10_applicable = train_sharpe > 0.1
        if _g10_applicable:
            wfe = s["val_sharpe"] / train_sharpe
            g10_passed = bool(wfe >= 0.3)
        elif train_sharpe < 0.0 and s["val_sharpe"] > 0.0:
            # NEGATIVE in-sample + positive OOS = textbook overfit-to-val (val luck).
            g10_passed = False
        else:
            # WFE undefined / not applicable (gray zone, breakeven, no-data) -> pass.
            g10_passed = True
        s["g10_passed"] = g10_passed

        # G11 (transferability): the edge must SURVIVE calibrated terminal noise —
        # ANNOTATE-ONLY. QC's live option-chain terminals (IV, spread) are noisier
        # than the History()-built training parquet (measured proxy↔QC gap). A
        # strategy whose edge evaporates under that noise is a clean-data artifact
        # that will not transfer. mc_median_sharpe is annotated by the post-evolution
        # mc_noise_robustness_gate when mc_robustness_enabled; None means it was NOT
        # run (discovery mode) -> NOT APPLICABLE -> g11_passed = True (no-op, never a
        # drop). When present, g11_passed = (median Sharpe under noise > floor).
        _mc = s.get("mc_median_sharpe")
        _g11_applicable = _mc is not None
        if _g11_applicable:
            g11_passed = bool(_mc > SURVIVAL_GATES["G11_min_mc_median_sharpe"])
        else:
            g11_passed = True  # not run -> not applicable -> passes (NA)
        s["g11_passed"] = g11_passed

        # COMPOSITE survival flag — the STRICT set a confirmatory run would filter
        # on. G2 (persistence) and the DSR significance are ADVISORY annotations and
        # are deliberately NOT part of survival_passed (they would empty the fleet);
        # see g2_persisted / val_dsr_passed for those.
        s["survival_passed"] = bool(
            positivity_passed and g3_passed and g10_passed and g11_passed)

        # gates_passed reflects which gates this individual ACTUALLY passed. G1 here
        # is the positivity tag (its DSR is annotate-only, see val_dsr_passed). G2 is
        # advisory only. G10/G11 are tagged only when APPLICABLE and passed (so the
        # tag means "passed an applicable WFE / noise-robustness test", not "skipped").
        _gates = []
        if positivity_passed:
            _gates.append("G1")
        if s.get("g2_persisted"):
            _gates.append("G2_advisory")
        if g3_passed:
            _gates.append("G3")
        if _g10_applicable and g10_passed:
            _gates.append("G10")
        if _g11_applicable and g11_passed:
            _gates.append("G11")
        s["gates_passed"] = _gates
        front.append(s)  # ANNOTATE-DON'T-DESTROY: keep EVERY individual, tagged.

    # BLOCKER-1 (2026-06-01 audit): folds_positive histogram over the FULL input
    # fleet — makes cross-fold persistence VISIBLE without G2 emptying the fleet.
    # 0 = never positive (or single-fold candidate), k = same canonical strategy
    # positive in k folds. Over INDEPENDENTLY-evolved folds, mass at k>=2 is rare
    # and advisory (anchored-expanding folds are non-independent), so this is a
    # diagnostic, not a pass/fail signal.
    _fp_hist: Dict[int, int] = {}
    for _s in strategies:
        _fp = int(_s.get("folds_positive", 0))
        _fp_hist[_fp] = _fp_hist.get(_fp, 0) + 1
    _n_persisted = sum(c for fp, c in _fp_hist.items()
                       if fp >= SURVIVAL_GATES["G2_min_folds_positive"])
    print("  folds_positive histogram (advisory; G2 is annotate-only): "
          + ", ".join(f"{fp}f→{_fp_hist[fp]}" for fp in sorted(_fp_hist))
          + f"  | {_n_persisted}/{len(strategies)} persist in ≥"
          f"{SURVIVAL_GATES['G2_min_folds_positive']} folds")

    # G9: Portfolio diversification — at most 1 strategy per correlation group —
    # ANNOTATE-DON'T-DESTROY. Prevents selecting IC + BPC + BCC (which = 2 correlated
    # IC bets). Previously this DROPPED all but the per-group winner; under annotate-
    # don't-destroy it instead TAGS the per-group champion (g9_passed=True, "G9" in
    # gates_passed) and KEEPS every other member (g9_passed=False). The G9 champion
    # is chosen among the survival_passed members of the group (the deployable set);
    # if a group has no survival_passed member, none is G9-tagged.
    #
    # HIGH-5 (2026-06-01 audit): champion selection ranks on the DRAWDOWN-ADJUSTED
    # val_sharpe (same penalty as the evolution objective, fitness.py:1132/1140), so
    # a high-Sharpe / high-drawdown individual cannot beat a near-equal-Sharpe /
    # low-drawdown sibling. raw val_sharpe is retained for reporting / G1 / WFE.
    from layer2.fitness import DRAWDOWN_FREE_LEVEL, DRAWDOWN_PENALTY_WEIGHT

    def _dd_adjusted_val_sharpe(x: dict) -> float:
        _vs = float(x.get("val_sharpe", 0.0))
        _adj_dd = float(x.get("val_max_drawdown", 0.0)) / max(
            float(x.get("val_avg_position_size", 0.0)), 0.05)
        return _vs - DRAWDOWN_PENALTY_WEIGHT * max(_adj_dd - DRAWDOWN_FREE_LEVEL, 0.0)

    # default every member to g9_passed=False; promote one champion per group below.
    for s in front:
        s["g9_passed"] = False
    _g9_seen: set = set()
    for s in sorted(front, key=_dd_adjusted_val_sharpe, reverse=True):
        if not s.get("survival_passed"):
            continue  # only the deployable set competes for the per-group champion slot
        group = _template_group(s.get("template_name", ""))
        if group not in _g9_seen:
            _g9_seen.add(group)
            s["g9_passed"] = True
            s["gates_passed"].append("G9")

    # G5: Per-regime Sharpe — ANNOTATE-DON'T-DESTROY. Previously DROPPED strategies
    # whose worst regime Sharpe fell below the threshold; now TAGS g5_passed and
    # keeps the full fleet. Requires regime_sharpes (Conditions with PredRegime);
    # when absent, g5_passed = True (no regime data -> pass by default, as before).
    g5_threshold = SURVIVAL_GATES["G5_min_regime_sharpe"]
    for s in front:
        regime_sharpes = s.get("regime_sharpes")
        if regime_sharpes is None:
            _g5_ok = True  # no regime data -> pass by default
        else:
            worst_regime = min(regime_sharpes.values()) if regime_sharpes else 0.0
            _g5_ok = bool(worst_regime >= g5_threshold)
        s["g5_passed"] = _g5_ok
        if _g5_ok:
            s["gates_passed"].append("G5")

    # GATE FUNNEL — counts over the FULL front so the operator SEES the funnel
    # WITHOUT it emptying the fleet (annotate-don't-destroy). Each count is the
    # number of individuals that pass that gate's per-individual annotation; the
    # full front (every member tagged) is what is returned and serialized.
    _N = len(front)
    _c_pos = sum(1 for s in front if s.get("positivity_passed"))
    _c_g3 = sum(1 for s in front if s.get("g3_passed"))
    _c_g10 = sum(1 for s in front if s.get("g10_passed"))
    _c_g11 = sum(1 for s in front if s.get("g11_passed"))
    _c_g9 = sum(1 for s in front if s.get("g9_passed"))
    _c_g5 = sum(1 for s in front if s.get("g5_passed"))
    _c_g2 = sum(1 for s in front if s.get("g2_persisted"))
    _c_surv = sum(1 for s in front if s.get("survival_passed"))
    print(
        "\n  GATE FUNNEL (ANNOTATE-DON'T-DESTROY — full front KEPT, nothing dropped):\n"
        f"    total front            : {_N}\n"
        f"    positivity_passed      : {_c_pos}/{_N}\n"
        f"    g3_passed (>= {SURVIVAL_GATES['G3_min_trades_val']} trades): {_c_g3}/{_N}\n"
        f"    g10_passed (WFE)       : {_c_g10}/{_N}\n"
        f"    g11_passed (noise)     : {_c_g11}/{_N}\n"
        f"    g9_passed (1/group)    : {_c_g9}/{_N}\n"
        f"    g5_passed (regime)     : {_c_g5}/{_N}\n"
        f"    --> survival_passed    : {_c_surv}/{_N}  "
        "(=positivity AND g3 AND g10 AND g11; the strict set a CONFIRMATORY run "
        "would filter on)\n"
        f"    g2_persisted (advisory): {_c_g2}/{_N}\n"
        "    NOTE: G1 DSR (val_dsr_passed) + G2 persistence are ANNOTATE-ONLY and "
        "NOT part of survival_passed.")
    return front  # FULL front, every member tagged — never an empty fleet.


# ---------------------------------------------------------------------------
# Per-template result + run-level result
# ---------------------------------------------------------------------------

@dataclass
class TemplateRunResult:
    template_name: str
    final_population_size: int
    pareto_front_size: int
    pareto_front: List[Dict]               # serializable: tree s-exprs + fitness
    metrics_log: List[Dict]                # per-generation metrics
    fitness_cache_stats: Dict
    wall_time_s: float
    # Audit trail (Model QA Finding 1 on Commit 4): record the seed trees
    # as ACTUALLY USED in this run — after any condition-specific
    # substitution. Keyed by slot ("entry"/"exit"/"size"). Under
    # scalar-only, this may differ from the template's declared seed
    # (e.g. PredRV15 → RealizedVol30m). Holders of results.json can
    # reconstruct the initial-population seed trees without re-executing
    # the substitution helper.
    seed_trees_original: Dict[str, str] = field(default_factory=dict)
    seed_trees_effective: Dict[str, str] = field(default_factory=dict)
    # Commit 6.5: H2 primary outcome infrastructure. val_hypervolume is the
    # pre-registered scalar per (condition, seed, template) cell used by
    # the Mann-Whitney U statistical test. test_hypervolume is the held-out
    # sanity check (reported but not part of the primary test).
    val_hypervolume: Optional[float] = None
    test_hypervolume: Optional[float] = None
    hv_reference_point: Optional[List[float]] = None


@dataclass
class ExperimentResult:
    config: Dict                           # ExperimentConfig as dict
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    template_results: List[TemplateRunResult]
    total_wall_time_s: float


# ---------------------------------------------------------------------------
# Per-fold seed re-derivation (P1-A: look-ahead fix for the Level-B base-template
# entry seeds, which were grid-searched on the FULL dataset incl. test windows).
# The candidate POOL is a fixed hypothesis space (not data-derived -> not a leak);
# the CHOICE among candidates is made per-fold on TRAIN data only. Mirrors the
# per-fold probe refit. See (internal doc).
# ---------------------------------------------------------------------------

# ITEM #5 (docs/viable_run_plan_2026_06_03.md): the compound VRP entry threshold
# quantile, derived per-fold on TRAIN-only normalized terminals. θ = the 75th
# percentile of the per-fold robust-normalized terminal (top quartile of IV-days /
# premium-rich days). JUSTIFICATION (measured, not guessed): the +1.16-frictionless
# probe used ~1σ ATM_IV and ~0.5σ IVRVGap5d (test_entry_cost_double_count.py); on
# fold-1 train those σ-thresholds sit at the 78.6th (IV) and 70.6th (gap) percentile
# of the normalized terminals, i.e. ~the top third — the 75th percentile is the
# common midpoint that reproduces the probe's selectivity (114 trades vs the probe's
# 114) and gave the BEST train Sharpe among {0.66, 0.70, 0.75} swept through the
# production evaluator (q75 dominated on both fold-1 and fold-2). Per-fold derivation
# uses the SAME quantile on each fold's own train data → no full-dataset leak.
VRP_SEED_QUANTILE: float = 0.75


def _norm_terminal_quantile(train_terminal_data, name: str, q: float):
    """The q-quantile of a NORMALIZED terminal array on this fold's train data.

    Returns None if the terminal is absent / not a finite 1-D array — the caller
    then skips the data-derived VRP candidate (falling back to the static pool)."""
    import numpy as np
    arr = train_terminal_data.get(name) if train_terminal_data else None
    if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 1:
        return None
    a = arr.astype(np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return None
    return float(np.percentile(a, q * 100.0))


def derive_vrp_compound_seed(train_terminal_data, q: float = VRP_SEED_QUANTILE):
    """Build the compound VRP entry seed AND(GT(ATM_IV,θ1), GT(IVRVGap5d,θ2)) with
    θ1, θ2 = the q-quantile of each terminal's NORMALIZED train array (ITEM #5).

    Look-ahead-safe: thresholds come ONLY from the fold's train terminal data
    (already normalized with the fold's TRAIN-only robust stats). Returns None if
    either terminal is unavailable, so the caller falls back to the static pool /
    template seed."""
    from layer2.templates import _vrp_compound_entry
    t1 = _norm_terminal_quantile(train_terminal_data, "ATM_IV", q)
    t2 = _norm_terminal_quantile(train_terminal_data, "IVRVGap5d", q)
    if t1 is None or t2 is None:
        return None
    return _vrp_compound_entry(iv_threshold_norm=t1, gap_threshold_norm=t2)


def candidate_entry_seeds(train_terminal_data=None,
                          vrp_quantile: float = VRP_SEED_QUANTILE) -> List["Node"]:
    """Pool of simple entry-condition seeds, built from the same terminals the
    2026-05-19 grid search used (ATM_IV, SessionReturn, RealizedVol30m) at the
    median/zero threshold (EphReal(0.0) in normalized space). Returns BOOL trees.

    ITEM #5: when `train_terminal_data` is provided, ALSO append the data-derived
    compound VRP candidate AND(GT(ATM_IV,θ1), GT(IVRVGap5d,θ2)) whose θ are the
    per-fold `vrp_quantile` of the NORMALIZED terminals (no leak — train-only). The
    pool is otherwise a FIXED hypothesis space (like the grammar); per-fold SELECTION
    among the pool (derive_per_fold_entry_seed) removes the full-data leak in the
    hard-coded seeds. The static-threshold members carry no look-ahead; the VRP
    member's thresholds are train-derived (the candidate STRUCTURE is still fixed)."""
    from layer2.templates import _real_term, _ephemeral_real, _gt, _lt, _and
    terms = ("ATM_IV", "SessionReturn", "RealizedVol30m")
    singles = [op(_real_term(t), _ephemeral_real(0.0))
               for t in terms for op in (_gt, _lt)]
    pairs = [_and(ivop(_real_term("ATM_IV"), _ephemeral_real(0.0)),
                  srop(_real_term("SessionReturn"), _ephemeral_real(0.0)))
             for ivop in (_gt, _lt) for srop in (_gt, _lt)]
    pool = singles + pairs  # 6 singles + 4 IV*SR pairs = 10
    if train_terminal_data is not None:
        vrp = derive_vrp_compound_seed(train_terminal_data, q=vrp_quantile)
        if vrp is not None:
            pool.append(vrp)  # +1 data-derived compound VRP candidate (ITEM #5)
    return pool


def derive_per_fold_entry_seed(template, train_data, train_terminal_data,
                               min_trades: int = 20):
    """Re-select a base template's entry seed using ONLY this fold's train window.

    Scores each candidate via the proxy backtester on TRAIN data (exit/size/delta
    taken from the template), returns the best-Sharpe candidate that clears
    min_trades. The candidate pool includes the data-derived compound VRP seed
    (ITEM #5) when train_terminal_data is available. Falls back to the template's
    existing entry_seed if none qualify. Look-ahead-safe: only train_data /
    train_terminal_data are used.
    """
    from layer2.evaluator_vectorized import vectorized_backtest
    best_tree, best_score = template.entry_seed, -1e9
    for cand in candidate_entry_seeds(train_terminal_data=train_terminal_data):
        try:
            r = vectorized_backtest(
                cand, template.exit_seed, template.size_seed, train_data, template,
                delta_tree=template.delta_seed, terminal_data=train_terminal_data,
            )
        except Exception:
            continue
        if getattr(r, "total_trades", 0) >= min_trades and r.sharpe > best_score:
            best_score, best_tree = r.sharpe, cand
    return best_tree


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    *,
    _preloaded_data: Optional[tuple] = None,
    _precomputed_l1_sha256: Optional[str] = None,
) -> ExperimentResult:
    """Run a complete L2 GP experiment: load → split → shuffle (if needed) →
    evolve per template → aggregate Pareto fronts → save.

    Args:
        config: Experiment configuration.
        _preloaded_data: If set, (DataFrame, schema) tuple to use instead of
            loading from config.l1_parquet_path. Used by run_walk_forward()
            for per-fold probe refit.
        _precomputed_l1_sha256: SHA256 of the original Parquet file, computed
            once by the caller. Used for fingerprinting when _preloaded_data
            is provided.
    """
    t_start = time.time()
    print(f"=== L2 experiment: condition={config.condition} seed={config.seed} ===")

    # Grammar-size drift guard: fail BEFORE any evolution work happens if the
    # live grammar has drifted from the H2 pre-registration. A grammar refactor
    # would silently change what's being pre-registered; catching it at
    # experiment init makes the drift a clear, actionable error.
    # Skip for vectorized mode: minute Parquet is scalar-only, grammar
    # assertion is for the full-grammar H2 design.
    # P1: the H2 pre-registration is VOIDED ((internal doc)),
    # and the live grammar has legitimately gained one terminal since it was written
    # (157 functions, 52 terminals — verified: build_terminal_set() has 52 unique
    # names, no duplicates). Enforcing a voided pre-reg's stale count is wrong and
    # tampering with the retained-for-audit doc is worse, so enforcement is SOFT
    # (warn, don't raise) until a FRESH pre-registration is filed for the H2 re-run.
    # The fresh pre-reg must lock the live grammar (and per-condition scalar-only
    # counts), after which this should be reverted to a hard assertion.
    if not config.use_vectorized:
        try:
            assert_grammar_matches_preregistration()
        except AssertionError as _drift:
            import warnings
            warnings.warn(
                f"Grammar drift vs the VOIDED H2 pre-registration (soft enforcement "
                f"until a fresh pre-reg is filed — P1): {_drift}",
                stacklevel=2,
            )

    # Load + validate L1 Parquet. TOCTOU fix (Model QA R1 hardening):
    # compute the Parquet SHA256 AT LOAD TIME, not at save time.
    _l1_parquet_path_abs = Path(config.l1_parquet_path).resolve()
    if _preloaded_data is not None:
        df, schema = _preloaded_data
        _l1_parquet_sha256 = _precomputed_l1_sha256 or _file_sha256(_l1_parquet_path_abs)
        _l1_parquet_size = _l1_parquet_path_abs.stat().st_size
        print(f"\n[1/4] Using pre-loaded data ({len(df)} rows, per-fold probes)")
    else:
        print(f"\n[1/4] Loading {'minute' if config.use_vectorized else 'L1'} "
              f"Parquet: {config.l1_parquet_path}")
        _l1_parquet_sha256 = _file_sha256(_l1_parquet_path_abs)
        _l1_parquet_size   = _l1_parquet_path_abs.stat().st_size
        if config.use_vectorized:
            df, schema = load_minute_parquet(config.l1_parquet_path)
        else:
            df, schema = load_l1_parquet(config.l1_parquet_path)
    print(f"  schema: {schema.n_rows} rows, {schema.n_dates} dates, "
          f"vector_dim={schema.typed_vector_dim}")
    print(f"  l1_parquet_sha256: {_l1_parquet_sha256[:16]}... "
          f"({_l1_parquet_size:,} bytes)")

    # Split by date
    print(f"\n[2/4] Splitting train / val / test (train_end={config.train_end})")
    splits = split_by_date(
        df, train_end=config.train_end, val_end=config.val_end,
        embargo_days=config.embargo_days,
        train_start=config.train_start, test_end=config.test_end,
    )
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(f"  train={n_train} val={n_val} test={n_test}")

    # Hardening #8: minimum training rows (prevent meaningless GP on thin data)
    _min_train_rows = getattr(config, 'min_train_rows', 1000)
    if config.use_vectorized and n_train < _min_train_rows and _min_train_rows > 0:
        raise RuntimeError(
            f"Training split has only {n_train} rows (need >= {_min_train_rows} for minute-resolution GP). "
            f"Check Parquet date coverage vs fold train_end={config.train_end}."
        )

    # Fold-shape fail-fast (2026-06-01, run-operability sweep BLOCKER-2): validate
    # DISTINCT val/test TRADING DAYS, not just rows. A degenerate split (e.g. fold-4's
    # 1-day test after the embargo consumes the tail) silently reports garbage val/test
    # Sharpe/hypervolume on a multi-day run. Val is load-bearing (the survival gates
    # select on it) → RAISE; test is secondary reporting → WARN. Catches the issue
    # BEFORE compute, not after days.
    if config.use_vectorized:
        _MIN_VAL_DAYS, _MIN_TEST_DAYS = 20, 20
        _n_val_days = int(splits["val"]["date"].nunique()) if n_val else 0
        _n_test_days = int(splits["test"]["date"].nunique()) if n_test else 0
        if config.val_end is not None and _n_val_days < _MIN_VAL_DAYS:
            raise RuntimeError(
                f"Validation split has only {_n_val_days} distinct trading days "
                f"(need >= {_MIN_VAL_DAYS}; the survival gates select on val). Check "
                f"the fold windows + embargo vs parquet coverage "
                f"(train_end={config.train_end}, val_end={config.val_end})."
            )
        if config.test_end is not None and 0 < _n_test_days < _MIN_TEST_DAYS:
            import warnings as _warnings
            _warnings.warn(
                f"Test split has only {_n_test_days} distinct trading days "
                f"(< {_MIN_TEST_DAYS}): test_sharpe/test_hv are UNRELIABLE for this "
                f"fold — do not rely on them (val-based selection is unaffected).")

    # H2 pre-reg v4: scale min_trades floor with validation length so the
    # B2 gate provides enough statistical power regardless of pilot size.
    # effective_min_trades = max(config.min_trades, ceil(COEFF × n_val_days), 3)
    # config.min_trades remains as a user-settable override — the floor
    # only RAISES it, never lowers. The mutation lands BEFORE
    # _config_fingerprint() so the fingerprint reflects the effective value
    # that actually gates fitness.
    #
    # v8 (RF-5): coefficient 0.10 → 0.05. At n_val_days=76, floor drops from
    # 8 to 4 — relaxation motivated by 3-seed smoke showing 96% of candidate
    # cells were zeroed by the v4 floor, killing statistical power. 0.05 is
    # still 33% stricter than v1's absolute min_trades=3 default. Change is
    # pre-hoc (pre-20-seed), outcome-motivated (gate miscalibration
    # observed) but NOT outcome-tuned (relaxation is symmetric across all
    # arms; not designed to favor real-l1).
    import math as _math
    # 2026-06-02 (user decision (b)): 0.05→0.03 so the per-fold FLOOR does not
    # re-raise the relaxed min_trades=20 back toward 38 on the long-train folds
    # (0.05×750≈38). At 0.03 the effective floor is ~20-23 across folds — keeps a
    # mild train-length scaling while preserving the selectivity reach.
    _MIN_TRADES_COEFF = 0.03
    # Fix #175: use TRAIN days for scaling (was val days — trivial leak of val metadata)
    _n_train_days = int(splits["train"]["date"].nunique()) if n_train else 0
    _scaled_min_trades = max(3, _math.ceil(_MIN_TRADES_COEFF * _n_train_days))
    _effective_min_trades = max(config.min_trades, _scaled_min_trades)
    if _effective_min_trades != config.min_trades:
        print(
            f"  [min_trades floor] config.min_trades={config.min_trades} "
            f"→ effective={_effective_min_trades} "
            f"(n_train_days={_n_train_days}, ceil({_MIN_TRADES_COEFF} × n_train_days)="
            f"{_scaled_min_trades})"
        )
        config.min_trades = _effective_min_trades

    # Shuffle if shuffled-l1 condition
    train_data = splits["train"]
    if config.condition == "shuffled-l1":
        print(f"\n[2b] Applying shuffled-L1 control (seed={config.shuffle_seed})")
        train_data = shuffle_l1(
            train_data, seed=config.shuffle_seed,
            block="window",  # regime_stratify=False (default) — stronger null
        )


    # Pre-compute terminal_data for vectorized evaluation (once per split).
    # This avoids re-normalizing 205K+ rows per strategy evaluation.
    _train_terminal_data = None
    _fold_norm_stats = None  # per-fold normalization to prevent look-ahead bias
    if config.use_vectorized:
        from layer2.evaluator_vectorized import prepare_terminal_data
        # Compute normalization stats from THIS fold's training data (not global frozen stats).
        # Prevents look-ahead: Fold 1 (train_end=2024-03-29) won't use stats
        # computed from data extending to 2024-09-30.
        #
        # M1 fix (audit 2026-06-02): compute_norm_stats_from_data reads raw
        # DataFrame COLUMNS, so only the ~8 base bypass scalars get a per-fold
        # stat; the 8+ SYNTHESIZED terminals (ThetaUrgency, RV5d, VIXChange,
        # SPXReturn3d, VIXMean5d, IVRVGap5d, ATM_IV_5m, RealizedVol30m_5m, …) are
        # NOT columns and silently fell back to the frozen date<=2024-09-30
        # constants — a per-fold leak (for early folds the frozen window overlaps
        # the fold's own validation span). Fix: first SYNTHESIZE the train
        # terminals un-normalized (same VIX-lag/vix_prior the normalized pass will
        # use, so the fitted stats match the series actually consumed), then fit
        # per-fold stats over THOSE arrays so every synthesized terminal gets a
        # TRAIN-only (center, scale). Base columns are in the dict too, so the
        # base-column path is preserved (and now bypasses the C4 ATM_IV
        # forward-fill / VIX lag inconsistency the column path had).
        from layer2.terminal_stats import compute_norm_stats_from_arrays
        _vix_prior_train = splits.get("vix_prior", {}).get("train", {})
        _train_synth_raw = prepare_terminal_data(
            train_data, normalize_terminals=False, vix_prior=_vix_prior_train)
        # ITEM #4 (docs/viable_run_plan_2026_06_03.md): robust=True forces median /
        # (IQR/1.349) per-fold scaling for every continuous terminal. The expanding
        # train window's 2022-bear heavy tail inflates STD ~2x (measured robust/
        # meanstd scale ratio 0.506 on fold-1), HALVING within-regime IV
        # discrimination; robust scaling recovers it (daily ATM_IV IQR gap 0.611σ
        # mean/std -> 1.207σ robust, 1.97x). Still TRAIN-only per fold (no look-
        # ahead). RevIN (Kim et al. ICLR 2022) x Huber (1981); Cont (2001) tails.
        # The SAME dict is persisted onto each Pareto record (fold_norm_stats) and
        # consumed verbatim by L3 codegen, so QC normalizes with the IDENTICAL
        # robust stats (train/serve parity, Sculley et al. 2015).
        _fold_norm_stats = compute_norm_stats_from_arrays(_train_synth_raw, robust=True)
        _norm_mode = getattr(config, "norm_mode", "expanding")
        if _norm_mode == "trailing_rolling":
            # plan #4: LEVEL terminals via causal per-day trailing robust stats; the
            # expanding _fold_norm_stats become the static-terminal stats + warmup
            # fallback. Reuses the already-synthesized raw arrays (same synthesis as
            # the normalized pass) so only the normalization step changes.
            from layer2.trailing_norm import apply_trailing_norm
            _W = int(getattr(config, "trailing_window", 20))
            print(f"\n[2c] Pre-computing terminal_data for vectorized evaluator "
                  f"({len(train_data)} rows, TRAILING-ROLLING (W={_W}d, causal) robust "
                  f"norm for LEVEL terminals + expanding fallback for the rest)")
            _train_terminal_data, _ = apply_trailing_norm(
                _train_synth_raw, train_data["date"].values, W=_W,
                static_stats=_fold_norm_stats)
        else:
            print(f"\n[2c] Pre-computing terminal_data for vectorized evaluator "
                  f"({len(train_data)} rows, per-fold ROBUST (median/IQR) normalization "
                  f"incl. synthesized terminals)")
            _train_terminal_data = prepare_terminal_data(
                train_data, norm_stats_override=_fold_norm_stats,
                vix_prior=_vix_prior_train)
        print(f"  terminals: {sorted(_train_terminal_data.keys())}")

        # Hardening #5: validate grammar terminals have data backing
        from layer2.grammar import GType
        from layer2.grammar import build_scalar_only_terminal_set as _bsots
        _grammar_real_terms = {t.name for t in _bsots()
                               if t.ret_type == GType.REAL
                               and t.name not in ("EphReal", "EphInt")}
        _missing = _grammar_real_terms - set(_train_terminal_data.keys())
        if _missing:
            import warnings
            warnings.warn(
                f"Grammar terminals with NO data (will evaluate as 0.0): {sorted(_missing)}. "
                f"GP trees using these terminals have dead branches.",
                stacklevel=2,
            )

        # Hardening #2: check for NaN in scalar terminal arrays
        import numpy as _np
        for _tname, _tarr in _train_terminal_data.items():
            if isinstance(_tarr, _np.ndarray) and _tarr.ndim == 1:
                _nan_count = int(_np.isnan(_tarr).sum())
                if _nan_count > 0:
                    import warnings
                    warnings.warn(
                        f"Terminal '{_tname}' has {_nan_count}/{len(_tarr)} NaN values "
                        f"({100*_nan_count/len(_tarr):.1f}%). These bars evaluate as 0.0.",
                        stacklevel=2,
                    )

    # Resolve template list
    if config.template_names:
        if config.level_b:
            templates = [base_template_by_name(n) for n in config.template_names]
        else:
            templates = [template_by_name(n) for n in config.template_names]
    else:
        templates = all_templates()

    # P1-A: per-fold seed re-derivation. The hard-coded Level-B base-template entry
    # seeds were grid-searched on the FULL dataset (incl. test windows) -> look-ahead
    # leak. When enabled, re-select each base template's entry seed using ONLY this
    # fold's train data (fixed candidate pool; per-fold choice). recalibrate_seed_
    # thresholds still runs downstream on the chosen seed.
    _ttd = locals().get("_train_terminal_data")
    _trd = locals().get("train_data")
    if config.per_fold_seeds and config.level_b and _ttd is not None and _trd is not None:
        for _t in templates:
            try:
                _t.entry_seed = derive_per_fold_entry_seed(_t, _trd, _ttd)
            except Exception as _seed_exc:
                print(f"  [per-fold seed] {_t.name}: kept template seed "
                      f"({type(_seed_exc).__name__}: {_seed_exc})", file=sys.stderr)

    print(f"\n[3/4] Evolving {len(templates)} templates")

    # Per-template islands
    template_results: List[TemplateRunResult] = []
    # Grammar construction per condition:
    # A (scalar-only): bypass scalars + structural — no probes, no embeddings
    # B (probes-only): probes + structural only — no bypass scalars, no embeddings ()
    # C (emb-only): embeddings + structural only — no bypass scalars, no probes ()
    # D (real-l1): full grammar (everything)
    if config.condition == "scalar-only":
        grammar = Grammar(
            functions=SCALAR_ONLY_FUNCTIONS,
            terminals=build_scalar_only_terminal_set(),
            max_depth=config.grammar_max_depth,
            max_nodes=config.grammar_max_nodes,
        )
        print(f"  [scalar-only] grammar: {len(grammar.functions)} funcs, "
              f"{len(grammar.terminals)} terms (EMB_* and probes stripped)")
    elif config.condition == "probes-only":
        grammar = Grammar(
            functions=PROBES_ONLY_FUNCTIONS,
            terminals=build_probes_only_terminal_set(),
            max_depth=config.grammar_max_depth,
            max_nodes=config.grammar_max_nodes,
        )
        print(f"  [probes-only] grammar: {len(grammar.functions)} funcs, "
              f"{len(grammar.terminals)} terms (EMB_* stripped, probes retained)")
    elif config.condition == "emb-only":
        # : Use screened grammar — removes unsupervised ops (EmbNorm/Cos/Sub/Lag)
        # and filters EmbProj to top 3 per group (BH q<0.10). Reduces functions
        # from 157 to ~48, removing search space confound from ablation.
        from layer2.grammar import build_emb_only_screened_functions
        _emb_funcs = build_emb_only_screened_functions()
        grammar = Grammar(
            functions=_emb_funcs,
            terminals=build_emb_only_terminal_set(),
            max_depth=config.grammar_max_depth,
            max_nodes=config.grammar_max_nodes,
        )
        print(f"  [emb-only] grammar: {len(grammar.functions)} funcs, "
              f"{len(grammar.terminals)} terms (L2.64 screened: unsupervised ops removed, "
              f"EmbProj filtered to top-3/group)")
    else:
        grammar = Grammar(
            max_depth=config.grammar_max_depth,
            max_nodes=config.grammar_max_nodes,
        )

    # Commit 6.6/6.8: Resume detection with config + environment integrity
    # guard. If output_dir has per-template artifacts from a partial
    # previous run, verify the fingerprint matches and environment hasn't
    # drifted; only then reuse the artifacts.
    # v8 Fix 1 (C3/R2): EAGER-load PCA bases if the grammar has EmbProj_*
    # operators. Silent-zero fallback on missing bases would silently
    # degenerate the real-l1 arm at run time — looks exactly like "GP
    # ignores EmbProj" which is the pre-v8 pathology. Fail T+0, not T+72h.
    # Skip in vectorized mode: minute Parquet is scalar-only, no EmbProj.
    from layer2.evaluator import _load_pca_bases_cached as _load_pca
    _has_embproj = any(
        f.name.startswith("EmbProj_") for f in grammar.functions
    )
    if _has_embproj and not config.use_vectorized:
        _load_pca(raise_on_missing=True)  # raises RuntimeError on missing/SHA mismatch

        # v8 Fix 2 (C7/M1): hard cross-check — the PCA basis must have been fit
        # against the same L1 checkpoint the L2 run is consuming. Otherwise the
        # EmbProj operators project onto directions from a DIFFERENT encoder
        # output distribution, silently corrupting the real-l1 arm.
        _pca_manifest_path = Path(__file__).resolve().parent / "pca_bases_manifest.json"
        if _pca_manifest_path.exists():
            try:
                _pm = json.loads(_pca_manifest_path.read_text())
                _pca_l1_sha = _pm.get("l1_checkpoint_sha256", "")
                _l1_sidecar_path = Path(
                    config.l1_parquet_path
                ).resolve().with_suffix(".provenance.json")
                _l1_sidecar_l1_sha = ""
                if _l1_sidecar_path.exists():
                    _sidecar = json.loads(_l1_sidecar_path.read_text())
                    _l1_sidecar_l1_sha = _sidecar.get("checkpoint_sha256", "")
                if _pca_l1_sha and _l1_sidecar_l1_sha and (
                    _pca_l1_sha != _l1_sidecar_l1_sha
                ):
                    raise RuntimeError(
                        f"PCA basis was fit against a DIFFERENT L1 checkpoint "
                        f"than the L2 run is using:\n"
                        f"  pca_bases_manifest.l1_checkpoint_sha256: {_pca_l1_sha}\n"
                        f"  l1_parquet.provenance.checkpoint_sha256:  "
                        f"{_l1_sidecar_l1_sha}\n"
                        f"Refit PCA bases before re-running:\n"
                        f"    python -m layer2.inference.pca_bases"
                    )
            except RuntimeError:
                raise
            except Exception:
                # Malformed manifest / sidecar — don't block the run on it
                # here; the provenance block records the read error.
                pass

    output_path = Path(config.output_dir) if config.output_dir else None
    current_fingerprint = _config_fingerprint(config, _l1_parquet_sha256)
    current_env         = _env_snapshot()
    current_grammar_sig = _grammar_signature(grammar)
    resumed_results: Dict[str, TemplateRunResult] = {}
    if output_path is not None and output_path.exists():
        # 6.8 BLOCKING fix: check the run-level fingerprint before trusting
        # any resumed template. Prevents the "operator changes seed then
        # resumes" footgun Senior Dev F3 / Model QA flagged.
        _check_resume_fingerprint(
            output_path, current_fingerprint, current_env, current_grammar_sig,
            config=config,
        )
        resumed_results = _load_completed_templates(output_path)
        if resumed_results:
            print(f"  [resume] found {len(resumed_results)} completed template(s): "
                  f"{sorted(resumed_results.keys())}")
    # Write / refresh the run fingerprint file so subsequent resume checks
    # have something to verify against.
    if output_path is not None:
        _write_run_fingerprint(
            output_path, current_fingerprint, current_env,
            current_grammar_sig, config, _l1_parquet_sha256,
        )

    # 6.8 R7: warnings channel — accumulates structured entries to be
    # serialized to results.json. Operators and auditors see failures,
    # substitutions, and empty-split events in one place.
    #
    # Commit 6.8 review fix: warnings MUST persist across resumed runs —
    # otherwise a crash after template 2 + resume produces a warnings
    # list reflecting only the this-invocation loop, silently losing
    # template-1/2 substitution events. Persist to warnings.jsonl
    # (append-only) after each warning; reload existing entries on resume.
    warnings_log: List[Dict] = []
    warnings_path = output_path / "warnings.jsonl" if output_path is not None else None
    if warnings_path is not None and warnings_path.exists():
        # Resume: reload prior warnings (de-duplicated by template+kind+slot
        # to avoid re-logging scalar-only substitutions that ran earlier).
        try:
            prior_warnings = [
                json.loads(line)
                for line in warnings_path.read_text().splitlines()
                if line.strip()
            ]
            warnings_log.extend(prior_warnings)
        except json.JSONDecodeError:
            pass  # torn write — let the rest of the run rebuild

    def _warn(entry: Dict):
        """Append a warning to both in-memory log AND append-only JSONL
        (6.8 review: no dedup — append-only journal semantics, auditor
        can collapse duplicates downstream if desired)."""
        warnings_log.append(entry)
        if warnings_path is not None:
            with warnings_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    # Only log empty-split warnings ONCE per run (idempotent) — reuse
    # prior entries if already logged, otherwise record now.
    def _already_warned(kind: str, **keys) -> bool:
        for w in warnings_log:
            if w.get("kind") == kind and all(w.get(k) == v for k, v in keys.items()):
                return True
        return False

    if len(splits["val"]) == 0 and not _already_warned("empty_split", split="val"):
        _warn({
            "kind": "empty_split", "split": "val",
            "reason": "val_end is None or no rows fall within val window",
        })
    if len(splits["test"]) == 0 and not _already_warned("empty_split", split="test"):
        _warn({
            "kind": "empty_split", "split": "test",
            "reason": "no rows fall beyond val_end",
        })

    # Sequential template evolution is load-bearing for B3 RNG isolation —
    # see gp_engine.py module docstring and _EVOLVE_INFLIGHT_LOCK. Do not
    # parallelize this loop without first threading per-template RNG
    # instances through grammar.py + gp_engine.py.
    for template in templates:
        if template.name in resumed_results:
            print(f"\n  → template: {template.name}  [SKIP — resumed from existing artifacts]")
            template_results.append(resumed_results[template.name])
            continue

        # Commit 6.6: wrap per-template body in try/except. A backtester
        # crash or evolve-loop exception on one template must NOT abort
        # the whole H2 run — log traceback + carry on to the next template.
        try:
            result = _run_one_template(
                template=template, config=config, grammar=grammar,
                train_data=train_data, splits=splits,
                train_terminal_data=_train_terminal_data,
                fold_norm_stats=_fold_norm_stats,
            )
        except KeyboardInterrupt:
            # Explicit propagate so operator can abort cleanly
            raise
        except Exception as exc:
            print(f"    [FAILED] template {template.name!r}: "
                  f"{type(exc).__name__}: {exc}")
            if output_path is not None:
                _write_failure_log(output_path, template.name, exc)
            _warn({
                "kind": "template_failure", "template": template.name,
                "exc_type": type(exc).__name__, "message": str(exc),
            })
            continue

        template_results.append(result)
        # G11 transferability annotation (post-evolution, FRONT ONLY — does NOT
        # perturb the NSGA-III search). Re-scores each front member under calibrated
        # terminal noise (the measured proxy↔QC option-chain gap) and attaches
        # mc_median_sharpe, which the cross-fold survival gate G11 filters on. OFF by
        # default; a discovery pilot sees the raw front first.
        if getattr(config, "mc_robustness_enabled", False) and result.pareto_front:
            try:
                for _s in result.pareto_front:
                    _s.setdefault("template_name", template.name)
                mc_noise_robustness_gate(
                    result.pareto_front, train_data, {template.name: template},
                    _train_terminal_data,
                    n_realizations=getattr(config, "mc_robustness_realizations", 20),
                    noise_scale=1.0,
                )
                _n_robust = sum(1 for _s in result.pareto_front
                                if (_s.get("mc_median_sharpe") or -1.0) > 0)
                print(f"    G11 mc-robustness: {_n_robust}/{len(result.pareto_front)} "
                      f"front members keep median Sharpe > 0 under terminal noise")
            except Exception as _exc:
                # FAIL-SAFE: the operator enabled G11 to FILTER. If annotation fails
                # (import/data error — distinct from the inner per-realization guard),
                # do NOT let the front pass silently unverified. Mark it rejecting so
                # G11 drops it (loud: 0 survivors from this template) rather than a
                # silent no-op. -999 (not -inf) keeps the JSONL snapshot valid.
                print(f"    [G11 mc-robustness FAILED -> fail-safe reject] "
                      f"{type(_exc).__name__}: {_exc}")
                for _s in result.pareto_front:
                    _s["mc_median_sharpe"] = -999.0
        # 6.8 R7: record scalar-only substitutions as warnings.
        # Dedup guard: if a prior run logged this substitution, don't
        # re-log on resume (would duplicate under the append-only journal).
        if config.condition == "scalar-only":
            for slot, eff in result.seed_trees_effective.items():
                orig = result.seed_trees_original.get(slot, "")
                if eff != orig and not _already_warned(
                    "scalar_only_substitution",
                    template=template.name, slot=slot,
                ):
                    _warn({
                        "kind": "scalar_only_substitution",
                        "template": template.name, "slot": slot,
                        "from_tree": orig, "to_tree": eff,
                    })
        # 6.8 telemetry restoration (Code Reviewer nit): restore
        # cache_hit_rate + wall_time_s, which Commit 6.6's refactor dropped.
        val_str = (f"{result.val_hypervolume:.4f}"
                   if result.val_hypervolume is not None else "None")
        hit_rate = result.fitness_cache_stats.get("cache_hit_rate", 0.0)
        print(f"    pareto_front_size={result.pareto_front_size}  "
              f"val_hv={val_str}  cache_hit_rate={hit_rate:.1%}  "
              f"wall={result.wall_time_s:.1f}s")

        # Commit 6.6: incremental per-template save. Write this template's
        # artifacts IMMEDIATELY so a mid-run crash doesn't lose completed
        # work. Resume detection (above) then skips it on restart.
        # Commit 6.8: stamp env_snapshot + grammar_signature at SAVE TIME
        # (not at resume time) so resumed runs don't misattribute post-
        # facto env metadata.
        if output_path is not None:
            _save_single_template_result(
                result, output_path,
                env_snapshot=current_env,
                grammar_signature=current_grammar_sig,
            )

    elapsed_this_invocation = time.time() - t_start
    # 6.8 F2 fix: total_wall_time_s must aggregate across resumed +
    # fresh templates. time.time() - t_start captures only THIS
    # invocation's wall-clock. For the H2 dissertation report, we want
    # total cumulative compute: sum per-template wall_time_s.
    total_wall = sum(tr.wall_time_s for tr in template_results)
    er = ExperimentResult(
        config=asdict(config),
        n_train_rows=n_train, n_val_rows=n_val, n_test_rows=n_test,
        template_results=template_results,
        total_wall_time_s=total_wall,
    )

    # Save if output_dir specified
    if config.output_dir:
        print(f"\n[4/4] Writing results to {config.output_dir}/")
        # Compute ref_dirs for provenance record (same values as evolve()
        # uses). Done outside evolve for the audit trail — evolve already
        # computes these internally per-template.
        from pymoo.util.ref_dirs import get_reference_directions
        _ref_dirs_for_provenance = get_reference_directions(
            "das-dennis", len(DEFAULT_OBJECTIVES), n_partitions=config.n_partitions,
        )
        _save_experiment_result(
            er, Path(config.output_dir),
            config=config, grammar=grammar, splits=splits,
            ref_dirs=_ref_dirs_for_provenance,
            l1_sha256=_l1_parquet_sha256,
            l1_size_bytes=_l1_parquet_size,
            warnings=warnings_log,
        )
    else:
        print(f"\n[4/4] (no output_dir specified — results in-memory only)")

    print(f"\n=== Done: {total_wall:.1f}s total ===")
    return er


def _apply_dsr_front_gate(
    pareto_serialized: List[Dict],
    front: List[Individual],
    *,
    train_T: int,
    evo_records: Dict,
    dsr_gate_enabled: bool,
    dsr_threshold: float,
    template_name: str,
    min_days: int = 20,
) -> Tuple[List[Dict], List[Individual]]:
    """Evolution-level Deflated-Sharpe-Ratio FILTER gate at champion selection.

    Deflates each champion's TRAINING Sharpe (``entry['train_sharpe']``) against
    SR* computed ONCE from the whole evolution's distinct-trial population
    (``evo_records``, training Sharpes). The gate FILTERS: champions failing
    ``DSR < dsr_threshold`` are removed from BOTH ``pareto_serialized`` and
    ``front`` in lock-step (index-aligned by construction). It is keyed on the
    TRAINING window ``train_T`` — below ``min_days`` the gate is INCONCLUSIVE, so
    it annotates ``dsr_passed=False`` + ``dsr_inconclusive=True`` and does NOT
    filter (MEDIUM-8, 2026-06-01: was ``dsr_passed=None`` — ambiguous; a strict
    reader idiom is now ``dsr_passed is True`` ⇒ significant, and
    ``dsr_inconclusive`` distinguishes "assessed and failed" from "could not
    assess"). NOTE: this gate is currently ANNOTATE-ONLY at the assessable path too
    (the full front is KEPT; ``dsr_passed`` flags the τ-significant subset) — see the
    annotate-don't-destroy block. Fail-open: on any gate error the front is kept
    intact (annotated inconclusive). Mutates each serialized entry in place with the
    dsr annotations. Does NOT touch NSGA-III objectives, per-individual fitness, or
    ``result.psr``.

    Extracted from ``_run_one_template`` (behavior-preserving) so the gate's
    train-Sharpe wiring is unit-testable at the boundary with controlled
    champion Sharpes — the GP integration fixtures produce only degenerate,
    all-sentinel fronts that cannot exercise Sharpe discrimination (see
    ``tests/test_experiment.py::test_dsr_front_gate_*``).

    Returns ``(kept_serialized, kept_front)``.
    """
    _dsr_assessable = (
        dsr_gate_enabled and bool(pareto_serialized) and train_T >= min_days
    )
    if dsr_gate_enabled and pareto_serialized and not _dsr_assessable:
        print(
            f"    DSR gate ({template_name}): SKIPPED — train window T_train="
            f"{train_T} < min_days={min_days}; cannot assess significance "
            f"(annotate-only, front NOT filtered).",
            file=sys.stderr,
        )
        # MEDIUM-8 (2026-06-01 audit): on an INCONCLUSIVE (short) window, set
        # dsr_passed=False (NOT None) plus a separate dsr_inconclusive=True flag.
        # CANONICAL READER IDIOM: treat `dsr_passed` as a strict boolean — a
        # candidate is τ-significant iff `dsr_passed is True`. The prior None made
        # `if dsr_passed:` (falsey) and `if dsr_passed is None:` (truthy) disagree
        # about the same record, and a downstream `dsr_passed != False` admitted
        # the inconclusive case as if it had passed. With dsr_passed=False the
        # default "significant?" check is correct (fails closed); dsr_inconclusive
        # distinguishes "assessed and failed" from "could not assess".
        for _e in pareto_serialized:
            _e["dsr"] = None
            _e["dsr_passed"] = False
            _e["dsr_inconclusive"] = True
    if not _dsr_assessable:
        return pareto_serialized, front

    from layer2.pbo import dsr_gate_evolution, evolution_n_v_from_records
    # Evolution-level (N, V) from the distinct-individual TRAINING Sharpes
    # recorded across the run (surfaced by evolve() in _val_tracking).
    _dsr_N, _dsr_V = evolution_n_v_from_records(evo_records, min_days=min_days)
    _gate_members = [
        {
            "sharpe": float(e.get("train_sharpe", 0.0)),
            "n_days": train_T,
            "skew": float(e.get("train_return_skew", 0.0)),
            "kurtosis": float(e.get("train_return_kurtosis", 0.0)),  # excess (scipy)
        }
        for e in pareto_serialized
    ]
    try:
        _dsr_res = dsr_gate_evolution(
            _gate_members,
            n_trials=_dsr_N,
            sharpe_variance=_dsr_V,
            threshold=dsr_threshold,
            min_days=min_days,
            kurtosis_is_excess=True,
        )
        for _e, _dsr, _pass in zip(
            pareto_serialized, _dsr_res["dsr"], _dsr_res["passed"]
        ):
            _e["dsr"] = round(float(_dsr), 6)
            _e["dsr_passed"] = bool(_pass)
            _e["dsr_inconclusive"] = False  # MEDIUM-8: assessed (not a short window)
            _e["dsr_sr_star"] = round(float(_dsr_res["sr_star"]), 6)
            _e["dsr_sr_star_annualized"] = round(
                float(_dsr_res["sr_star_annualized"]), 6)
            _e["dsr_n_trials"] = int(_dsr_res["n_trials"])
        print(
            f"    DSR gate ({template_name}): N={_dsr_res['n_trials']} "
            f"(distinct evo trials) V={_dsr_res['sharpe_variance']:.4g} "
            f"SR*={_dsr_res['sr_star']:.4f} (daily; "
            f"ann={_dsr_res['sr_star_annualized']:.3f}) T_train={train_T} "
            f"thr={dsr_threshold} → {_dsr_res['n_passed']}/"
            f"{len(pareto_serialized)} champions pass",
            file=sys.stderr,
        )
        # ANNOTATE-DON'T-DESTROY (2026-06-01, empty-fleet fix — discovery sweep
        # BLOCKER-1): every champion is now ANNOTATED (dsr / dsr_passed above) but the
        # FULL front is KEPT, not filtered. At production N the multiple-testing
        # SR* ≈ 1.4-2.0 annualized, so a HARD filter at any τ empties the fleet in the
        # current harsh proxy (reproduced on real data: front 60 → 0) — a multi-day run
        # would then yield ZERO output and the operator could not tell "no edge" from
        # "undiscovered bug". Persisting the full annotated front guarantees the run
        # yields the best proxy champions (for transfer investigation / L3 / diagnosis),
        # while `dsr_passed` flags the τ-significant subset for any confirmatory/
        # viability CLAIM. (User choice 2026-06-01: keep candidates; the significance
        # bar annotates, it does not destroy. A viability run filters on dsr_passed at
        # τ=0.5; the cross-fold survival gates still run downstream on the full front.)
        _n_passed = sum(1 for _e in pareto_serialized if _e.get("dsr_passed"))
        print(
            f"    DSR gate ({template_name}): ANNOTATE-only — {_n_passed}/"
            f"{len(pareto_serialized)} clear τ={dsr_threshold}; full front KEPT "
            f"(dsr_passed flags the significant subset).",
            file=sys.stderr,
        )
        return pareto_serialized, front
    except Exception as _dsr_exc:
        # Fail-open: on any gate error, annotate as unknown and keep the front
        # intact rather than silently dropping every champion. Overwrite ALL
        # members to None (not setdefault) so a mid-loop failure cannot leave a
        # mix of real + None annotations on the returned front (review L2).
        print(f"    [DSR gate] {type(_dsr_exc).__name__}: {_dsr_exc}",
              file=sys.stderr)
        # MEDIUM-8: fail-OPEN keeps the front, but mark the annotation as
        # inconclusive (gate errored) so a strict `dsr_passed is True` reader does
        # NOT mistake an un-run gate for a pass.
        for _e in pareto_serialized:
            _e["dsr"] = None
            _e["dsr_passed"] = False
            _e["dsr_inconclusive"] = True
        return pareto_serialized, front


def _run_one_template(template: Template, config: ExperimentConfig,
                      grammar: Grammar, train_data: pd.DataFrame,
                      splits: Dict[str, pd.DataFrame],
                      train_terminal_data: Optional[Dict[str, np.ndarray]] = None,
                      fold_norm_stats: Optional[Dict] = None,
                      ) -> TemplateRunResult:
    """Run a single template's evolution + val/test scoring. Returns a
    populated TemplateRunResult. Raises on any failure — caller wraps in
    try/except and logs. Extracted from run_experiment's per-template
    loop for testability + Commit 6.6 resilience."""
    t_template = time.time()
    template_seed = _derive_template_seed(config.seed, template.name)
    print(f"\n  → template: {template.name} (seed={template_seed})")

    original_seeds = {
        "entry": to_str(template.entry_seed),
        "exit":  to_str(template.exit_seed),
        "size":  to_str(template.size_seed),
    }
    if template.delta_seed is not None:
        original_seeds["delta"] = to_str(template.delta_seed)

    if config.condition == "scalar-only":
        # BLOCKER-14 (2026-06-01 audit): also substitute the delta_seed (Level-B 4th
        # tree). delta_seed is normally a bare ephemeral, but routing it through the
        # same scalar-only substitution guarantees NO seed slot can inject a QC-non-
        # transferable scalar into the seeded fraction of the population.
        template = _dc_replace(
            template,
            entry_seed=substitute_stripped_terminals_for_scalar_only(template.entry_seed),
            exit_seed=substitute_stripped_terminals_for_scalar_only(template.exit_seed),
            size_seed=substitute_stripped_terminals_for_scalar_only(template.size_seed),
            delta_seed=(substitute_stripped_terminals_for_scalar_only(template.delta_seed)
                        if template.delta_seed is not None else None),
        )
    elif config.condition == "probes-only":
        template = _dc_replace(
            template,
            entry_seed=substitute_for_probes_pure(template.entry_seed),
            exit_seed=substitute_for_probes_pure(template.exit_seed),
            size_seed=substitute_for_probes_pure(template.size_seed),
        )
    elif config.condition == "emb-only":
        template = _dc_replace(
            template,
            entry_seed=substitute_for_emb_pure(template.entry_seed),
            exit_seed=substitute_for_emb_pure(template.exit_seed),
            size_seed=substitute_for_emb_pure(template.size_seed),
        )
    # Recalibrate seed tree EphReal thresholds to per-fold norm stats.
    # Seed trees are constructed at import time using frozen TERMINAL_NORM_STATS.
    # When per-fold normalization is active, terminal data is normalized with
    # different center/scale, so seed thresholds must be re-normalized to match.
    if fold_norm_stats is not None:
        from layer2.terminal_stats import recalibrate_seed_thresholds
        # plan #4 (2026-06-04 review LOW fix): under trailing-rolling norm the LEVEL
        # terminals are trailing-normalized, NOT in the fold-expanding frame, so their
        # frozen seed thresholds must NOT be recalibrated to expanding stats. (The VRP
        # compound seed is tagged fold_normalized → already skipped; this covers the
        # non-VRP frozen LEVEL-terminal seeds, e.g. RPB/BWB ATM_IV gates.)
        _skip = None
        if getattr(config, "norm_mode", "expanding") == "trailing_rolling":
            from layer2.trailing_norm import LEVEL_TERMINALS as _skip
        for seed_tree in (template.entry_seed, template.exit_seed, template.size_seed):
            if seed_tree is not None:
                recalibrate_seed_thresholds(seed_tree, fold_norm_stats, skip_terminals=_skip)
        if template.delta_seed is not None:
            recalibrate_seed_thresholds(template.delta_seed, fold_norm_stats, skip_terminals=_skip)

    # F1 fix (2026-04-25, BLOCKER): scalar-only arm must NOT consume
    # encoder-derived columns at backtester time. Even though F1 also
    # strips IN_REGIME/REGIME_IS from SCALAR_ONLY_FUNCTIONS and Regime
    # literals from build_scalar_only_terminal_set(), this is defense-in-
    # depth: forbidden_terminal_columns drops PredRegime / RegimeProb* /
    # other probe scalars from bar_data so EvaluationContext.update()
    # cannot populate ctx.current_regime from L1's 4-class regime label.
    # Probe scalars (PredRV15/30/PredSpread) are also forbidden because
    # they are encoder probe outputs — substitution rewrites them in seed
    # trees but a tree evolved later via mutation/crossover could reach
    # a non-substituted probe terminal name; we block at the data layer.
    # defense-in-depth: strip forbidden terminals from data layer
    # so the evaluator cannot access encoder-input scalars in B/C conditions.
    from layer2.grammar import _ENCODER_INPUT_SCALARS
    if config.condition == "probes-only":
        _forbidden_terminal_names = _ENCODER_INPUT_SCALARS
    elif config.condition == "emb-only":
        # Emb-only: strip encoder-input scalars AND probe scalars AND
        # synthesized regime indicators (defense-in-depth, ablation audit 2026-05-14)
        from layer2.io import PROBE_SCALAR_COLUMNS_V2, REGIME_PROB_COLUMNS
        _forbidden_terminal_names = (
            _ENCODER_INPUT_SCALARS
            | frozenset(PROBE_SCALAR_COLUMNS_V2)
            | frozenset(REGIME_PROB_COLUMNS)
            | frozenset({"RegimeAboveLow", "RegimeIsHigh", "RegimeIsPremium"})
        )
    elif config.condition == "scalar-only":
        # Defense-in-depth: strip probe + regime scalars from vectorized path
        # too (grammar gate prevents access, but the non-vectorized path strips
        # them, so both paths should be consistent for ablation purity).
        from layer2.io import PROBE_SCALAR_COLUMNS_V2, REGIME_PROB_COLUMNS
        _forbidden_terminal_names = (
            frozenset(PROBE_SCALAR_COLUMNS_V2)
            | frozenset(REGIME_PROB_COLUMNS)
        )
    else:
        _forbidden_terminal_names = frozenset()

    # Warn if non-vectorized evaluator is used for multi-leg templates.
    # The non-vectorized MultiLegOptionsBacktester uses stale B-S pricing
    # (no skew, no stop-loss, no credit haircut, no calibrated costs).
    # Production runs should always use --minute (which sets use_vectorized=True).
    _use_vec = config.use_vectorized
    if not _use_vec and template.n_legs > 1:
        import warnings
        warnings.warn(
            f"Non-vectorized evaluator used for multi-leg template "
            f"'{template.name}' — results use stale pricing model "
            f"(no skew, no stop-loss, no credit haircut). "
            f"Use --minute for production runs.",
            stacklevel=2,
        )

    _fold_recenter = None  # set in vectorized path for B/C/D EmbProj recenter

    if _use_vec:
        # Vectorized evaluator — 30-50x faster, required for 1-minute resolution
        assert train_terminal_data is not None, (
            "use_vectorized=True requires pre-computed train_terminal_data"
        )
        # Strip forbidden terminals from the data dict ()
        if _forbidden_terminal_names:
            train_terminal_data = {
                k: v for k, v in train_terminal_data.items()
                if k not in _forbidden_terminal_names
            }
        # #197: Extend warmup to 60 for B/C/D conditions — encoder needs
        # 60 bars of context. Bars 0-59 have zero embedding vectors.
        _emb_cols = [c for c in train_data.columns if c.startswith("EMB_")]
        _effective_warmup = max(config.backtester_warmup_bars,
                                60 if _emb_cols else config.backtester_warmup_bars)
        # v10: Load fold-recenter stats for B/C/D EmbProj drift correction.
        # Without this, val/test EmbProj values drift 2.8-6.2 sigma from
        # training distribution, biasing the ablation against embedding arms.
        _fold_recenter = None
        _train_fold_id = None
        if _emb_cols and config.condition not in ("scalar-only",):
            try:
                from layer2.evaluator import (
                    _load_fold_recenter_cached,
                    _resolve_fold_ids_for_data, _dominant_fold_id,
                )
                _fold_recenter = _load_fold_recenter_cached(raise_on_missing=True)
                _train_fids = _resolve_fold_ids_for_data(train_data)
                _train_fold_id = _dominant_fold_id(_train_fids)
            except Exception:
                pass  # graceful: no recenter = identity (worse but not crash)
        fe = VectorizedFitnessEvaluator(
            template=template, data=train_data,
            terminal_data=train_terminal_data,
            min_trades=config.min_trades,
            warmup_bars=_effective_warmup,
            cost_multiplier=config.cost_multiplier,
            fold_recenter_stats=_fold_recenter,
            fold_id=_train_fold_id,
            regime_gate_enabled=config.regime_gate_enabled,
            regime_gate_margin_k=config.regime_gate_margin_k,
        )
    else:
        # Original per-bar evaluator
        if config.condition == "scalar-only":
            from layer2.io import PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS
            _forbidden_cols = tuple(PROBE_SCALAR_COLUMNS) + tuple(REGIME_PROB_COLUMNS)
        elif config.condition == "probes-only":
            # Condition B: block encoder-input scalars but KEEP probes
            _forbidden_cols = tuple(_forbidden_terminal_names)
        elif config.condition == "emb-only":
            # Condition C: block encoder-input scalars AND probes
            from layer2.io import PROBE_SCALAR_COLUMNS, REGIME_PROB_COLUMNS
            _forbidden_cols = (
                tuple(_forbidden_terminal_names)
                + tuple(PROBE_SCALAR_COLUMNS)
                + tuple(REGIME_PROB_COLUMNS)
            )
        else:
            _forbidden_cols = ()
        fe = FitnessEvaluator(
            template=template, data=train_data,
            backtester_kwargs={
                "warmup_bars": config.backtester_warmup_bars,
                "default_minutes_to_expiry": config.backtester_minutes_to_expiry,
                "forbidden_terminal_columns": _forbidden_cols,
            },
            min_trades=config.min_trades,
        )
    # Commit 6.8 review fix: FitnessEvaluator.objectives is NOT currently
    # a field on ExperimentConfig and therefore not in the run fingerprint.
    # Assert invariance so the future-drift surface is closed at the call site.
    assert fe.objectives == DEFAULT_OBJECTIVES, (
        f"Evaluator objectives ({fe.objectives}) differ from "
        f"DEFAULT_OBJECTIVES ({DEFAULT_OBJECTIVES}) — resume fingerprint "
        f"does not cover objectives drift. Either revert to defaults or "
        f"thread `objectives` through ExperimentConfig + _FINGERPRINT_FIELDS."
    )
    # Condition C (emb-only): entry seeds collapse after substitution
    # (ATM_IV → SessionReturn → AND(SR>0, SR<0) = always-false). But
    # exit and delta seeds ARE valid. Use seed_fraction for exit/delta
    # while randomizing entry trees via seed_entry_random flag.
    # Condition C (emb-only): entry seeds collapse after substitution
    # (ATM_IV → SessionReturn → AND(SR>0, SR<0) = always-false).
    # seed_entry_random=True randomizes entry trees in the seeded fraction
    # while keeping valid exit/delta seeds from the template.
    _seed_entry_random = (config.condition == "emb-only")
    evo_config = EvolutionConfig(
        pop_size=config.pop_size, n_generations=config.n_generations,
        seed=template_seed, crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate, seed_fraction=config.seed_fraction,
        seed_entry_random=_seed_entry_random,
        n_partitions=config.n_partitions,
    )

    # H2 pre-reg v4: per-generation heartbeat. Writes a small JSON file
    # after each generation's metrics are computed so an external monitor
    # (or the operator) can tell the difference between "GP is slow" and
    # "GP is dead" without tail-f'ing stdout. Atomic write → the reader
    # never sees a torn file.
    _hb_path = (
        Path(config.output_dir) / f"heartbeat_{template.name}.json"
        if config.output_dir is not None else None
    )
    _hb_start = time.time()

    def _heartbeat_cb(m: "GenerationMetrics", _pop: "List[Individual]") -> None:
        if _hb_path is None:
            return
        try:
            # feasible_count (2026-06-01 run-observability fix): the number of
            # NON-sentinel individuals. front_size counts the Pareto front including
            # all-sentinel members, so an operator watching front_size cannot tell a
            # 100%-infeasible population (going nowhere) from real progress. A run
            # whose feasible_count stays 0 for many generations is producing nothing.
            from layer2.fitness import FAILED_FITNESS_SENTINEL as _SENT
            _feasible = sum(
                1 for _i in _pop
                if _i.fitness is not None
                and float(_i.fitness[0]) < _SENT * 0.5
            )
            _atomic_write_text(_hb_path, json.dumps({
                "template":          template.name,
                "generation":        int(m.generation),
                "elapsed_s":         round(time.time() - _hb_start, 3),
                "wall_time_s_gen":   round(float(m.wall_time_s), 3),
                "population_size":   int(m.population_size),
                "front_size":        int(m.front_size),
                "feasible_count":    int(_feasible),
                "best_per_objective":  list(m.best_per_objective),
                "median_per_objective": list(m.median_per_objective),
                "nan_count":         int(m.nan_count),
                "cache_hit_rate":    float(m.cache_hit_rate),
                "stamped_utc":       time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
            }, indent=2))
            # Snapshot current Pareto front — overwritten each generation so
            # only the latest survives. Enables mid-run strategy extraction
            # without waiting for all generations to complete.
            _evaluated = [i for i in _pop if i.fitness is not None]
            _front = pareto_front(_evaluated) if _evaluated else []
            if _front:
                _snap_path = _hb_path.parent / f"snapshot_{template.name}.jsonl"
                _lines = []
                for _ind in _front:
                    _snap = {
                        "template": template.name,
                        "generation": int(m.generation),
                        "entry_tree": to_str(_ind.entry_tree),
                        "exit_tree": to_str(_ind.exit_tree),
                        "size_tree": to_str(_ind.size_tree),
                        "stop_mult": float(_ind.stop_mult),
                        "fitness": list(_ind.fitness) if _ind.fitness is not None else None,
                    }
                    if _ind.delta_tree is not None:
                        _snap["delta_tree"] = to_str(_ind.delta_tree)
                    _lines.append(json.dumps(_snap) + "\n")
                _atomic_write_text(_snap_path, "".join(_lines))
        except Exception as _hb_exc:
            # Heartbeat failure must NEVER kill evolution — better to run
            # without visibility than to crash a 4-hour pilot on a disk-full.
            # But log the FIRST failure so the operator knows visibility is lost.
            if not getattr(_heartbeat_cb, '_warned', False):
                print(f"[heartbeat] {type(_hb_exc).__name__}: {_hb_exc}",
                      file=sys.stderr)
                _heartbeat_cb._warned = True

    # Pre-compute filtered terminal data for val/test to match
    # forbidden-terminal stripping applied to train data. Without this,
    # val/test evaluation would include encoder-input scalars in B/C.
    _rc_terminal_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # Map each slice object to its prior-session VIX (embargo-gap close) so the
    # VIX lookahead-lag has no residual first-slice-day lookahead on val/test.
    _vix_prior_by_id = {}
    if isinstance(splits, dict) and splits.get("vix_prior"):
        _vix_prior_by_id[id(splits["val"])] = splits["vix_prior"].get("val", {})
        _vix_prior_by_id[id(splits["test"])] = splits["vix_prior"].get("test", {})

    def _get_filtered_terminal_data(data) -> Optional[Dict[str, np.ndarray]]:
        """Prepare terminal data with L2.60 forbidden terminals stripped."""
        if data is None or not config.use_vectorized:
            return None
        _did = id(data)
        if _did not in _rc_terminal_cache:
            from layer2.evaluator_vectorized import prepare_terminal_data as _ptd
            _vp = _vix_prior_by_id.get(id(data), {})
            if getattr(config, "norm_mode", "expanding") == "trailing_rolling":
                # plan #4: trailing-rolling robust norm for LEVEL terminals (causal,
                # per-day). val/test bars use their OWN trailing context; the first W
                # warmup days fall back to the expanding fold stats. (Seeding val's
                # window with the train tail — the QC-side online behavior — is a
                # Phase-3 refinement; impact bounded to the first W of ~125 val days.)
                from layer2.trailing_norm import apply_trailing_norm
                _raw = _ptd(data, normalize_terminals=False, vix_prior=_vp)
                td, _ = apply_trailing_norm(
                    _raw, data["date"].values,
                    W=int(getattr(config, "trailing_window", 20)),
                    static_stats=fold_norm_stats)
            else:
                td = _ptd(data, norm_stats_override=fold_norm_stats, vix_prior=_vp)
            if _forbidden_terminal_names:
                td = {k: v for k, v in td.items()
                      if k not in _forbidden_terminal_names}
            _rc_terminal_cache[_did] = td
        return _rc_terminal_cache[_did]

    val_data  = splits["val"]  if len(splits["val"])  > 0 else None

    # Task #34: Build inner validation evaluator callback.
    # Evaluates Pareto front on val data every 10 generations (observation
    # only -- NOT used for NSGA-III selection). Returns mean val_sharpe.
    _val_evaluator_fn = None
    if config.inner_val_enabled and val_data is not None and len(val_data) > 0 and _use_vec:
        _val_td = _get_filtered_terminal_data(val_data)
        # Resolve val fold_id for EmbProj recenter
        _inner_val_fold_id = None
        if _fold_recenter is not None:
            try:
                from layer2.evaluator import _resolve_fold_ids_for_data, _dominant_fold_id
                _inner_val_fold_id = _dominant_fold_id(_resolve_fold_ids_for_data(val_data))
            except Exception:
                pass
        _inner_val_extra = {
            "fold_recenter_stats": _fold_recenter,
            "fold_id": _inner_val_fold_id,
        }

        def _val_evaluator_fn(front_individuals):
            """Evaluate Pareto front on validation data, return mean val_sharpe."""
            sharpes = []
            for ind in front_individuals:
                try:
                    _fitness = fe.score_on_data(
                        ind.entry_tree, ind.exit_tree, ind.size_tree,
                        val_data, terminal_data=_val_td,
                        delta_tree=ind.delta_tree, stop_mult=ind.stop_mult,
                        **_inner_val_extra,
                    )
                    # neg_sharpe is objective 0 (negated), so val_sharpe = -fitness[0]
                    # But fitness includes parsimony penalty, so extract raw:
                    _val_sharpe = -float(_fitness[0])  # undo negation
                    if _val_sharpe > -1e5:  # filter sentinel values (regime gate etc)
                        sharpes.append(_val_sharpe)
                except Exception:
                    pass  # skip individuals that fail on val data
            return float(np.mean(sharpes)) if sharpes else 0.0

    final_pop, metrics_log, _val_tracking = evolve(
        template, fe, grammar, evo_config,
        on_generation=_heartbeat_cb,
        val_evaluator=_val_evaluator_fn,
    )
    front = pareto_front(final_pop)

    # Dedup the Pareto front: during-evolution dedup only covers the population
    # each generation, but the final crossover/mutation can reintroduce duplicates
    # that land on the front. Dedup on full signature (entry+exit+size+delta).
    # canonical_key collapses commutative-equivalent twins (AND(a,b)==AND(b,a),
    # GT(x,y)==LT(y,x)) so the reported front isn't padded with clones.
    _seen_front: dict = {}
    _deduped_front = []
    for ind in front:
        _sig = (
            canonical_key(ind.entry_tree),
            canonical_key(ind.exit_tree),
            canonical_key(ind.size_tree),
            canonical_key(ind.delta_tree) if ind.delta_tree is not None else "",
        )
        if _sig not in _seen_front:
            _seen_front[_sig] = True
            _deduped_front.append(ind)
    if len(_deduped_front) < len(front):
        print(f"    Pareto front dedup: {len(front)} → {len(_deduped_front)}", file=sys.stderr)
    front = _deduped_front

    # Task #34: Save best-val snapshot to disk if available.
    if (_val_tracking.get("best_val_front") is not None
            and config.output_dir is not None):
        _bestval_snap_path = Path(config.output_dir) / f"snapshot_bestval_{template.name}.jsonl"
        _bestval_lines = []
        for _ind in _val_tracking["best_val_front"]:
            _snap = {
                "template": template.name,
                "generation": _val_tracking["best_val_generation"],
                "best_val_sharpe": _val_tracking["best_val_sharpe"],
                "entry_tree": to_str(_ind.entry_tree),
                "exit_tree": to_str(_ind.exit_tree),
                "size_tree": to_str(_ind.size_tree),
                "stop_mult": float(_ind.stop_mult),
                "fitness": list(_ind.fitness) if _ind.fitness is not None else None,
            }
            if _ind.delta_tree is not None:
                _snap["delta_tree"] = to_str(_ind.delta_tree)
            _bestval_lines.append(json.dumps(_snap) + "\n")
        try:
            _atomic_write_text(_bestval_snap_path, "".join(_bestval_lines))
        except Exception as _bv_exc:
            print(f"[best-val snapshot] {type(_bv_exc).__name__}: {_bv_exc}",
                  file=sys.stderr)

    # Task #34: Update final heartbeat with val tracking data.
    if _hb_path is not None and _val_tracking.get("val_checkpoints"):
        try:
            _hb_content = json.loads(_hb_path.read_text()) if _hb_path.exists() else {}
            _hb_content["val_sharpe_checkpoints"] = [
                {"generation": g, "val_sharpe": round(s, 6)}
                for g, s in _val_tracking["val_checkpoints"]
            ]
            _hb_content["best_val_generation"] = _val_tracking["best_val_generation"]
            _hb_content["best_val_sharpe"] = (
                round(_val_tracking["best_val_sharpe"], 6)
                if _val_tracking["best_val_sharpe"] is not None else None
            )
            _atomic_write_text(_hb_path, json.dumps(_hb_content, indent=2))
        except Exception as _hb_val_exc:
            print(f"[heartbeat val update] {type(_hb_val_exc).__name__}: {_hb_val_exc}",
                  file=sys.stderr)

    test_data = splits["test"] if len(splits["test"]) > 0 else None
    # : use filtered terminal data for val/test scoring to prevent
    # encoder-input scalars from leaking into B/C conditions' hypervolume.
    # _val_td may have been computed above for inner val; _get_filtered_terminal_data
    # caches by id(data), so no redundant recomputation.
    _val_td = _get_filtered_terminal_data(val_data) if val_data is not None else None
    _test_td = _get_filtered_terminal_data(test_data) if test_data is not None else None
    # v10: Resolve val/test fold_ids for EmbProj recenter correction.
    # Each split's rows belong to one fold; _dominant_fold_id extracts it.
    _val_fold_id = None
    _test_fold_id = None
    if _fold_recenter is not None:
        try:
            from layer2.evaluator import _resolve_fold_ids_for_data, _dominant_fold_id
            if val_data is not None and len(val_data) > 0:
                _val_fold_id = _dominant_fold_id(_resolve_fold_ids_for_data(val_data))
            if test_data is not None and len(test_data) > 0:
                _test_fold_id = _dominant_fold_id(_resolve_fold_ids_for_data(test_data))
        except Exception as _fold_exc:
            print(f"  [score] fold_id resolution failed for val/test: "
                  f"{type(_fold_exc).__name__}: {_fold_exc}", file=sys.stderr)
    # Build kwargs for score_on_data — vectorized path supports fold_recenter;
    # non-vectorized path does not.
    _val_extra = {}
    _test_extra = {}
    if _use_vec:
        _val_extra = {"fold_recenter_stats": _fold_recenter, "fold_id": _val_fold_id}
        _test_extra = {"fold_recenter_stats": _fold_recenter, "fold_id": _test_fold_id}
    front_val_fitness  = (np.array([
        fe.score_on_data(ind.entry_tree, ind.exit_tree, ind.size_tree,
                         val_data, terminal_data=_val_td,
                         delta_tree=ind.delta_tree, stop_mult=ind.stop_mult,
                         **_val_extra)
        for ind in front
    ]) if val_data is not None else None)
    front_test_fitness = (np.array([
        fe.score_on_data(ind.entry_tree, ind.exit_tree, ind.size_tree,
                         test_data, terminal_data=_test_td,
                         delta_tree=ind.delta_tree, stop_mult=ind.stop_mult,
                         **_test_extra)
        for ind in front
    ]) if test_data is not None else None)
    val_hv  = (_compute_val_hypervolume(front_val_fitness,  H2_HV_REFERENCE_POINT)
               if front_val_fitness  is not None else None)
    test_hv = (_compute_val_hypervolume(front_test_fitness, H2_HV_REFERENCE_POINT)
               if front_test_fitness is not None else None)

    effective_seeds = {
        "entry": to_str(template.entry_seed),
        "exit":  to_str(template.exit_seed),
        "size":  to_str(template.size_seed),
    }
    # v8 (RF pre-filter F4 fix): persist trade counts + val/test Sharpes so the
    # `scripts/pilot_to_qc_handoff.py` F4 gate can actually be enforced. Prior
    # to v8 these fields weren't stored, so F4 was a silent no-op. Also emits
    # per-individual train/val/test trade counts for the handoff + QC backtest
    # survival audit.
    #
    # v8 Fix 5 (C4): memoize by (tree_hash, id(data)). Backtester is
    # deterministic given tree + data (verified by Code Reviewer), so
    # repeated calls on the same (tree, data) pair can hit the cache.
    # Not global — scoped to this template's front scoring.
    # F5 note: score_on_data (lines 1070-1077) and _run_counts both evaluate
    # the same individual on val/test data independently. _run_counts has its
    # own cache, but score_on_data doesn't share it. For a Pareto front of
    # 256 individuals, this means ~512 redundant backtests. A full dedup
    # would require unifying both into a single cached evaluation function
    # that returns (fitness_vector, total_trades, sharpe). Deferred to a
    # future performance pass — correctness is unaffected.
    _run_counts_cache: Dict[Tuple[str, int], tuple] = {}

    def _run_counts(ind: Individual, data):
        """Re-run backtester to extract stats tuple.
        Returns: (total_trades, sharpe, error_flag, sortino, psr,
                  exit_utilization, return_skew, return_kurtosis,
                  max_drawdown_uncapped, avg_position_size)
        error_flag is 0 on success, 1 on caught exception.
        max_drawdown_uncapped / avg_position_size feed the HIGH-5 drawdown-aware
        champion sort (2026-06-01 audit) — same adj_dd inputs evolution penalizes.
        """
        if data is None or len(data) == 0:
            return (0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        _key = (
            fe._tree_hash(ind.entry_tree, ind.exit_tree, ind.size_tree,
                          ind.delta_tree, stop_mult=ind.stop_mult),
            id(data),
        )
        if _key in _run_counts_cache:
            return _run_counts_cache[_key]
        try:
            if config.use_vectorized:
                from layer2.evaluator_vectorized import vectorized_backtest as _vbt
                # Determine fold_id for this data split (for EmbProj recenter)
                _rc_fold_id = None
                if _fold_recenter is not None:
                    try:
                        from layer2.evaluator import (
                            _resolve_fold_ids_for_data, _dominant_fold_id,
                        )
                        _rc_fold_id = _dominant_fold_id(
                            _resolve_fold_ids_for_data(data))
                    except Exception as _fold_exc:
                        print(f"  [_run_counts] fold_id resolution failed: "
                              f"{type(_fold_exc).__name__}: {_fold_exc}",
                              file=sys.stderr)
                result = _vbt(
                    ind.entry_tree, ind.exit_tree, ind.size_tree,
                    data, template, delta_tree=ind.delta_tree,
                    # Re-score with the SAME evolved stop the GP selected on, else
                    # the serialized train/val/test Sharpe + trade counts would
                    # diverge from the fitness the front was chosen by.
                    stop_loss_credit_multiple=ind.stop_mult,
                    cost_multiplier=config.cost_multiplier,
                    warmup_bars=_effective_warmup,
                    terminal_data=_get_filtered_terminal_data(data),
                    fold_recenter_stats=_fold_recenter,
                    fold_id=_rc_fold_id,
                )
            else:
                result = fe._backtester.run(
                    entry_tree=ind.entry_tree, exit_tree=ind.exit_tree,
                    size_tree=ind.size_tree, data=data,
                )
            out = (int(result.total_trades), float(result.sharpe), 0,
                   float(result.sortino), float(result.psr),
                   float(result.exit_utilization),
                   float(result.return_skew), float(result.return_kurtosis),
                   float(getattr(result, "max_drawdown_uncapped", 0.0)),
                   float(getattr(result, "avg_position_size", 0.0)))
        except Exception as _exc:
            print(f"  [_run_counts] {type(_exc).__name__}: {_exc}",
                  file=sys.stderr)
            out = (0, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        _run_counts_cache[_key] = out
        return out

    pareto_serialized = []
    for idx, ind in enumerate(front):
        entry = _serialize_individual(ind)
        if front_val_fitness is not None:
            entry["val_fitness"] = front_val_fitness[idx].tolist()
        if front_test_fitness is not None:
            entry["test_fitness"] = front_test_fitness[idx].tolist()
        # Train-run trade count (re-run to get it; fitness path doesn't
        # preserve the BacktestResult)
        (n_tr_train, sh_train, err_train, sort_train, psr_train,
         eu_train, skew_train, kurt_train, dd_train, aps_train) = \
            _run_counts(ind, fe.data)
        (n_tr_val, sh_val, err_val, sort_val, psr_val,
         eu_val, skew_val, kurt_val, dd_val, aps_val) = \
            _run_counts(ind, val_data)
        (n_tr_test, sh_test, err_test, sort_test, psr_test,
         eu_test, skew_test, kurt_test, dd_test, aps_test) = \
            _run_counts(ind, test_data)
        entry["total_trades"] = n_tr_train    # train-run — what F4 gates on
        entry["n_trades_val"] = n_tr_val
        entry["n_trades_test"] = n_tr_test
        entry["train_sharpe"] = sh_train
        entry["val_sharpe"] = sh_val
        entry["test_sharpe"] = sh_test
        entry["val_sortino"] = sort_val
        # MEDIUM-9 (2026-06-01 audit): val_psr/test_psr are NO LONGER serialized.
        # BacktestResult.psr is fragile — the evaluator property reshapes per-bar→
        # n_days only when _daily_returns is absent, and the vectorized result does
        # not set it at all (so it defaults to a non-comparable value). Nothing
        # downstream consumes val_psr/test_psr; the DSR (val_dsr / front dsr) is the
        # honest, units-correct multiple-testing-aware significance metric. Dropping
        # the serialized PSR removes a latent foot-gun (a future reader trusting an
        # unreliable field). psr_val/psr_test remain in the _run_counts tuple but are
        # intentionally unused here.
        # HIGH-5 (2026-06-01 audit): drawdown inputs for the drawdown-aware
        # champion sort (G9). avg_position_size is the sizing-exploit-proof
        # denominator (matches the evolution penalty's adj_dd).
        entry["val_max_drawdown"] = dd_val
        entry["val_avg_position_size"] = aps_val
        entry["test_sortino"] = sort_test
        # DSR return distribution moments (Bailey & Lopez de Prado 2014)
        entry["val_return_skew"] = skew_val
        entry["val_return_kurtosis"] = kurt_val
        entry["train_return_skew"] = skew_train
        entry["train_return_kurtosis"] = kurt_train
        entry["test_return_skew"] = skew_test
        entry["test_return_kurtosis"] = kurt_test
        entry["exit_utilization_train"] = eu_train
        entry["exit_utilization_val"] = eu_val
        entry["exit_utilization_test"] = eu_test
        # v8 Fix 5: surface backtester errors rather than silently zero.
        if err_train or err_val or err_test:
            entry["backtester_errors"] = {
                "train": bool(err_train), "val": bool(err_val),
                "test": bool(err_test),
            }
        # C5 fix (2026-05-31, audit gate-asymmetry leak): `val_sharpe` above is a
        # GATE-FREE _run_counts re-score, but the GP's val objective
        # (`val_fitness`, gated by the random-entry/regime/fire-rate/exit-util
        # cascade) may have SENTINEL'd this strategy. A strategy that FAILED the
        # val gates must not carry a clean gate-free val_sharpe into G1/sorting.
        # Reconcile: keep the raw value for transparency, but if the gated val
        # objective is sentinel, set val_sharpe to the sentinel-implied (very
        # negative) value so G1/WFE/sorting see it as the failure it is.
        if front_val_fitness is not None:
            from layer2.fitness import FAILED_FITNESS_SENTINEL as _SENT
            _val_gate_failed = bool(float(front_val_fitness[idx][0]) >= _SENT)
            entry["val_sharpe_gatefree"] = sh_val
            entry["val_gate_failed"] = _val_gate_failed
            if _val_gate_failed:
                # gated neg_sharpe == +SENTINEL → implied val Sharpe == −SENTINEL
                entry["val_sharpe"] = -float(front_val_fitness[idx][0])
        pareto_serialized.append(entry)

    # P1-B (REDESIGNED): Deflated-Sharpe-Ratio FILTER gate at champion
    # selection. The multiple-testing inflation SR* is computed ONCE from the
    # WHOLE evolution's distinct trials (N, V) -- NOT the tiny final front
    # (which would invert V: an overfit spike inflates the bar and then clears
    # it; see Blocker B). N = count of distinct individuals evaluated across all
    # generations (deduped by canonical_key, recorded in fitness.py); V =
    # variance of their DAILY training Sharpes over members with n_days >=
    # min_days (invalid short-sample members excluded). Each champion's
    # TRAINING Sharpe is then deflated against SR* and a champion that fails
    # (DSR < dsr_threshold) is REMOVED from the returned front (this gate
    # FILTERS -- it is not annotate-only). It does NOT alter the NSGA-III
    # objective vector, per-individual fitness, or result.psr.
    #
    # N/V judgment call (pinnable in the fresh pre-reg): N counts distinct
    # individuals across ALL generations whose backtest produced a finite,
    # non-phantom training Sharpe with n_days >= min_days; V is the population
    # variance (ddof=0) of those DAILY Sharpes. Members are NOT additionally
    # required to have passed the GP quality/regime gates -- a backtest that
    # ran is a multiple-testing trial. (See dsr_gate_evolution docstring.)
    _DSR_MIN_DAYS = 20
    # We deflate the TRAINING Sharpe (the quantity NSGA-III selected on, hence the
    # one inflated by the N-trial search) against the training-trial SR* — classic
    # Bailey & Lopez de Prado, consistent IS units. So T is the TRAINING window.
    _train_T = int(fe.data["date"].nunique()) if (
        fe.data is not None and len(fe.data) > 0 and "date" in fe.data.columns
    ) else 0
    pareto_serialized, front = _apply_dsr_front_gate(
        pareto_serialized, front,
        train_T=_train_T,
        evo_records=(_val_tracking.get("evo_trial_records", {}) or {}),
        dsr_gate_enabled=config.dsr_gate_enabled,
        dsr_threshold=config.dsr_threshold,
        template_name=template.name,
        min_days=_DSR_MIN_DAYS,
    )

    # Hardening #1: detect degenerate Pareto fronts (all individuals have
    # FAILED_FITNESS_SENTINEL). Report actual front size = 0 in that case.
    from layer2.fitness import FAILED_FITNESS_SENTINEL as _SENTINEL
    _actual_front_size = len(front)
    if front and all(
        ind.fitness is not None and all(f >= _SENTINEL for f in ind.fitness)
        for ind in front
    ):
        import warnings
        warnings.warn(
            f"Template '{template.name}': ALL {len(front)} Pareto front individuals have "
            f"sentinel fitness (zero productive strategies found in {len(metrics_log)} generations).",
            stacklevel=2,
        )
        _actual_front_size = 0

    # P0-6 (A): persist the per-fold MINUTE normalization each strategy was
    # evolved/selected under, so the QC deploy can normalize terminals with the
    # IDENTICAL stats (codegen norm_stats override) instead of the frozen daily
    # constants -- eliminating train/serve normalization skew (Sculley et al.,
    # 2015). JSON serializes the (center, scale, method) tuples to lists; the
    # codegen override reads positionally so lists are fine. The same dict is
    # shared by every strategy from this fold/template (read-only).
    if fold_norm_stats:
        _fns_serial = {
            k: [v[0], v[1]] + ([v[2]] if len(v) >= 3 else [])
            for k, v in fold_norm_stats.items()
        }
        for _ent in pareto_serialized:
            _ent["fold_norm_stats"] = _fns_serial

    return TemplateRunResult(
        template_name=template.name,
        final_population_size=len(final_pop),
        pareto_front_size=_actual_front_size,
        pareto_front=pareto_serialized,
        metrics_log=[_serialize_metrics(m) for m in metrics_log],
        fitness_cache_stats=fe.stats,
        wall_time_s=time.time() - t_template,
        seed_trees_original=original_seeds,
        seed_trees_effective=effective_seeds,
        val_hypervolume=val_hv,
        test_hypervolume=test_hv,
        hv_reference_point=list(H2_HV_REFERENCE_POINT),
    )


def _serialize_individual(ind: Individual) -> Dict:
    d = {
        "template_name": ind.template_name,
        "entry_tree": to_str(ind.entry_tree),
        "exit_tree": to_str(ind.exit_tree),
        "size_tree": to_str(ind.size_tree),
        # Evolved stop-loss gene — ALWAYS serialized (even 0.0 = hold-to-expiry)
        # so L3 codegen emits the exact stop the strategy was evolved under
        # (proxy↔QC parity). Plain float, not gated on a sentinel.
        "stop_mult": float(ind.stop_mult),
        "fitness": ind.fitness.tolist() if ind.fitness is not None else None,
        "age": ind.age,
        "total_nodes": ind.total_nodes(),
    }
    if ind.delta_tree is not None:
        d["delta_tree"] = to_str(ind.delta_tree)
    return d


def _save_single_template_result(result: TemplateRunResult, output_dir: Path,
                                  env_snapshot: Optional[Dict] = None,
                                  grammar_signature: Optional[Dict] = None):
    """Write ONE template's artifacts immediately after it completes.
    Commit 6.6: converts "mid-run crash loses all work" into "lose at most
    the currently-running template."

    Commit 6.8: env_snapshot + grammar_signature are stamped INTO the
    per-template summary so a resumed run can detect environment drift.
    Model QA required fix: _provenance_record running at save time
    otherwise misattributes current env to templates that ran earlier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Per-strategy JSONL — atomic write so a concurrent reader never sees
    # a half-flushed file.
    strat_path = output_dir / f"strategies_{result.template_name}.jsonl"
    # BLOCKER-3 (2026-06-01 audit): a degenerate template (pareto_front_size==0,
    # detected upstream as "ALL members sentinel") would otherwise serialize
    # FAILED_FITNESS_SENTINEL pseudo-strategies (fitness=[1e6,1e6,1e6],
    # val_sharpe=-1e6) as "champions". Write an EMPTY strategies file in that
    # case, and defensively drop any individual member whose fitness is all-
    # sentinel or whose val_sharpe is the sentinel (belt-and-suspenders for a
    # mixed front). Downstream readers (cross-fold persistence, handoff F5) then
    # never see a sentinel masquerading as a candidate.
    from layer2.fitness import FAILED_FITNESS_SENTINEL as _SENT

    def _is_sentinel_member(ind: Dict) -> bool:
        _fit = ind.get("fitness")
        if isinstance(_fit, (list, tuple)) and _fit and all(
                (f is not None and f >= _SENT) for f in _fit):
            return True
        try:
            return float(ind.get("val_sharpe", 0.0)) <= -1e5
        except (TypeError, ValueError):
            return False

    if result.pareto_front_size == 0:
        _front_to_write: List[Dict] = []  # degenerate template → empty file
    else:
        _front_to_write = [ind for ind in result.pareto_front
                           if not _is_sentinel_member(ind)]
    _atomic_write_text(
        strat_path,
        "".join(json.dumps(ind) + "\n" for ind in _front_to_write),
    )
    # Per-generation metrics
    metrics_path = output_dir / f"metrics_{result.template_name}.json"
    _atomic_write_text(
        metrics_path, json.dumps(result.metrics_log, indent=2),
    )
    # Per-template summary — used by resume detection
    summary_path = output_dir / f"template_{result.template_name}.json"
    _atomic_write_text(summary_path, json.dumps({
        "template_name":            result.template_name,
        "final_population_size":    result.final_population_size,
        "pareto_front_size":        result.pareto_front_size,
        "fitness_cache_stats":      result.fitness_cache_stats,
        "wall_time_s":              result.wall_time_s,
        "seed_trees_original":      result.seed_trees_original,
        "seed_trees_effective":     result.seed_trees_effective,
        "val_hypervolume":          result.val_hypervolume,
        "test_hypervolume":         result.test_hypervolume,
        "hv_reference_point":       result.hv_reference_point,
        # 6.8: per-template env snapshot (so resume can detect drift)
        "env_snapshot":             env_snapshot or _env_snapshot(),
        "grammar_signature":        grammar_signature,
    }, indent=2))


def _write_run_fingerprint(output_dir: Path, fingerprint: str,
                            env_snapshot: Dict, grammar_signature: Dict,
                            config: ExperimentConfig, l1_sha256: str):
    """Commit 6.8: stamp a run_fingerprint.json so resume detection can
    verify the current config hasn't drifted from the one that produced
    existing per-template artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_path = output_dir / "run_fingerprint.json"
    _atomic_write_text(fp_path, json.dumps({
        "fingerprint":        fingerprint,
        "env_snapshot":       env_snapshot,
        "grammar_signature":  grammar_signature,
        "config_summary": {
            k: getattr(config, k) for k in _FINGERPRINT_FIELDS
        },
        "l1_parquet_sha256":  l1_sha256,
    }, indent=2))


def _check_resume_fingerprint(output_dir: Path, current_fingerprint: str,
                               current_env: Dict, current_grammar_sig: Dict,
                               config: Optional["ExperimentConfig"] = None):
    """Commit 6.8 BLOCKING fix: verify existing per-template artifacts in
    output_dir were produced under the SAME config + environment as the
    current run. If not, raise RuntimeError with a clear message directing
    the operator to either (a) use a different output_dir, (b) delete the
    stale artifacts, or (c) revert the environment.

    This is the single biggest scientific-validity gate in the resilience
    layer — without it, an operator who changes seed/pymoo-version and
    resumes would silently mix old artifacts with new evolution.
    """
    fp_path = output_dir / "run_fingerprint.json"
    if not fp_path.exists():
        # No prior fingerprint — check for stale per-template files anyway.
        # If per-template summaries exist without a fingerprint, they were
        # produced by a pre-6.8 run and we cannot guarantee integrity.
        stale = list(output_dir.glob("template_*.json"))
        if stale:
            raise RuntimeError(
                f"output_dir {output_dir} has {len(stale)} template_*.json "
                f"artifact(s) but no run_fingerprint.json. These artifacts "
                f"were produced before config-fingerprint integrity checking "
                f"(Commit 6.8) landed; the current run cannot safely resume "
                f"against them. Delete the stale artifacts or use a fresh "
                f"output_dir."
            )
        return  # clean start
    try:
        with fp_path.open() as f:
            prior = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"run_fingerprint.json in {output_dir} is corrupted ({exc}). "
            f"Delete it + the per-template artifacts, or use a fresh output_dir."
        )
    if prior.get("fingerprint") != current_fingerprint:
        # Backwards-compatible fallback: if _FINGERPRINT_FIELDS grew (e.g.,
        # probe_bundle_sha256 was added), the hash changes even though the
        # experiment is semantically identical. Fall back to field-by-field
        # comparison: accept the resume if every field in the STORED config
        # matches the current config, and any NEW fields default to None.
        if config is None:
            # No config available for field-by-field comparison — strict reject
            raise RuntimeError(
                f"Config fingerprint mismatch in {output_dir}:\n"
                f"  prior run: {prior.get('fingerprint')}\n"
                f"  this run:  {current_fingerprint}\n"
                f"One or more fields in {_FINGERPRINT_FIELDS} differ between\n"
                f"the prior run and this one. Resume would mix incompatible\n"
                f"artifacts — use a fresh output_dir or delete existing\n"
                f"template_*.json / strategies_*.jsonl / metrics_*.json files."
            )
        prior_summary = prior.get("config_summary", {})
        _mismatch_fields = []
        for fld in _FINGERPRINT_FIELDS:
            current_val = getattr(config, fld, None)
            if fld in prior_summary:
                # Field existed in old run — values must match
                if repr(prior_summary[fld]) != repr(current_val):
                    _mismatch_fields.append(
                        f"  {fld}: prior={prior_summary[fld]!r} vs current={current_val!r}")
            else:
                # Field is new (not in old run) — only accept if default/None
                if current_val is not None:
                    _mismatch_fields.append(
                        f"  {fld}: not in prior run, current={current_val!r} (expected None)")
        if _mismatch_fields:
            raise RuntimeError(
                f"Config fingerprint mismatch in {output_dir}:\n"
                f"  prior run: {prior.get('fingerprint')}\n"
                f"  this run:  {current_fingerprint}\n"
                f"Field differences:\n" + "\n".join(_mismatch_fields) + "\n"
                f"Resume would mix incompatible artifacts — use a fresh "
                f"output_dir or delete existing template_*.json / "
                f"strategies_*.jsonl / metrics_*.json files."
            )
        # Hash differs but all fields match — fingerprint schema grew.
        # Update the stored fingerprint to the new schema so subsequent
        # resumes use the current hash directly.
        print(f"  [resume] fingerprint schema updated ({len(prior_summary)} → "
              f"{len(_FINGERPRINT_FIELDS)} fields), config is compatible")
    # Check environment drift. Split between BLOCKING (scientific-validity
    # critical) and WARNING (I/O drift tolerable per requirements-l2.txt).
    # Commit 6.8 review fix: previously ALL drifts blocked, which would
    # false-positive-abort a long pilot on a pandas patch bump. Only the
    # evolution-math-relevant libs (pymoo/numpy/scipy/python) are blocking.
    prior_env = prior.get("env_snapshot", {})
    blocking_drift = [
        k for k in _ENV_DRIFT_BLOCKING_KEYS
        if prior_env.get(k) != current_env.get(k)
    ]
    warning_drift = [
        k for k in _ENV_DRIFT_WARNING_KEYS
        if prior_env.get(k) != current_env.get(k)
    ]
    if blocking_drift:
        raise RuntimeError(
            f"BLOCKING environment drift between prior run and this one:\n"
            f"  drifted keys: {blocking_drift}\n"
            f"  prior: {' '.join(f'{k}={prior_env.get(k)}' for k in blocking_drift)}\n"
            f"  now:   {' '.join(f'{k}={current_env.get(k)}' for k in blocking_drift)}\n"
            f"These libraries affect evolution math / random-stream\n"
            f"semantics. Resume would misattribute post-facto env\n"
            f"metadata to templates that ran under the earlier stack.\n"
            f"Use a fresh output_dir or revert to the pinned versions\n"
            f"(see requirements-l2.txt)."
        )
    if warning_drift:
        print(
            f"  [WARNING] non-blocking env drift on resume:\n"
            f"    keys: {warning_drift}\n"
            f"    prior: {' '.join(f'{k}={prior_env.get(k)}' for k in warning_drift)}\n"
            f"    now:   {' '.join(f'{k}={current_env.get(k)}' for k in warning_drift)}\n"
            f"    These libraries are declared tolerant of minor-version\n"
            f"    drift in requirements-l2.txt. Continuing."
        )
    # Check grammar signature — a grammar refactor invalidates resumed work
    prior_gram = prior.get("grammar_signature", {})
    if prior_gram.get("sha256") != current_grammar_sig.get("sha256"):
        raise RuntimeError(
            f"Grammar signature mismatch in {output_dir}:\n"
            f"  prior: {prior_gram.get('sha256')}\n"
            f"  now:   {current_grammar_sig.get('sha256')}\n"
            f"A grammar function-set / terminal-set change makes prior\n"
            f"evolved trees incompatible. Use a fresh output_dir."
        )


def _load_completed_templates(output_dir: Path) -> Dict[str, TemplateRunResult]:
    """Resume detection (Commit 6.6): scan output_dir for template_<name>.json
    files written by _save_single_template_result and rebuild in-memory
    TemplateRunResult objects so the caller can skip those templates in
    the main loop.

    Only the fields NEEDED for results.json summary are rebuilt — the
    per-strategy and per-metrics details live in their own files and are
    loaded on demand. Returns {template_name: TemplateRunResult}.
    """
    loaded: Dict[str, TemplateRunResult] = {}
    for summary_path in output_dir.glob("template_*.json"):
        try:
            with summary_path.open() as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupted summary — skip and let the template re-run
            continue
        name = d.get("template_name")
        if not name:
            continue
        # Check companion artifacts exist (don't trust a half-finished write)
        strat_path  = output_dir / f"strategies_{name}.jsonl"
        metrics_path = output_dir / f"metrics_{name}.json"
        if not (strat_path.exists() and metrics_path.exists()):
            continue
        # Load pareto_front from JSONL (Commit 6.8: torn-write safety —
        # wrap the comprehension in try/except so a half-written line
        # doesn't abort the whole resume path).
        try:
            pareto_front = [
                json.loads(line)
                for line in strat_path.read_text().splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError:
            # Truncated / corrupted jsonl — discard this template's partial
            # artifacts and let it re-run (do NOT delete files here; let
            # the operator inspect / clean up).
            continue
        try:
            with metrics_path.open() as f:
                metrics_log = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        loaded[name] = TemplateRunResult(
            template_name=name,
            final_population_size=d.get("final_population_size", 0),
            pareto_front_size=d.get("pareto_front_size", len(pareto_front)),
            pareto_front=pareto_front,
            metrics_log=metrics_log,
            fitness_cache_stats=d.get("fitness_cache_stats", {}),
            wall_time_s=float(d.get("wall_time_s", 0.0)),
            seed_trees_original=d.get("seed_trees_original", {}),
            seed_trees_effective=d.get("seed_trees_effective", {}),
            val_hypervolume=d.get("val_hypervolume"),
            test_hypervolume=d.get("test_hypervolume"),
            hv_reference_point=d.get("hv_reference_point"),
        )
    return loaded


def _write_failure_log(output_dir: Path, template_name: str, exc: BaseException):
    """Commit 6.6: failed-template traceback log for post-mortem."""
    import io as _io
    import traceback
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"failed_{template_name}.log"
    buf = _io.StringIO()
    buf.write(f"Template: {template_name}\n")
    buf.write(f"Exception: {type(exc).__name__}: {exc}\n\n")
    buf.write("Traceback:\n")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=buf)
    _atomic_write_text(log_path, buf.getvalue())


def _serialize_metrics(m: GenerationMetrics) -> Dict:
    return asdict(m)


def _provenance_record(config: Optional[ExperimentConfig] = None,
                       grammar: Optional[Grammar] = None,
                       splits: Optional[Dict[str, pd.DataFrame]] = None,
                       ref_dirs: Optional[np.ndarray] = None,
                       l1_sha256: Optional[str] = None,
                       l1_size_bytes: Optional[int] = None) -> Dict:
    """Audit-trail metadata about the algorithm stack + data used by this run.

    Model QA (Commit 5 + Commit 6.7 + Commit 6.8 reviews) required: NSGA-III
    niching algorithm identity, library versions, seed derivation, L1
    Parquet provenance + realized split boundaries + grammar signature +
    ref_dirs count + requirements-l2.txt SHA must be recorded in
    results.json so an auditor with only the artifacts can reconstruct
    the run without guessing. Commit 6.8: l1_sha256 / l1_size_bytes are
    passed in (computed at LOAD time in run_experiment) to close the
    TOCTOU window where the Parquet could be modified mid-run.
    """
    out: Dict = dict(_env_snapshot())  # python_version, pymoo, numpy, scipy, pandas, pyarrow
    out.update({
        "niching_algorithm": (
            "pymoo.algorithms.moo.nsga3 {HyperplaneNormalization, "
            "associate_to_niches, calc_niche_count, niching} — "
            "canonical Deb & Jain (2014) Algorithm 1"
        ),
        "niching_rng_derivation": (
            "np.random.default_rng(config.seed) per evolve() call; one "
            "stream reused across generations of a single template"
        ),
        "per_template_seed_derivation": (
            "BLAKE2b(str(master_seed)|template_name) → uint32 via "
            "experiment._derive_template_seed"
        ),
        # Commit 6.5: H2 primary outcome specification
        "h2_primary_outcome": (
            "hypervolume on validation split using pymoo.indicators.hv.HV "
            "with FIXED reference point (pre-declared, derived from "
            "worst-feasible fitness values, NOT from observed data)"
        ),
        "h2_hv_reference_point": list(H2_HV_REFERENCE_POINT),
        "h2_objectives_order":   list(DEFAULT_OBJECTIVES),
    })
    # Commit 6.7 additions — auditability R1 (L1 Parquet identity),
    # R2 (realized split boundaries), R3 (ref_dirs count), R9 (grammar hash)
    # Commit 6.8: prefer the load-time l1_sha256 passed in (TOCTOU fix)
    if config is not None:
        parquet_path_abs = Path(config.l1_parquet_path).resolve()
        if l1_sha256 is not None:
            out["l1_parquet_sha256"]      = l1_sha256
            out["l1_parquet_size_bytes"]  = l1_size_bytes
            out["l1_parquet_path"]        = str(parquet_path_abs)
            out["l1_parquet_hash_timing"] = "at_load_time"  # TOCTOU-safe
        elif parquet_path_abs.exists():
            # Fallback: hash at save time (TOCTOU window exists here)
            out["l1_parquet_sha256"]      = _file_sha256(parquet_path_abs)
            out["l1_parquet_size_bytes"]  = parquet_path_abs.stat().st_size
            out["l1_parquet_path"]        = str(parquet_path_abs)
            out["l1_parquet_hash_timing"] = "at_save_time"
    if splits is not None:
        out["split_boundaries"] = _split_boundaries(splits)
    if ref_dirs is not None:
        out["n_ref_dirs"] = int(ref_dirs.shape[0])
    if grammar is not None:
        out["grammar_signature"] = _grammar_signature(grammar)
    # Commit 6.8: requirements-l2.txt SHA for dep-pin integrity
    req_path = Path(__file__).parent.parent / "requirements-l2.txt"
    if req_path.exists():
        out["requirements_l2_sha256"] = _file_sha256(req_path)

    # H2 pre-reg v4 additions (audit trail completeness):
    # git_sha — repo HEAD at L2 run time (this commit)
    # l1_checkpoint_sha256 — SHA256 of the SSL encoder used to produce the
    # L1 Parquet; read from the Parquet's sidecar
    # (`<parquet>.provenance.json`) so the L2 run
    # doesn't need access to the checkpoint file.
    # l1_generator_git_sha — repo HEAD when the L1 Parquet was generated;
    # diverges from `git_sha` if the GP runs after
    # additional commits land.
    out["git_sha"] = _git_sha_l2()
    if config is not None:
        sidecar_path = Path(config.l1_parquet_path).resolve().with_suffix(
            ".provenance.json"
        )
        if sidecar_path.exists():
            try:
                with sidecar_path.open("r") as _f:
                    _sidecar = json.load(_f)
                if "checkpoint_sha256" in _sidecar:
                    out["l1_checkpoint_sha256"] = _sidecar["checkpoint_sha256"]
                if "git_sha" in _sidecar:
                    out["l1_generator_git_sha"] = _sidecar["git_sha"]
                if "checkpoint_path" in _sidecar:
                    out["l1_checkpoint_path"] = _sidecar["checkpoint_path"]
            except Exception as _exc:  # noqa: BLE001
                out["l1_sidecar_read_error"] = repr(_exc)

    # v8 Fix 2 (C7/M1): record PCA basis SHA + verify it was fit against the
    # same L1 checkpoint the L2 run is using. Cross-check closes a retroactive
    # basis-swap audit gap: without this, someone could refit PCA on different
    # data and L2 runs wouldn't notice.
    _pca_bases_manifest = (
        Path(__file__).resolve().parent / "pca_bases_manifest.json"
    )
    if _pca_bases_manifest.exists():
        try:
            with _pca_bases_manifest.open("r") as _f:
                _pm = json.load(_f)
            out["pca_bases_sha256"] = _pm.get("pca_bases_sha256", "")
            out["pca_bases_n_components"] = _pm.get("n_components", None)
            out["pca_bases_train_end"] = _pm.get("train_end", None)
            out["pca_bases_n_fit_windows"] = _pm.get("n_fit_windows", None)
            _pca_l1_sha = _pm.get("l1_checkpoint_sha256", "")
            # Cross-check: PCA basis was fit against the SAME L1 checkpoint
            # the current L2 run is reading features from.
            _l2_l1_sha = out.get("l1_checkpoint_sha256", "")
            if _pca_l1_sha and _l2_l1_sha and _pca_l1_sha != _l2_l1_sha:
                out["pca_bases_l1_checkpoint_mismatch"] = {
                    "pca_manifest_says":   _pca_l1_sha,
                    "l1_parquet_sidecar_says": _l2_l1_sha,
                    "severity": "BLOCKER — refit PCA bases against the "
                                "current L1 checkpoint before running.",
                }
        except Exception as _exc:  # noqa: BLE001
            out["pca_bases_manifest_read_error"] = repr(_exc)
    return out


def _git_sha_l2() -> str:
    """Current repo HEAD commit SHA for audit-trail purposes.

    Returns empty string if not in a git repo or git is unavailable — never
    raises. The L2 run still works without git (e.g. tarball deployments)
    but the provenance record will note the absence by emitting ''.
    """
    import subprocess as _sp
    try:
        return _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=_sp.DEVNULL,
            timeout=10,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return ""


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` via a temp file + os.replace.

    H2 pre-reg v4: all L2 output artifacts (results.json, per-template
    summaries, strategies JSONL, metrics JSON, run_fingerprint, error
    logs) must appear complete-or-absent to any reader. A concurrent
    dashboard or pilot aggregator scanning output_dir while a template
    is being written would otherwise see a truncated file mid-flush.
    Using tmp + os.replace makes the visible transition atomic on POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as _f:
        _f.write(content)
        _f.flush()
        try:
            os.fsync(_f.fileno())  # force bytes to disk before rename
        except (OSError, AttributeError):
            pass  # non-POSIX or unsupported fd; best-effort
    os.replace(tmp, path)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 of a file (streaming). Auditability R1 — allows the auditor
    to verify they're looking at the same L1 Parquet that fed the run."""
    import hashlib as _hashlib
    h = _hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _split_boundaries(splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Realized split date boundaries (Auditability R2). Unlike the
    config's train_end / val_end (which are REQUESTED cutoffs), this
    records the ACTUAL first and last date that landed in each split —
    the authoritative post-embargo boundaries."""
    out: Dict[str, Dict] = {}
    for name, df in splits.items():
        if not isinstance(df, pd.DataFrame):
            continue  # skip non-DataFrame metadata (e.g. split_by_date's vix_prior)
        if df is None or len(df) == 0 or "date" not in df.columns:
            out[name] = {"n_rows": 0, "n_dates": 0, "date_min": None, "date_max": None}
            continue
        dates = sorted(df["date"].astype(str).unique())
        out[name] = {
            "n_rows":   int(len(df)),
            "n_dates":  len(dates),
            "date_min": dates[0],
            "date_max": dates[-1],
        }
    return out


def _grammar_signature(grammar: Grammar) -> Dict:
    """Grammar function-set + terminal-set hash. Auditability R9 — detects
    a future grammar mutation that would silently change semantics without
    touching results.json otherwise."""
    import hashlib as _hashlib
    func_names = tuple(sorted(f.name for f in grammar.functions))
    term_names = tuple(sorted(t.name for t in grammar.terminals))
    sig_str = f"functions={func_names}|terminals={term_names}"
    return {
        "n_functions":    len(func_names),
        "n_terminals":    len(term_names),
        "max_depth":      grammar.max_depth,
        "max_nodes":      grammar.max_nodes,
        "function_names": list(func_names),
        "terminal_names": list(term_names),
        "sha256":         _hashlib.sha256(sig_str.encode()).hexdigest()[:16],
    }


def _save_experiment_result(result: ExperimentResult, output_dir: Path,
                            config: Optional[ExperimentConfig] = None,
                            grammar: Optional[Grammar] = None,
                            splits: Optional[Dict[str, pd.DataFrame]] = None,
                            ref_dirs: Optional[np.ndarray] = None,
                            l1_sha256: Optional[str] = None,
                            l1_size_bytes: Optional[int] = None,
                            warnings: Optional[List[Dict]] = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Top-level results.json — atomic write so concurrent aggregators never
    # see a half-flushed file (v4).
    results_json = output_dir / "results.json"
    _atomic_write_text(results_json, json.dumps({
        "config": result.config,
        "provenance": _provenance_record(
            config=config, grammar=grammar, splits=splits,
            ref_dirs=ref_dirs, l1_sha256=l1_sha256,
            l1_size_bytes=l1_size_bytes,
        ),
        "warnings": warnings or [],  # 6.8 R7 — explicit warnings channel
        "n_train_rows": result.n_train_rows,
        "n_val_rows": result.n_val_rows,
        "n_test_rows": result.n_test_rows,
        "total_wall_time_s": result.total_wall_time_s,
        "templates": [
            {
                "name": tr.template_name,
                "pareto_front_size": tr.pareto_front_size,
                "final_population_size": tr.final_population_size,
                "wall_time_s": tr.wall_time_s,
                "fitness_cache_stats": tr.fitness_cache_stats,
                "best_objectives": (
                    tr.metrics_log[-1]["best_per_objective"]
                    if tr.metrics_log else []
                ),
                # Audit trail: declared seeds vs. seeds actually used
                # after condition-specific substitution. Auditor reading
                # this file can reconstruct the initial population.
                "seed_trees_original":  tr.seed_trees_original,
                "seed_trees_effective": tr.seed_trees_effective,
                # Commit 6.5: H2 primary outcome per (condition, seed,
                # template) cell. val_hypervolume is the scalar the
                # Mann-Whitney U test operates on.
                "val_hypervolume":   tr.val_hypervolume,
                "test_hypervolume":  tr.test_hypervolume,
                "hv_reference_point": tr.hv_reference_point,
            }
            for tr in result.template_results
        ],
    }, indent=2))
    # Per-template strategy archive + metrics are now written incrementally
    # by _save_single_template_result (Commit 6.6). Still ensure they exist
    # for any template result that was resumed or constructed in-memory
    # without having been saved yet (defensive for programmatic callers).
    for tr in result.template_results:
        strat_path = output_dir / f"strategies_{tr.template_name}.jsonl"
        metrics_path = output_dir / f"metrics_{tr.template_name}.json"
        summary_path = output_dir / f"template_{tr.template_name}.json"
        if not (strat_path.exists() and metrics_path.exists()
                and summary_path.exists()):
            _save_single_template_result(tr, output_dir)
    print(f"  wrote {results_json}, {len(result.template_results)} templates")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layer2.experiment",
        description="L2 STVGP evolution experiment runner",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a single GP experiment")
    run_p.add_argument("--l1-parquet", required=True, help="Path to L1Output Parquet")
    run_p.add_argument("--condition", default="real-l1",
                       choices=["real-l1", "shuffled-l1", "scalar-only", "probes-only", "emb-only"])
    run_p.add_argument("--templates", default="",
                       help="Comma-separated template names (default: all)")
    run_p.add_argument("--pop-size", type=int, default=256)
    run_p.add_argument("--generations", type=int, default=100)
    run_p.add_argument("--seed", type=int, default=42,
                       help="Master seed — per-template GP seeds and shuffle seed derive from this")
    run_p.add_argument("--shuffle-seed", type=int, default=None,
                       help="Optional independent shuffle seed. If omitted, uses seed + 1_000_000. "
                            "Only consulted when condition=shuffled-l1.")
    run_p.add_argument("--n-partitions", type=int, default=12)
    run_p.add_argument("--min-trades", type=int, default=10,
                       help="Minimum trades an individual must produce to avoid "
                            "FAILED_FITNESS_SENTINEL (B2 gate). Default=10.")
    run_p.add_argument("--train-end", default="2024-09-30")
    run_p.add_argument("--val-end", default="2025-01-31")
    run_p.add_argument("--embargo-days", type=int, default=5)
    run_p.add_argument("--test-end", default=None,
                       help="Test split end date (default: None)")
    run_p.add_argument("--output", default=None,
                       help="Output directory (default: results/<timestamp>/)")
    run_p.add_argument("--minute", action="store_true", default=False,
                       help="Use 1-minute Parquet with vectorized evaluator. "
                            "Expects l1_minute_scalars.parquet (scalar-only, "
                            "357K rows). ~30-50x faster than per-bar evaluator.")
    run_p.add_argument("--cost-multiplier", type=float, default=1.0,
                       help="Scale all entry/exit costs by this factor. "
                            "1.0=baseline, 1.5=stress test at 150%% costs.")
    run_p.add_argument("--grammar-max-nodes", type=int, default=15,
                       help="Max nodes per tree (default=15).")
    run_p.add_argument("--level-b", action="store_true", default=False,
                       help="Use Level B base templates (5 templates with delta_tree).")
    run_p.add_argument("--per-fold-seeds", action="store_true", default=False,
                       help="P1-A: re-derive Level-B entry seeds per fold on train data (no leak).")
    run_p.add_argument("--no-inner-val", action="store_true", default=False,
                       help="Disable inner validation monitoring during evolution.")
    run_p.add_argument("--no-regime-gate", action="store_true", default=False,
                       help="Disable per-regime Sharpe hard gate.")
    run_p.add_argument("--regime-gate-margin-k", type=float, default=1.0,
                       help="Noise-aware per-regime gate margin in combined-SE units "
                            "(reject only if worse than random by >k SE; 0=old hard "
                            "threshold). Default 1.0.")

    # walk-forward subcommand
    wf_p = sub.add_parser("walk-forward",
                                  help="Walk-forward GP evolution (SVP-001)")
    wf_p.add_argument("--l1-parquet", required=True)
    wf_p.add_argument("--condition", default="scalar-only",
                       choices=["real-l1", "shuffled-l1", "scalar-only", "probes-only", "emb-only"])
    wf_p.add_argument("--templates", default="")
    wf_p.add_argument("--pop-size", type=int, default=256)
    wf_p.add_argument("--generations", type=int, default=100)
    wf_p.add_argument("--seed", type=int, default=42)
    wf_p.add_argument("--minute", action="store_true", default=False)
    wf_p.add_argument("--folds", default="1,2,3,4",
                       help="Comma-separated fold IDs to run (default: 1,2,3,4)")
    wf_p.add_argument("--output", default=None,
                       help="Base output directory (default: results/wf_<condition>_<timestamp>/)")
    wf_p.add_argument("--cost-multiplier", type=float, default=1.0,
                       help="Scale all entry/exit costs (1.0=baseline, 1.5=stress test).")
    wf_p.add_argument("--grammar-max-nodes", type=int, default=15,
                       help="Max nodes per tree (default=15).")
    wf_p.add_argument("--level-b", action="store_true", default=False,
                       help="Use Level B base templates (5 templates with delta_tree).")
    wf_p.add_argument("--per-fold-seeds", action="store_true", default=False,
                       help="P1-A: re-derive Level-B entry seeds per fold on train data (no leak).")
    wf_p.add_argument("--no-inner-val", action="store_true", default=False,
                       help="Disable inner validation monitoring during evolution.")
    wf_p.add_argument("--no-regime-gate", action="store_true", default=False,
                       help="Disable per-regime Sharpe hard gate.")
    wf_p.add_argument("--regime-gate-margin-k", type=float, default=1.0,
                       help="Noise-aware per-regime gate margin in combined-SE units "
                            "(reject only if worse than random by >k SE; 0=old hard "
                            "threshold). Default 1.0.")
    wf_p.add_argument("--probe-bundle", default=None,
                       help="Path to probe_fitting_bundle.npz for per-fold probe refit "
                            "(required for Conditions B/D with early folds).")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        template_names = tuple(
            n.strip() for n in args.templates.split(",") if n.strip()
        )
        output_dir = args.output or f"results/run_{int(time.time())}"
        config = ExperimentConfig(
            l1_parquet_path=args.l1_parquet,
            condition=args.condition,
            template_names=template_names,
            pop_size=args.pop_size,
            n_generations=args.generations,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            n_partitions=args.n_partitions,
            min_trades=args.min_trades,
            train_end=args.train_end,
            val_end=args.val_end,
            test_end=args.test_end,
            embargo_days=args.embargo_days,
            output_dir=output_dir,
            use_vectorized=args.minute,
            cost_multiplier=args.cost_multiplier,
            grammar_max_nodes=args.grammar_max_nodes,
            level_b=args.level_b,
            per_fold_seeds=getattr(args, 'per_fold_seeds', False),
            inner_val_enabled=not getattr(args, 'no_inner_val', False),
            regime_gate_enabled=not getattr(args, 'no_regime_gate', False),
            regime_gate_margin_k=getattr(args, 'regime_gate_margin_k', 1.0),
        )
        run_experiment(config)
        return 0
    if args.cmd == "walk-forward":
        template_names = tuple(
            n.strip() for n in args.templates.split(",") if n.strip()
        )
        fold_ids = [int(f.strip()) for f in args.folds.split(",")]
        folds = [f for f in WALK_FORWARD_FOLDS if f["fold_id"] in fold_ids]
        if not folds:
            print(f"No matching folds for IDs: {fold_ids}")
            return 1
        output_base = args.output or f"results/wf_{args.condition}_{int(time.time())}"
        base_config = ExperimentConfig(
            l1_parquet_path=args.l1_parquet,
            condition=args.condition,
            template_names=template_names,
            pop_size=args.pop_size,
            n_generations=args.generations,
            seed=args.seed,
            use_vectorized=args.minute,
            cost_multiplier=args.cost_multiplier,
            grammar_max_nodes=args.grammar_max_nodes,
            level_b=args.level_b,
            per_fold_seeds=getattr(args, 'per_fold_seeds', False),
            probe_bundle_path=args.probe_bundle,
            inner_val_enabled=not getattr(args, 'no_inner_val', False),
            regime_gate_enabled=not getattr(args, 'no_regime_gate', False),
            regime_gate_margin_k=getattr(args, 'regime_gate_margin_k', 1.0),
        )
        run_walk_forward(base_config, folds=folds, output_base=output_base)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
