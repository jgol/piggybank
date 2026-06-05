"""Multi-objective fitness evaluation for NSGA-III over template-based GP trees.

Wraps vectorized backtester (production) or `MultiLegOptionsBacktester` (legacy)
to produce NumPy fitness vectors compatible with pymoo's NSGA-III interface.
Handles:
  - Multi-objective vector construction (Sharpe, Sortino, trade_count_score)
  - NaN/Inf guards (any non-finite objective → -1e6 sentinel that NSGA-III
    will dominate against)
  - Tree-hash caching within a single GP run (Software Architect recommendation
    — 20-40% hit rate at population=256, gen=100 with NSGA-III elite preservation)
  - Quality gates: fire rate, exit utilization, random-entry null (nursery-ramped),
    churning (signal + stop-loss), max-hold (90%), clock-exit (std<3)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from layer2.evaluator import (
    BacktestResult, MultiLegOptionsBacktester,
)
from layer2.grammar import Node, canonical_key, to_str
from layer2.templates import Template


# ---------------------------------------------------------------------------
# Objective configuration
# ---------------------------------------------------------------------------

# 3-objective configuration (restored from design).
# NSGA-III minimizes; we negate "good" objectives so all point in the same
# direction.
#
# neg_win_rate REMOVED: correlated with Sharpe; adds a 4th objective without
# independent information, inflating the Pareto front and diluting selection.
#
# neg_exit_utilization: measures fraction of trades closed by the exit signal
# (vs MAX HOLD / stop-loss / EOD). Strategies with 0% exit utilization have
# vestigial exit trees — "junk DNA" that the GP carries without selection
# pressure. This was previously removed with the incorrect justification that
# "hold to expiry IS the economically optimal exit for credit spreads."
# That is WRONG: hold-to-expiry maximizes gamma exposure at the worst time
# (expiry), forgoes profit-taking on decayed credits, and ignores regime
# changes. The structural degeneracy filter catches SOME dead exits but not
# all (e.g., CrossAbove(Sqrt(EphReal(x)), terminal) slipped through).
# EXIT UTILIZATION SHOULD BE RESTORED as a fitness objective or hard gate.
# With 3 objectives and n_partitions=12: Das-Dennis gives 91 reference
# directions — well-matched to pop=256. The previous 5-objective config
# gave 1820 ref dirs vs pop=256, collapsing selection to random.
DEFAULT_OBJECTIVES: Tuple[str, ...] = (
    "neg_sharpe",            # minimize → maximize Sharpe
    "neg_sortino",           # minimize → maximize Sortino (tail-risk sensitive)
    "neg_trade_count_score", # minimize → maximize trade frequency (0.3-3/day sweet spot)
    # REVERTED: neg_exit_utilization and neg_conditional_sharpe_gap were objectives
    # but created a zero-profit Pareto attractor. Zero-Sharpe strategies with
    # 100% exit_util produced [0,0,-1,0] which was Pareto-optimal — no positive-
    # Sharpe strategy could dominate them. 196/203 BCC front members had this
    # identical fitness, killing selection pressure. Both are now GATES only.
    # NOT ADDED: max_drawdown as a 4th objective (considered 2026-06-01) — same
    # attractor failure mode (a low-drawdown ~zero-Sharpe strategy is non-dominated
    # on the drawdown axis). Drawdown enters as a SOFT PENALTY on the return axes
    # instead (DRAWDOWN_PENALTY_* below), which shapes the search without adding a
    # Pareto axis. Absolute ruin is gated at the QC deployment stage (the proxy's
    # equity floor censors true ruin — it cannot be a proxy gate; see calibration).
)

# Soft drawdown penalty (2026-06-01): added to neg_sharpe ONLY (MH4 fix — Sharpe and
# Sortino are ~0.99 correlated, so penalizing both double-weighted it) so high-
# drawdown strategies are pushed DOWN the primary return axis — no new objective,
# no attractor. Uses uncapped drawdown / avg_position_size (sizing-exploit-proof).
# These are MILD within-template RANK pressure, NOT absolute thresholds: proxy
# drawdown magnitude does not transfer to QC (template-dependent + intraday-margin
# censored), so an absolute drawdown gate is ungrounded — but the within-template
# RANK transfers, which is all a soft nudge needs. A strategy at the free level pays
# nothing; RPB-like adj-dd≈6 pays ~1.0 Sharpe (significant but not overwhelming).
DRAWDOWN_FREE_LEVEL: float = 1.0      # adj-drawdown below which no penalty
DRAWDOWN_PENALTY_WEIGHT: float = 0.20  # Sharpe penalty per unit adj-drawdown above free

# Sentinel value for failed evaluations (NaN/Inf protection). Large positive
# value ensures NSGA-III dominates against this strategy on every objective.
FAILED_FITNESS_SENTINEL = 1e6

# 2026-06-02 (finding #4): graded-infeasibility span. An infeasible individual's
# fitness is FAILED_FITNESS_SENTINEL + violation, where violation in [0, SPAN)
# encodes HOW FAR from feasible it is (0 = just barely infeasible, ~SPAN = maximally
# infeasible). A less-violating infeasible now Pareto-dominates a more-violating one,
# so NSGA-III keeps a selection gradient toward feasibility instead of a FLAT 1e6
# landscape (the flat landscape was why all-infeasible folds, e.g. iron_condor fold 1,
# drifted to 0/256 feasible: identical fitness vectors -> non-dominated sort ranks
# everyone equal -> random drift, no climb).
#
# DOMINANCE PRESERVED (the load-bearing invariant, verified in code review): a
# feasible individual can NEVER be dominated by a 1e6-based infeasible. NOT because
# every feasible objective is small — `neg_sharpe` is actually UNBOUNDED above (the
# `max(std,1e-9)` Sharpe floor can blow it past 1e6 for a near-zero-variance artifact)
# — but because the OTHER TWO objectives are hard-bounded far below 1e6:
# `neg_sortino` is clipped to [-10,10] (+ a small drawdown penalty) and
# `neg_trade_count_score` lives in [-1,1]. To DOMINATE, the infeasible must be <= on
# ALL THREE axes; on those two bounded axes the feasible is always << 1e6, so the
# infeasible can at best tie on `neg_sharpe` — never dominate. SPAN (1.0) << that gap.
# (The unbounded-`neg_sharpe` floor artifact is a separate pre-existing issue — it can
# put a phantom near-zero-variance individual on the front; tracked for its own fix.)
_INFEASIBLE_GRADE_SPAN = 1.0


def _graded_infeasible(n_obj: int, violation: float) -> np.ndarray:
    """Infeasible fitness vector: the 1e6 sentinel base + a small graded term in
    [0, _INFEASIBLE_GRADE_SPAN) for the constraint-violation magnitude.

    ``violation`` is clipped to [0, 1]: 0 = barely infeasible (just outside a gate),
    1 = maximally infeasible. The same term is added to every objective so a
    less-violating individual Pareto-dominates a more-violating one. Feasible
    individuals (objectives ~ <= +5) remain strictly dominant over ALL infeasibles.
    """
    v = float(np.clip(violation, 0.0, 1.0)) * _INFEASIBLE_GRADE_SPAN
    return np.full(n_obj, FAILED_FITNESS_SENTINEL + v)


def _count_ite(node) -> int:
    """Count IfThenElse nodes in a tree (opacity metric).
    Delegates to grammar._count_ite_nodes for DRY."""
    from layer2.grammar import _count_ite_nodes
    return _count_ite_nodes(node)


def trade_count_score(total_trades: int, n_days: int) -> float:
    """Shared trade-count scoring: flat plateau 0.3-3.0 trades/day (=1.0).

    Dead zone below 0.1 trades/day (1 trade per 10 days) returns -0.5,
    making ultra-low-trade strategies strongly dominated by NSGA-III even
    if their Sharpe is high. This prevents the BPC_wf1 failure mode where
    9 trades in 380 days achieved +2.46 test Sharpe on pure noise.

    MEDIUM-11 (2026-06-01 audit) — the three trade-frequency specs serve
    DISTINCT roles and must not be read as one target:
      * THIS optimizer objective shapes toward 0.3-3.0/day (the grounded
        0DTE defined-risk regime; QC fires ~0.75/day post eval-schedule fix).
        Reward profile (per trades/day): dead zone < 0.1 -> -0.5; partial-reward
        RAMP 0.1->0.3 (linear 0->1); FULL-REWARD plateau 0.3-3.0 (=1.0); decay
        3.0->8.0; zero >= 8.0.
      * experiment.py G3 (`G3_min_trades_val`=20, lowered from 40 on 2026-06-02
        finding (b)) and the per-individual min_trades=20 gate are STATISTICAL-
        POWER / selectivity-REACH floors (enough val trades for a meaningful
        Sharpe), NOT a frequency preference. CORRECTION (M4, 2026-06-02 audit):
        20 trades over a ~120-125-day val window = ~0.16-0.17/day, which sits in
        this objective's PARTIAL-REWARD RAMP (0.1-0.3), NOT at the full-reward
        plateau lower edge (0.3/day = ~37-38 trades/125 days). This is BY DESIGN:
        the gate's job is a minimum-sample selectivity reach, deliberately set
        BELOW the tc-objective's full-reward band so a genuinely selective low-
        frequency champion (partial-but-positive tc reward) is admitted rather
        than rejected. The earlier note that 20/40 trades aligned to the full-
        reward plateau LOWER EDGE was inaccurate — only ~37+ trades reach the
        plateau; 20 trades earn partial (ramp) tc reward, which is intended.
      * scripts/pilot_to_qc_handoff.py F4 deploy filter is realigned (2026-06-01) to
        `(0.3, 8.0)/day` = THIS objective's FULL-REWARD plateau (0.3-3.0) extended to
        its zero-point (8.0); it excludes the (0.1, 0.3) partial-reward ramp as a thin-
        sample noise risk. Deploy filter and search objective now agree (was a stale
        `(3,20)` band pre eval-schedule fix that rejected every realistic champion).
    """
    if total_trades == 0:
        return -1.0
    trades_per_day = total_trades / max(n_days, 1)
    if trades_per_day < 0.1:
        return -0.5  # dead zone: too few trades for statistical validity
    elif trades_per_day <= 0.3:
        return (trades_per_day - 0.1) / 0.2  # linear ramp: 0.0 at 0.1, 1.0 at 0.3
    elif trades_per_day <= 3.0:
        return 1.0  # flat plateau from 0.3 to 3.0 trades/day
    elif trades_per_day <= 8.0:
        return max(0.0, 1.0 - (trades_per_day - 3.0) / 5.0)
    else:
        return 0.0

# Minimum trades an evaluation must produce to earn a real fitness vector.
# Below this, the strategy is degenerate (typically a tree that never fires
# entry, giving fitness (0, 0, +1) which is incomparable with productive
# strategies on max_drawdown=0 and thus always survives on the Pareto
# front). B2 fix: gate zero-trade strategies with FAILED_FITNESS_SENTINEL
# on every objective so NSGA-III correctly dominates against them.
DEFAULT_MIN_TRADES = 1


# ---------------------------------------------------------------------------
# Evolution-level trial recorder (DSR gate, P1-B redesign)
# ---------------------------------------------------------------------------
# The Deflated-Sharpe-Ratio front gate must deflate champions against the
# WHOLE evolution's distinct trials, not the tiny final Pareto front. (Using
# the final front inverts the variance: a single overfit spike inflates V and
# clears its own bar — see Blocker B in the P1-B redesign.) To compute the
# correct (N, V) we record, across every generation, the TRAINING annualized
# Sharpe of each genuinely-distinct individual evaluated during the run.
#
# Distinctness is by `canonical_key` over the (entry, exit, size, delta) quad —
# the same commutative/canonical signature the per-generation and final-front
# dedups use — so N counts genuinely-distinct strategy configurations and is
# NOT padded by commutative twins. The record stores RAW backtest moments
# (sharpe_ann, n_days) BEFORE any quality/regime gate, because a strategy that
# ran a backtest IS a multiple-testing trial regardless of whether it later
# passed the fitness gates. (Members are filtered by n_days >= min_days at the
# gate; that is the only validity filter — see dsr_gate_evolution in pbo.py.)
#
# Recording is opt-in: the sink dict is None unless `enable_trial_recording()`
# is called (gp_engine.evolve does this at entry), so non-evolution evaluate()
# calls (val/test scoring, unit tests) pay nothing.

def _canonical_signature(entry: Node, exit_: Node, size: Node,
                         delta: "Optional[Node]" = None) -> str:
    """Canonical (commutative-collapsed) signature of a tree quad.

    Matches the dedup key used by gp_engine's per-generation phenotypic dedup
    and experiment.py's final-front dedup, so the recorded distinct-individual
    count N is consistent with the deduped front size.
    """
    return (
        canonical_key(entry) + "|" + canonical_key(exit_) + "|"
        + canonical_key(size) + "|"
        + (canonical_key(delta) if delta is not None else "")
    )


def _record_trial(sink: "Optional[Dict[str, Tuple[float, int]]]",
                  sig: str, sharpe_ann: float, n_days: int) -> None:
    """Record a distinct individual's RAW training Sharpe into the evo sink.

    First-seen wins (deterministic; avoids dependence on per-generation noise
    realizations when noise injection is active). Phantom/sentinel Sharpes
    (the -1e6 churning/clock-exit flag, or non-finite values) are skipped:
    they are not genuine Sharpe estimates and would corrupt V.
    """
    if sink is None or sig in sink:
        return
    if not np.isfinite(sharpe_ann) or sharpe_ann <= -1e5:
        return
    sink[sig] = (float(sharpe_ann), int(n_days))


# ---------------------------------------------------------------------------
# FitnessEvaluator — caches + multi-objective vector
# ---------------------------------------------------------------------------

@dataclass
class FitnessEvaluator:
    """Compute NSGA-III fitness vectors for evolved GP strategy trees.

    Each evaluator is template-bound and run-scoped: instantiate one per
    (template × evolution-run) pair. Cache lives for the run's lifetime.

    Args:
        template: the strategy template (legs + structural metadata)
        data: training data DataFrame (loaded via layer2.io.load_l1_parquet)
        backtester_kwargs: forwarded to MultiLegOptionsBacktester constructor
        objectives: tuple of objective names (per OBJECTIVE_REGISTRY below).
            Default = 3-objective Step-2 config; cost-stability appended later.
    """
    template: Template
    data: pd.DataFrame
    backtester_kwargs: Dict = field(default_factory=dict)
    objectives: Tuple[str, ...] = DEFAULT_OBJECTIVES
    # min_trades gate operates on `result.total_trades` from the
    # backtester — it is the count of COMPLETED trades. Commit 6 M2's
    # MAX_CONTRACTS cap in MultiLegOptionsBacktester bounds contracts
    # PER TRADE (tail-exposure clip) and cannot suppress whether a
    # trade fires at all. So the two interact orthogonally: M2 limits
    # per-trade size; min_trades gates aggregate trade count.
    min_trades: int = DEFAULT_MIN_TRADES
    # random_entry_margin: stub attribute for compatibility with evolve()'s
    # nursery ramp logic (which reads fitness_evaluator.random_entry_margin).
    # Non-vectorized evaluator does not implement the random-entry gate,
    # but evolve() accesses the field unconditionally. Default 0.0 = no gate.
    random_entry_margin: float = 0.0

    # Internal state — initialized in __post_init__
    _backtester: MultiLegOptionsBacktester = field(init=False, repr=False)
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _eval_count: int = field(default=0, init=False, repr=False)
    _cache_hit_count: int = field(default=0, init=False, repr=False)
    # Data fingerprint, computed once at __post_init__. Full SHA-256 (captures
    # row contents) is used only as the cache-key prefix; the per-call
    # liveness check is a cheap identity+shape probe so repeated cached
    # evaluate() calls don't re-hash the entire DataFrame (v4: 220 K×60 ms
    # fingerprint recompute was the dominant GP runtime cost at pop=128).
    _data_fingerprint: str = field(default="", init=False, repr=False)
    _data_probe: Tuple = field(default=(), init=False, repr=False)
    # v5: hard identity pin — see `evaluate()` for rationale. Caught by the
    # Code Reviewer audit: without this, someone reassigning `fe.data = ...`
    # would silently invalidate the cache (id() might recycle; the probe
    # is a weaker invariant than "same DataFrame object").
    _data_ref: object = field(default=None, init=False, repr=False)
    # v9 cache-key triplet (#7b). Populated once at __post_init__ from the
    # bound DataFrame + on-disk artifacts. Stored on the instance so the
    # per-call key has zero runtime cost.
    _pca_bases_sha: str = field(default="", init=False, repr=False)
    _fold_recenter_sha: str = field(default="", init=False, repr=False)
    _fold_id: str = field(default="", init=False, repr=False)
    # Evolution-level DSR trial recorder (P1-B). None unless enabled by
    # gp_engine.evolve via enable_trial_recording(). Maps canonical tree-quad
    # signature -> (raw training annualized Sharpe, n_days), first-seen.
    _evo_trial_records: "Optional[Dict[str, Tuple[float, int]]]" = field(
        default=None, init=False, repr=False)

    def enable_trial_recording(self) -> "Dict[str, Tuple[float, int]]":
        """Start recording distinct-individual training Sharpes for the DSR
        gate. Idempotent; returns the live sink dict (shared by reference)."""
        if self._evo_trial_records is None:
            self._evo_trial_records = {}
        return self._evo_trial_records

    @property
    def evo_trial_records(self) -> "Dict[str, Tuple[float, int]]":
        """Recorded {signature: (sharpe_ann, n_days)} (empty if not enabled)."""
        return self._evo_trial_records or {}

    def __post_init__(self):
        self._backtester = MultiLegOptionsBacktester(
            self.template, **self.backtester_kwargs
        )
        # MEDIUM-6 (2026-06-01 audit): snapshot the post-ramp min_trades target so
        # the DSR trial recorder only records once the gp_engine ramp completes
        # (mirrors the vectorized evaluator). Non-production path, but kept
        # consistent so the same ramp bug cannot reappear if it is ever exercised.
        self._target_min_trades = int(self.min_trades)
        # Validate that all requested objectives are known
        for obj in self.objectives:
            if obj not in OBJECTIVE_REGISTRY:
                raise ValueError(
                    f"unknown objective {obj!r}; available: "
                    f"{list(OBJECTIVE_REGISTRY.keys())}"
                )
        # Compute the full SHA-256 once — used as cache-key prefix AND as
        # the source-of-truth for what counts as "the data this evaluator
        # was built on".
        self._data_fingerprint = self._compute_data_fingerprint(self.data)
        # Cheap probe used on every cached evaluate(): (id(data), shape,
        # tuple-of-column-names, tuple-of-dtype-strs). Catches (a) the
        # DataFrame being swapped out for another instance, (b) column
        # additions/removals, (c) dtype coercions. Does not catch silent
        # in-place mutation of numeric cell values — that's the tradeoff
        # for the per-call speedup.
        self._data_probe = self._compute_data_probe(self.data)
        # v5 identity pin: the DataFrame object itself. `is` comparison on
        # every evaluate() call loud-fails if `fe.data` is ever reassigned
        # (different object → different id, regardless of id() recycling).
        self._data_ref = self.data
        # v9: capture the artifact shas + fold_id once. Lookups are
        # best-effort — missing artifacts produce empty-string entries
        # (legacy/test paths). Production runs always have both shas
        # populated because experiment.py eager-loads PCA bases.
        try:
            from layer2.evaluator import (
                _load_pca_bases_sha, _load_fold_recenter_sha,
                _resolve_fold_ids_for_data, _dominant_fold_id,
            )
            self._pca_bases_sha = _load_pca_bases_sha()
            self._fold_recenter_sha = _load_fold_recenter_sha()
            try:
                fids = _resolve_fold_ids_for_data(self.data)
                self._fold_id = _dominant_fold_id(fids) or ""
            except Exception:
                self._fold_id = ""
        except Exception:
            self._pca_bases_sha = ""
            self._fold_recenter_sha = ""
            self._fold_id = ""

    def evaluate(self, entry_tree: Node, exit_tree: Node,
                 size_tree: Node, delta_tree: "Optional[Node]" = None,
                 stop_mult: "Optional[float]" = None) -> np.ndarray:
        """Run backtest + compute fitness vector. Returns (n_objectives,) ndarray.

        stop_mult is accepted for signature parity with
        VectorizedFitnessEvaluator but is a NO-OP on this legacy (non-vectorized)
        path: the SimpleBacktester/OptionsBacktester it drives has no evolvable
        stop-loss base. The evolved stop gene is only honoured by the vectorized
        evaluator (--minute). Templates carrying a delta_tree already raise here,
        so the 5 Level-B base templates never reach this branch in production.

        Cached by data-fingerprint-prefixed tree-hash triple — repeated
        evaluations of the same trees on the same underlying data (common
        with NSGA-III elitism across generations) hit cache.

        Per-call liveness check is a cheap (id, shape, cols, dtypes) probe;
        if the probe differs from the one captured at __post_init__ we
        escalate to the full SHA-256 comparison and fail loudly. This keeps
        the common path O(1) while still catching the "someone swapped
        self.data" footgun that motivated the original hash-on-every-call
        design.

        v5 hardening: `self.data is self._data_ref` identity check fires
        FIRST. If someone reassigns `fe.data = new_df`, the probe and SHA
        approaches both become unreliable (id() can recycle; a new
        DataFrame with the same shape/dtypes/columns would pass the
        probe but carry different cell values). The identity pin catches
        reassignment with certainty; reassignment should never happen in
        normal GP usage — if it does, it's an operator bug and we
        surface it loudly.
        """
        if self.data is not self._data_ref:
            raise RuntimeError(
                "FitnessEvaluator.data was reassigned since construction. "
                "Cached fitness values are no longer trustworthy — rebuild "
                "the FitnessEvaluator with the new DataFrame rather than "
                "mutating `fe.data` in place."
            )
        current_probe = self._compute_data_probe(self.data)
        if current_probe != self._data_probe:
            current_fp = self._compute_data_fingerprint(self.data)
            assert current_fp == self._data_fingerprint, (
                "FitnessEvaluator.data has been swapped/mutated since "
                "construction — cached fitness values are no longer valid. "
                "Rebuild the FitnessEvaluator with the new DataFrame "
                "instead of replacing it in place. "
                f"(fingerprint at __post_init__={self._data_fingerprint[:12]}…, "
                f"now={current_fp[:12]}…)"
            )
            # Probe-drifted but contents identical — refresh the probe so
            # we don't keep falling into the slow path.
            self._data_probe = current_probe
        if delta_tree is not None:
            raise ValueError(
                "FitnessEvaluator (non-vectorized) does not support Level B "
                "delta_tree. Use VectorizedFitnessEvaluator (--minute flag) "
                "for templates with delta_range."
            )
        cache_key = self._tree_hash(entry_tree, exit_tree, size_tree, delta_tree)
        if cache_key in self._cache:
            self._cache_hit_count += 1
            return self._cache[cache_key]
        return self._evaluate_uncached(
            entry_tree, exit_tree, size_tree, data=self.data,
            cache_key=cache_key,
        )

    def score_on_data(self, entry_tree: Node, exit_tree: Node,
                      size_tree: Node, data: pd.DataFrame,
                      terminal_data=None,
                      delta_tree: "Optional[Node]" = None,
                      stop_mult: "Optional[float]" = None,
                      ) -> np.ndarray:
        """Re-score a tree triple on a DIFFERENT data split (val / test).

        stop_mult accepted for signature parity; NO-OP on the legacy path
        (see evaluate()).
        """
        if delta_tree is not None:
            raise ValueError(
                "FitnessEvaluator (non-vectorized) does not support Level B "
                "delta_tree. Use VectorizedFitnessEvaluator (--minute flag)."
            )
        return self._evaluate_uncached(
            entry_tree, exit_tree, size_tree, data=data, cache_key=None,
        )

    def _evaluate_uncached(self, entry_tree: Node, exit_tree: Node,
                           size_tree: Node, data: pd.DataFrame,
                           cache_key: Optional[str]) -> np.ndarray:
        """Core evaluation path — NON-VECTORIZED. Limited gates.

        BLOCKED for multi-leg templates in production: fire rate,
        conditional Sharpe gap, random-entry null, churning, max-hold,
        and clock-exit gates are ONLY in the vectorized path.
        """
        if hasattr(self.template, 'legs') and len(self.template.legs) >= 2:
            import warnings
            warnings.warn(
                f"Non-vectorized evaluator used for multi-leg template "
                f"'{self.template.name}' — results use stale pricing model "
                f"(no skew, no stop-loss, no credit haircut). "
                f"Use --minute for production runs.",
                UserWarning, stacklevel=2,
            )
        if cache_key is not None:
            self._eval_count += 1
        try:
            result = self._backtester.run(
                entry_tree=entry_tree, exit_tree=exit_tree,
                size_tree=size_tree, data=data,
            )
            # P1-B: record the RAW training Sharpe for the evolution-level DSR
            # trial population (pre-gate; only on the training-data cache path).
            # BLOCKER B2 (2026-06-02 audit): record EVERY distinct finite-Sharpe
            # trial from gen 0, regardless of the gp_engine min_trades ramp. A
            # backtest that RAN is a multiple-testing trial whether or not it later
            # cleared min_trades; the prior `min_trades >= _target_min_trades`
            # (MEDIUM-6) guard delayed recording until the ramp completed (~gen 30),
            # undercounting N ~30x and leaving SR* far too lenient. V stays
            # protected by `evolution_n_v_from_records`'s n_days >= min_days filter.
            if (self._evo_trial_records is not None and cache_key is not None):
                _record_trial(
                    self._evo_trial_records,
                    _canonical_signature(entry_tree, exit_tree, size_tree),
                    result.sharpe, result.n_days,
                )
            # B2 gate: degenerate (zero/few-trade) strategies get
            # FAILED_FITNESS_SENTINEL on every objective.
            # Exit utilization gate: strategies where the exit signal NEVER
            # fires (0% utilization) have vestigial exit trees. These
            # achieve proxy Sharpe through pure theta harvesting + MAX HOLD
            # but lose catastrophically in live execution (gamma risk,
            # BS pricing error). Require >= 10% of exits via signal.
            if result.total_trades < self.min_trades:
                fitness_vec = np.full(len(self.objectives), FAILED_FITNESS_SENTINEL)
            elif result.total_trades >= 10 and result.exit_utilization < 0.10:
                fitness_vec = np.full(len(self.objectives), FAILED_FITNESS_SENTINEL)
            else:
                from layer2.grammar import node_count
                tn = node_count(entry_tree) + node_count(exit_tree) + node_count(size_tree)
                ite = _count_ite(entry_tree) + _count_ite(exit_tree)
                fitness_vec = self._fitness_from_result(result, total_nodes=tn, ite_count=ite)
        except Exception as exc:
            import sys
            _exc_cls = type(exc).__name__
            if not hasattr(self, '_logged_exc_classes'):
                self._logged_exc_classes = set()
            if _exc_cls not in self._logged_exc_classes:
                self._logged_exc_classes.add(_exc_cls)
                print(f"[FitnessEvaluator] Backtester exception ({_exc_cls}): {exc}",
                      file=sys.stderr)
            fitness_vec = np.full(len(self.objectives), FAILED_FITNESS_SENTINEL)
        # NaN/Inf guard
        fitness_vec = np.where(
            np.isfinite(fitness_vec), fitness_vec, FAILED_FITNESS_SENTINEL
        )
        if cache_key is not None:
            self._cache[cache_key] = fitness_vec
        return fitness_vec

    def _fitness_from_result(self, result: BacktestResult,
                            total_nodes: int = 0,
                            ite_count: int = 0) -> np.ndarray:
        """Project a BacktestResult onto the configured objective vector.

        Args:
            total_nodes: total nodes across entry+exit+size trees.
            ite_count: count of IfThenElse nodes in entry+exit trees.
                Each IfThenElse adds 0.05 Sharpe penalty (opacity cost).
        """
        raw = self._backtester.compute_fitness(result)
        # Quadratic parsimony: 0.005 × (nodes - 10)². Superlinear pressure
        # makes complex trees increasingly expensive (Poli & McPhee 2008).
        # At 12 nodes: 0.02, 15: 0.125, 20: 0.50, 30: 2.0 (prohibitive).
        parsimony_penalty = max(0, total_nodes - 10) ** 2 * 0.005
        # IfThenElse opacity penalty: each ITE adds 0.05 Sharpe.
        parsimony_penalty += ite_count * 0.05
        # Normalize drawdown by position size. Skip for non-vectorized path
        # which uses the default avg_position_size=0.5 (Trade doesn't carry size).
        _raw_dd = -raw["neg_max_drawdown"]
        if result.avg_position_size != 0.5:  # real value from vectorized path
            _adj_dd = min(_raw_dd / max(result.avg_position_size, 0.05), 1.0)
        else:
            _adj_dd = min(_raw_dd, 1.0)  # no normalization for non-vectorized
        # Translate "compute_fitness" outputs into NSGA-III "minimize"-style objectives.
        # NOTE: this LEGACY (non-vectorized) path does NOT apply the soft drawdown
        # penalty (DRAWDOWN_* in the production VectorizedFitnessEvaluator path) and
        # its "max_drawdown" is the 1.0-CAPPED adj_dd — different semantics from the
        # production path's uncapped value. Stale/non-production (no --minute run uses
        # it); left as-is intentionally. Do not read its "max_drawdown" as uncapped.
        derived = {
            "neg_sharpe": -raw["sharpe"] + parsimony_penalty,
            "neg_sortino": -result.sortino + parsimony_penalty,
            "max_drawdown": _adj_dd,
            "neg_trade_count_score": -raw["trade_count_score"],
            "neg_win_rate": -raw["win_rate"],
            "neg_exit_utilization": -result.exit_utilization,
            "neg_conditional_sharpe_gap": -result.conditional_sharpe_gap,
            # Raw passthroughs (in case caller wants direct):
            "sharpe": raw["sharpe"],
            "sortino": result.sortino,
            "win_rate": raw["win_rate"],
            "psr": result.psr,
        }
        vec = np.array(
            [derived[obj] for obj in self.objectives], dtype=np.float64
        )
        return vec

    def _tree_hash(self, entry: Node, exit_: Node, size: Node,
                   delta: "Optional[Node]" = None,
                   stop_mult: "Optional[float]" = None) -> str:
        """Canonical-form cache key for the tree triple/quad.

        Level B: includes delta_tree hash when present. stop_mult is accepted
        for signature parity with the vectorized evaluator; it is a no-op on
        this legacy path (stop gene unsupported) but is still folded into the
        key when supplied so a caller mixing stop levels cannot get a stale hit.
        """
        base = (
            f"D:{self._data_fingerprint}|"
            f"PCA:{self._pca_bases_sha}|"
            f"FRC:{self._fold_recenter_sha}|"
            f"FID:{self._fold_id}|"
            f"E:{to_str(entry)}|X:{to_str(exit_)}|S:{to_str(size)}"
        )
        if delta is not None:
            base += f"|DT:{to_str(delta)}"
        if stop_mult is not None:
            base += f"|SM:{float(stop_mult):.3f}"
        return base

    @staticmethod
    def _compute_data_probe(data: pd.DataFrame) -> Tuple:
        """O(1)+O(V) cheap probe: (id, shape, cols, dtype-strs).

        Detects data swaps, column additions/removals, and dtype coercions
        without hashing row contents. Used on every cached evaluate() call;
        the expensive SHA-256 is the slow-path confirmation if the probe
        disagrees with the one captured at __post_init__.
        """
        return (
            id(data),
            tuple(data.shape),
            tuple(data.columns),
            tuple(str(dt) for dt in data.dtypes),
        )

    @staticmethod
    def _compute_data_fingerprint(data: pd.DataFrame) -> str:
        """SHA-256 of a canonical byte serialization of `data`.

        Canonicalization captures:
          * shape,
          * column names + dtypes (order-sensitive),
          * raw value bytes via DataFrame.values with np.ascontiguousarray.

        Only object-dtype columns fall back to str-repr hashing (the L1
        output parquet has list-of-float columns for typed vectors which
        land as dtype=object). That path is O(N·V) per evaluator
        construction but runs once; subsequent cached calls re-run the
        whole fingerprint as the liveness check, so the implementation
        stays conservative on speed vs. correctness.
        """
        h = hashlib.sha256()
        h.update(f"shape:{data.shape}\n".encode("utf-8"))
        # Column names + dtypes — catches schema changes that keep shape.
        for col in data.columns:
            h.update(f"{col}:{data[col].dtype}\n".encode("utf-8"))
        # Values — split numeric vs object to avoid object-array pickle pitfalls.
        for col in data.columns:
            s = data[col]
            if s.dtype == object:
                # List-typed columns (e.g. EMB_SHARED) — hash repr.
                for v in s.to_numpy():
                    if isinstance(v, np.ndarray):
                        h.update(np.ascontiguousarray(v).tobytes())
                    else:
                        h.update(repr(v).encode("utf-8"))
            else:
                arr = np.ascontiguousarray(s.to_numpy())
                h.update(arr.tobytes())
        return h.hexdigest()

    @property
    def cache_hit_rate(self) -> float:
        total = self._eval_count + self._cache_hit_count
        return self._cache_hit_count / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "evaluations": self._eval_count,
            "cache_hits": self._cache_hit_count,
            "cache_size": len(self._cache),
            "cache_hit_rate": self.cache_hit_rate,
        }


# Objective registry — names that FitnessEvaluator recognizes. Used for
# validation only; the actual computation is in `_fitness_from_result`.
OBJECTIVE_REGISTRY: Dict[str, str] = {
    "neg_sharpe": "minimize → maximize Sharpe",
    "max_drawdown": "minimize drawdown directly",
    "neg_trade_count_score": "minimize → maximize trade-count score (plateau 0.3-3.0 trades/day)",
    "neg_win_rate": "minimize → maximize win rate",
    "sharpe": "raw Sharpe (when caller wants minimization elsewhere)",
    "win_rate": "raw win rate",
    "neg_sortino": "minimize → maximize Sortino ratio (downside-deviation-only, tail-risk sensitive)",
    "neg_exit_utilization": "minimize → maximize exit signal utilization (penalizes max_hold-only strategies)",
    "neg_conditional_sharpe_gap": "minimize → maximize conditional Sharpe gap (entry-day Sharpe minus all-day Sharpe)",
}


# ---------------------------------------------------------------------------
# Per-regime noise-aware gate (Change 2, task #34)
# ---------------------------------------------------------------------------

REGIME_MIN_TRADES = 10  # min trades per regime for gate to apply


def _compute_regime_terciles(
    terminal_data: Dict[str, np.ndarray],
) -> Optional[Tuple[float, float]]:
    """Compute tercile boundaries from training ATM_IV distribution.

    Returns (q33, q67) thresholds in normalized space, or None if ATM_IV
    unavailable. Terciles adapt to each fold's training data — unlike
    fixed thresholds, they produce equal-sized regime buckets regardless
    of the ATM_IV distribution in the training period.
    """
    atm_iv = terminal_data.get("ATM_IV")
    if atm_iv is None:
        return None
    finite = atm_iv[np.isfinite(atm_iv)]
    if len(finite) < 30:
        return None
    q33 = float(np.percentile(finite, 33.33))
    q67 = float(np.percentile(finite, 66.67))
    return (q33, q67)


def _regime_gate_fails(
    result: "BacktestResult",
    terminal_data: Dict[str, np.ndarray],
    regime_terciles: Optional[Tuple[float, float]] = None,
    random_entry_regime_sharpes: Optional[Dict[str, float]] = None,
    random_entry_regime_se: Optional[Dict[str, float]] = None,
    margin_k: float = 1.0,
) -> bool:
    """Reject strategies SIGNIFICANTLY worse than random in any regime.

    Regime partitioning uses tercile boundaries from training ATM_IV.
    A strategy fails if, in any regime with >= REGIME_MIN_TRADES trades, its
    per-regime trade-level Sharpe is below the random-entry baseline's Sharpe
    by MORE than ``margin_k`` × the combined Lo(2002) sampling SE of the two
    estimates. ``margin_k=0`` reproduces the old hard "below at all" threshold,
    which decided feasibility WITHIN sampling noise (a ~0.3-SE iron_condor miss
    sentineled the whole population → fold-2/3 aborts). See memory
    project_per_regime_gate_noise_2026_06_04.

    If random-entry baselines are not provided, uses floor of -0.5 as
    a conservative fallback.

    Per Ang & Bekaert (2002): strategies must be evaluated per-regime
    to avoid survivorship bias from favorable regime dominance.

    Args:
        result: BacktestResult with trade PnLs.
        terminal_data: Dict with "ATM_IV" array for regime classification.
        regime_terciles: (q33, q67) boundaries from _compute_regime_terciles.
            If None, computes from terminal_data on the fly.
        random_entry_regime_sharpes: Dict mapping regime name to the
            random-entry baseline's trade-level Sharpe for that regime.
            If None, uses fixed floor of -0.5.
        random_entry_regime_se: Dict mapping regime name to the Lo(2002)
            sampling SE of that bar Sharpe. None ⇒ se_bar=0 (strat-only margin).
        margin_k: noise band in combined-SE units. Reject only if worse than
            the bar by > margin_k·sqrt(se_strat² + se_bar²). Default 1.0.
    """
    atm_iv_norm = terminal_data.get("ATM_IV")
    if atm_iv_norm is None:
        return False
    n_bars = len(atm_iv_norm)

    # Compute terciles if not provided
    if regime_terciles is None:
        regime_terciles = _compute_regime_terciles(terminal_data)
    if regime_terciles is None:
        return False
    q33, q67 = regime_terciles

    # Partition trades by regime at entry bar
    regime_pnls: Dict[str, list] = {"low": [], "mid": [], "high": []}
    for trade in result.trades:
        eb = trade.entry_bar
        if eb < 0 or eb >= n_bars:
            continue
        iv_norm = float(atm_iv_norm[eb])
        if iv_norm < q33:
            regime_pnls["low"].append(trade.pnl)
        elif iv_norm > q67:
            regime_pnls["high"].append(trade.pnl)
        else:
            regime_pnls["mid"].append(trade.pnl)

    # Check each regime with sufficient trades
    for regime_name, pnls in regime_pnls.items():
        if len(pnls) < REGIME_MIN_TRADES:
            continue
        arr = np.array(pnls, dtype=np.float64)
        std = float(np.std(arr))
        if std < 1e-12:
            if float(np.mean(arr)) < 0:
                return True
            continue
        regime_sharpe = float(np.mean(arr) / std)
        # Floor: random-entry per-regime Sharpe, or -0.5 fallback
        if random_entry_regime_sharpes is not None:
            floor = random_entry_regime_sharpes.get(regime_name, -0.5)
        else:
            floor = -0.5
        # Noise-aware gate (2026-06-04): both regime_sharpe and the bar are NOISY
        # estimates (Lo 2002 SE ≈ sqrt((1+0.5·SR²)/n)). Reject ONLY if the
        # strategy is worse than the bar by more than margin_k × the combined SE
        # of the difference — else the decision is inside sampling noise. (The
        # iron_condor / iron_butterfly fold-2/3 collapses were ~0.3–0.85-SE
        # misses; a genuine miss like RPB's −2.3-SE still fails.) margin_k=0
        # restores the old hard threshold.
        #
        # margin_k=1.0 is NOT tuned: it is the standard 1-SE "within sampling
        # error → indistinguishable" band, and it is ROBUST — the measured
        # per-template margins separate the noise-collapses (≤0.85 SE) from the
        # one genuine miss (RPB, 2.28 SE) by a wide gap, so any k∈[1,2] gives the
        # IDENTICAL verdict (scripts/diag_regime_gate_margins.py).
        #
        # combined_se uses the INDEPENDENT form sqrt(se_strat²+se_bar²). The two
        # Sharpes share the price path (positively correlated), so the true
        # difference-SE is smaller; the independent form OVER-estimates it →
        # widens the band → errs toward NOT rejecting. Conservative in the
        # intended (anti-spurious-rejection) direction.
        se_strat = float(np.sqrt((1.0 + 0.5 * regime_sharpe * regime_sharpe) / len(arr)))
        se_bar = 0.0
        if random_entry_regime_se is not None:
            se_bar = float(random_entry_regime_se.get(regime_name, 0.0))
        combined_se = float(np.sqrt(se_strat * se_strat + se_bar * se_bar))
        if regime_sharpe < floor - margin_k * combined_se:
            return True

    return False


# ---------------------------------------------------------------------------
# VectorizedFitnessEvaluator — drop-in replacement using vectorized_backtest
# ---------------------------------------------------------------------------

@dataclass
class VectorizedFitnessEvaluator:
    """Fitness evaluator using vectorized tree evaluation + sequential position loop.

    Same interface as FitnessEvaluator (evaluate, score_on_data, stats) but uses
    evaluator_vectorized.vectorized_backtest() instead of MultiLegOptionsBacktester.
    ~30-50x faster for large datasets (1-minute resolution).

    terminal_data is pre-computed once per split via prepare_terminal_data() and
    passed in to avoid re-normalizing 205K+ rows per evaluation.
    """
    template: Template
    data: pd.DataFrame
    terminal_data: Dict[str, np.ndarray]
    objectives: Tuple[str, ...] = DEFAULT_OBJECTIVES
    min_trades: int = DEFAULT_MIN_TRADES
    warmup_bars: int = 30  # Must equal max(_LAG_CHOICES)=30; bars < warmup have zero-fill Lag/Delta artifacts
    cost_multiplier: float = 1.0  # Level B: cost sensitivity sweep
    # Nursery ramp: random-entry gate margin starts at 0.0 and ramps to
    # 0.30 over the first 40% of generations. Set by gp_engine per gen.
    random_entry_margin: float = 0.30
    # Fold-recenter for EmbProj consistency across B/C/D conditions
    fold_recenter_stats: Optional[Dict] = None
    fold_id: Optional[str] = None
    # Noise injection scale. 0.0 = disabled (default for evolution).
    # Noise robustness is applied POST-GP via Monte Carlo re-scoring
    # of Pareto front strategies, not during evolution — per-generation
    # noise destroys NSGA-III convergence by changing the fitness
    # landscape every generation.
    noise_scale: float = 0.0
    # Per-regime noise-aware gate (Ang & Bekaert 2002). Tercile-based regime
    # partitioning from training ATM_IV. Strategies worse than the
    # random-entry baseline in any regime receive FAILED_FITNESS_SENTINEL.
    regime_gate_enabled: bool = True
    # Noise-aware per-regime gate (2026-06-04). The per-regime Sharpe and the
    # random-entry bar are NOISY estimates (Lo 2002 SE ≈ sqrt((1+0.5·SR²)/n)).
    # Reject only if the strategy is worse than the bar by MORE than
    # regime_gate_margin_k × the combined SE of the difference. margin_k=0
    # reproduces the old hard threshold, which decided feasibility WITHIN
    # sampling noise → spurious iron_condor/iron_butterfly fold-2/3 collapses
    # (z≈-0.3/-0.85 misses sentineled the whole population).
    regime_gate_margin_k: float = 1.0

    # Internal state
    _cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _eval_count: int = field(default=0, init=False, repr=False)
    _cache_hit_count: int = field(default=0, init=False, repr=False)
    _data_fingerprint: str = field(default="", init=False, repr=False)
    # Change A: Clean terminal_data copy for noise injection.
    # _clean_terminal_data stores the original (noise-free) terminal_data.
    # Per-generation, apply_generation_noise() creates a noisy copy and
    # stores it in self.terminal_data. Cache is cleared each generation
    # since the noisy data differs.
    _clean_terminal_data: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    # Evolution-level DSR trial recorder (P1-B). None unless enabled by
    # gp_engine.evolve via enable_trial_recording(). Maps canonical tree-quad
    # signature -> (raw training annualized Sharpe, n_days), first-seen.
    _evo_trial_records: "Optional[Dict[str, Tuple[float, int]]]" = field(
        default=None, init=False, repr=False)

    def enable_trial_recording(self) -> "Dict[str, Tuple[float, int]]":
        """Start recording distinct-individual training Sharpes for the DSR
        gate. Idempotent; returns the live sink dict (shared by reference).

        Noise consistency (review M1): the DSR gate later deflates each
        champion's CLEAN-data training Sharpe (``_run_counts`` uses
        ``prepare_terminal_data``). The recorder therefore only records when
        noise is INACTIVE (``_clean_terminal_data is None`` — see the recorder
        guard in ``_evaluate_uncached``). If during-evolution noise is active
        (``noise_scale > 0``) every evaluation runs on a NOISY terminal copy, so
        recording is SKIPPED to avoid a noisy-population-vs-clean-champion units
        mismatch; the gate then degrades to pure significance (SR*=0) rather
        than silently deflating against an inconsistent bar. During-evo noise is
        currently abandoned in favour of the post-GP MC robustness gate, so this
        never triggers in production — the guard exists so reviving noise can't
        silently corrupt the bar."""
        if self._evo_trial_records is None:
            self._evo_trial_records = {}
        if self.noise_scale > 0:
            import warnings
            warnings.warn(
                "DSR trial recording enabled with noise_scale > 0: per-generation "
                "noise makes recorded Sharpes inconsistent with the clean-data "
                "champion Sharpes the gate deflates, so recording is SKIPPED and "
                "the DSR gate degrades to pure significance (SR*=0). Wire "
                "clean-path recording before reviving during-evolution noise.",
                UserWarning, stacklevel=2,
            )
        return self._evo_trial_records

    @property
    def evo_trial_records(self) -> "Dict[str, Tuple[float, int]]":
        """Recorded {signature: (sharpe_ann, n_days)} (empty if not enabled)."""
        return self._evo_trial_records or {}

    def __post_init__(self):
        for obj in self.objectives:
            if obj not in OBJECTIVE_REGISTRY:
                raise ValueError(
                    f"unknown objective {obj!r}; available: "
                    f"{list(OBJECTIVE_REGISTRY.keys())}"
                )
        # G1 fix: content-based fingerprint (not id() which can recycle via GC).
        # S2 fix: filter to numeric columns only (object-dtype tobytes is non-deterministic).
        # M2 fix: hash every 10th row (not just 3) to prevent collision on
        # DataFrames that differ only in interior rows.
        import hashlib
        _h = hashlib.sha256()
        _h.update(f"{self.data.shape}:{list(self.data.dtypes)}".encode())
        if len(self.data) > 0:
            _numeric = self.data.select_dtypes(include=["number"])
            if len(_numeric.columns) > 0:
                _vals = _numeric.values
                _step = max(1, len(_vals) // 100)  # ~100 samples
                for _ri in range(0, len(_vals), _step):
                    _h.update(_vals[_ri].tobytes())
                # Always include last row
                _h.update(_vals[-1].tobytes())
        self._data_fingerprint = _h.hexdigest()[:16]

        # MEDIUM-6 (2026-06-01 audit): snapshot the POST-ramp min_trades target.
        # gp_engine ramps fitness_evaluator.min_trades from 1→target over the
        # first ~30 gens (reading the constructed value as the target at loop
        # entry). The DSR evolution-trial recorder must only record a trial's RAW
        # train Sharpe once min_trades has reached its FINAL value — otherwise an
        # individual first SEEN at the loose threshold (min_trades=1) records a
        # Sharpe that the final gate (min_trades=target) would SENTINEL, biasing
        # the evolution-level (N, V) the DSR gate deflates against. This snapshot
        # equals the target because __post_init__ runs at construction, BEFORE
        # gp_engine begins ramping min_trades down.
        self._target_min_trades = int(self.min_trades)

        # Change A: store clean terminal_data for per-generation noise injection.
        # The noisy copy replaces self.terminal_data each generation.
        if self.noise_scale > 0:
            self._clean_terminal_data = self.terminal_data

        # Random-entry null baseline: compute once per template.
        # If a strategy can't beat random entry by 0.2 Sharpe, it has
        # no genuine entry signal — it's just harvesting unconditional theta.
        self._random_entry_sharpe = self._compute_random_entry_baseline()

        # Regime gate: compute tercile boundaries from training ATM_IV
        # and per-regime random-entry Sharpe for the "worse than random
        # in any regime" rejection criterion.
        self._regime_terciles = _compute_regime_terciles(self.terminal_data)
        self._random_entry_regime_sharpes: Optional[Dict[str, float]] = None
        self._random_entry_regime_se: Optional[Dict[str, float]] = None
        if self._regime_terciles is not None and self.regime_gate_enabled:
            (self._random_entry_regime_sharpes,
             self._random_entry_regime_se) = self._compute_random_entry_per_regime()

    def _compute_random_entry_baseline(self) -> float:
        """Compute Sharpe of a random-entry strategy that holds to EOD.

        Random entry on ~25% of bars, NO exit signal (hold to max_bars or
        EOD close). This is the fairest null: it tests whether the GP's
        entry timing adds value beyond random timing + passive theta.

        Previous implementation used LT(MinutesToClose, -1.448) as exit,
        but this triggered the clock-exit hard gate (all signal exits at
        the same time → std < 3 → sentinel Sharpe -1e6). The gate
        comparison `strategy.sharpe <= -1e6 + margin` was always false,
        making the random-entry protection dead code.

        Now uses _override_exit_signals with all-zeros (never fire exit
        signal). Trades exit via max_hold (330 bars) or the EOD force-close
        (mtc <= EOD_FORCE_CLOSE_MTC == 15:50 ET); both are inherited from
        vectorized_backtest, so this null baseline tracks the R1/R2 gate.
        """
        from layer2.evaluator_vectorized import vectorized_backtest
        from layer2.grammar import TermNode, TermDef, GType
        # Random entry: fire on ~25% of bars (deterministic per data fingerprint)
        rng = np.random.RandomState(int(self._data_fingerprint[:8], 16) % 2**31)
        n_bars = len(self.data)
        random_entry_arr = (rng.random(n_bars) < 0.25).astype(np.float64)
        # No exit signal — hold to max_bars or EOD
        never_exit_arr = np.zeros(n_bars, dtype=np.float64)
        size_tree = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5)
        try:
            result = vectorized_backtest(
                entry_tree=None,  # overridden
                exit_tree=None,   # overridden
                size_tree=size_tree,
                data=self.data,
                template=self.template,
                terminal_data=self.terminal_data,
                warmup_bars=self.warmup_bars,
                cost_multiplier=self.cost_multiplier,
                _override_entry_signals=random_entry_arr,
                _override_exit_signals=never_exit_arr,
            )
            import sys
            print(f"  [random-entry baseline] template={self.template.name} "
                  f"sharpe={result.sharpe:.4f} trades={result.total_trades}",
                  file=sys.stderr)
            return result.sharpe
        except Exception as exc:
            import sys
            print(f"  [random-entry baseline] FAILED for {self.template.name}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 0.0

    def _compute_random_entry_per_regime(self) -> "Tuple[Dict[str, float], Dict[str, float]]":
        """Compute random-entry trade-level Sharpe per regime, AND its Lo(2002) SE.

        Uses the same random-entry baseline as _compute_random_entry_baseline
        but partitions trades by regime at entry bar using tercile boundaries.
        Returns (means, ses): regime -> trade-level Sharpe, and regime -> the
        Lo (2002) sampling SE of that Sharpe (sqrt((1+0.5·SR²)/n)). The SE feeds
        the noise-aware per-regime gate so the bar's own estimation error is
        carried into the feasible/infeasible decision.
        """
        from layer2.evaluator_vectorized import vectorized_backtest
        from layer2.grammar import TermNode, TermDef, GType
        n_bars = len(self.data)
        rng = np.random.RandomState(int(self._data_fingerprint[:8], 16) % 2**31)
        random_entry_arr = (rng.random(n_bars) < 0.25).astype(np.float64)
        never_exit_arr = np.zeros(n_bars, dtype=np.float64)
        size_tree = TermNode(defn=TermDef("EphReal", GType.REAL), value=0.5)
        try:
            result = vectorized_backtest(
                entry_tree=None, exit_tree=None, size_tree=size_tree,
                data=self.data, template=self.template,
                terminal_data=self.terminal_data,
                warmup_bars=self.warmup_bars,
                cost_multiplier=self.cost_multiplier,
                _override_entry_signals=random_entry_arr,
                _override_exit_signals=never_exit_arr,
            )
        except Exception:
            _f = {"low": -0.5, "mid": -0.5, "high": -0.5}
            return _f, {"low": 0.0, "mid": 0.0, "high": 0.0}

        atm_iv = self.terminal_data.get("ATM_IV")
        if atm_iv is None or self._regime_terciles is None:
            _f = {"low": -0.5, "mid": -0.5, "high": -0.5}
            return _f, {"low": 0.0, "mid": 0.0, "high": 0.0}
        q33, q67 = self._regime_terciles

        regime_pnls: Dict[str, list] = {"low": [], "mid": [], "high": []}
        for trade in result.trades:
            eb = trade.entry_bar
            if eb < 0 or eb >= len(atm_iv):
                continue
            iv = float(atm_iv[eb])
            if iv < q33:
                regime_pnls["low"].append(trade.pnl)
            elif iv > q67:
                regime_pnls["high"].append(trade.pnl)
            else:
                regime_pnls["mid"].append(trade.pnl)

        sharpes: Dict[str, float] = {}
        ses: Dict[str, float] = {}
        import sys
        # Bar uses a 5-trade floor; the STRATEGY gate trips only at
        # REGIME_MIN_TRADES=10. For a sparse bar (5<=n<10) the Lo SE below is
        # LARGE, which only WIDENS the gate band → conservative (errs toward not
        # rejecting). Below 5 trades the bar is too thin to estimate → fall back
        # to the -0.5 floor with se=0 (no SE cushion on an absent bar).
        for name, pnls in regime_pnls.items():
            if len(pnls) < 5:
                sharpes[name] = -0.5  # fallback
                ses[name] = 0.0
            else:
                arr = np.array(pnls, dtype=np.float64)
                std = float(np.std(arr))
                sr = float(np.mean(arr) / max(std, 1e-12))
                sharpes[name] = sr
                # Lo (2002) sampling SE of the per-regime trade-level Sharpe.
                ses[name] = float(np.sqrt((1.0 + 0.5 * sr * sr) / len(pnls)))
        print(f"  [regime baseline] {self.template.name}: "
              f"low={sharpes.get('low',0):.3f}±{ses.get('low',0):.3f} "
              f"mid={sharpes.get('mid',0):.3f}±{ses.get('mid',0):.3f} "
              f"high={sharpes.get('high',0):.3f}±{ses.get('high',0):.3f}", file=sys.stderr)
        return sharpes, ses

    def apply_generation_noise(self, generation: int, seed: int) -> None:
        """Apply per-generation calibrated noise to terminal data.

        Change A: Each generation sees the same noise realization (all
        individuals in one generation evaluated on identical noisy data),
        but different generations see different noise. This is cheaper than
        per-individual noise (N copies) while still achieving robustness --
        different generations explore different perturbations.

        The cache is cleared because terminal data changed. Clean terminal
        data is preserved in _clean_terminal_data for the next generation.

        Args:
            generation: current generation number (for logging).
            seed: base seed combined with generation for reproducibility.
        """
        if self.noise_scale <= 0 or self._clean_terminal_data is None:
            return
        from layer2.evaluator_vectorized import _add_terminal_noise
        rng = np.random.RandomState(seed + generation)
        self.terminal_data = _add_terminal_noise(
            self._clean_terminal_data, rng, noise_scale=self.noise_scale
        )
        # Cache invalidation: noisy data differs per generation so all
        # cached fitness values from prior generations are stale.
        self._cache.clear()

    def evaluate(self, entry_tree: Node, exit_tree: Node,
                 size_tree: Node, delta_tree: "Optional[Node]" = None,
                 stop_mult: "Optional[float]" = None) -> np.ndarray:
        """Vectorized backtest + fitness vector. Cached by tree hash.

        stop_mult: per-individual evolved stop-loss base credit-multiple
            (gp_engine.Individual.stop_mult). Flows into vectorized_backtest as
            `stop_loss_credit_multiple`; 0.0 ⇒ no stop / hold to expiry. None ⇒
            use the evaluator/backtester default (legacy path). The cache key
            INCLUDES stop_mult so two individuals differing only in their stop
            do not collide.
        """
        cache_key = self._tree_hash(entry_tree, exit_tree, size_tree, delta_tree,
                                    stop_mult=stop_mult)
        if cache_key in self._cache:
            self._cache_hit_count += 1
            return self._cache[cache_key]
        self._eval_count += 1
        fitness_vec = self._evaluate_on_data(
            entry_tree, exit_tree, size_tree,
            self.data, self.terminal_data,
            delta_tree=delta_tree,
            stop_mult=stop_mult,
        )
        self._cache[cache_key] = fitness_vec
        return fitness_vec

    def score_on_data(self, entry_tree: Node, exit_tree: Node,
                      size_tree: Node, data: pd.DataFrame,
                      terminal_data: Optional[Dict[str, np.ndarray]] = None,
                      fold_recenter_stats=None,
                      fold_id: Optional[str] = None,
                      delta_tree: "Optional[Node]" = None,
                      stop_mult: "Optional[float]" = None,
                      ) -> np.ndarray:
        """Re-score on a different split (val/test). No cache.

        If terminal_data is provided, it is used directly (caller is
        responsible for L2.60 forbidden-terminal stripping). Otherwise
        prepare_terminal_data(data) is called with no filtering.

        fold_recenter_stats / fold_id: override the training-fold values
        for val/test scoring. If not provided, falls back to self.* (training).
        For Condition A (no embeddings) this makes no difference; for B/C/D
        the caller MUST provide the correct fold's recenter stats.
        """
        if terminal_data is None:
            from layer2.evaluator_vectorized import prepare_terminal_data
            terminal_data = prepare_terminal_data(data)
        return self._evaluate_on_data(
            entry_tree, exit_tree, size_tree, data, terminal_data,
            fold_recenter_stats=fold_recenter_stats,
            fold_id=fold_id,
            delta_tree=delta_tree,
            stop_mult=stop_mult,
            # MH5 (2026-06-01 holistic review): val/test scoring is intentionally
            # gate-free, and this is SOUND (not a train-serve hole), for two reasons:
            # (1) the random-entry null baseline is calibrated on TRAIN data — applying
            # it to a val/test split would test against a mismatched baseline; and
            # (2) score_on_data only ever re-scores members of the NSGA Pareto FRONT,
            # which was selected under the GATED training objective — so any
            # noise-entry strategy is already sentinel'd OUT of the front before it
            # reaches val scoring. The gate-free val therefore measures OOS
            # performance of train-gate SURVIVORS; it cannot admit a strategy the
            # training objective rejected. Early-stop/best-val selection inherits
            # this property, and the shipped front is the train-gated population.
            _skip_random_gate=True,
            # MH5 (2026-06-01 audit): ALSO skip the per-regime hard gate on val/test.
            # The regime gate fires against TRAIN-fit tercile boundaries
            # (_regime_terciles) AND a TRAIN-fit per-regime random-entry baseline
            # (_random_entry_regime_sharpes), both computed ONCE from train in
            # __post_init__. On a regime-SHIFTED val window those train terciles/
            # baselines are mismatched, so the gate can SENTINEL the entire val front
            # — which then collapses the fleet via the val_sharpe gate-reconciliation
            # in _run_one_template (a sentinel val_fitness sets val_sharpe to
            # -FAILED_FITNESS_SENTINEL). Val/test scoring is therefore intentionally
            # regime-gate-free, identical to the random-gate treatment and SOUND for
            # the same reason: score_on_data only re-scores members already selected
            # under the GATED training objective.
            _skip_regime_gate=True,
        )

    def _evaluate_on_data(self, entry_tree: Node, exit_tree: Node,
                          size_tree: Node, data: pd.DataFrame,
                          terminal_data: Dict[str, np.ndarray],
                          fold_recenter_stats=None,
                          fold_id: Optional[str] = None,
                          delta_tree: "Optional[Node]" = None,
                          stop_mult: "Optional[float]" = None,
                          _skip_random_gate: bool = False,
                          _skip_regime_gate: bool = False,
                          ) -> np.ndarray:
        """Core evaluation: vectorized backtest → fitness vector."""
        from layer2.evaluator_vectorized import vectorized_backtest
        _frs = fold_recenter_stats if fold_recenter_stats is not None else self.fold_recenter_stats
        _fid = fold_id if fold_id is not None else self.fold_id
        # Thread the evolved stop gene through ONLY when supplied — when None we
        # let vectorized_backtest use its own default so legacy (gene-free)
        # callers are byte-for-byte unchanged.
        _stop_kw = {} if stop_mult is None else {
            "stop_loss_credit_multiple": float(stop_mult)}
        try:
            result = vectorized_backtest(
                entry_tree, exit_tree, size_tree, data, self.template,
                delta_tree=delta_tree,
                cost_multiplier=self.cost_multiplier,
                terminal_data=terminal_data, warmup_bars=self.warmup_bars,
                fold_recenter_stats=_frs,
                fold_id=_fid,
                **_stop_kw,
            )
            # P1-B: record the RAW training Sharpe for the evolution-level DSR
            # trial population (pre-gate). Only on the TRAINING frame — the
            # val/test path (score_on_data) passes a different `data` object,
            # so `data is self.data` cleanly excludes it. _record_trial skips
            # the -1e6 phantom Sharpe internally. M1: `_clean_terminal_data is
            # None` means noise is INACTIVE — under active noise every eval is on
            # a noisy terminal copy, inconsistent with the clean-data champion
            # Sharpe the gate deflates, so we skip (see enable_trial_recording).
            #
            # BLOCKER B2 (2026-06-02 audit) — record EVERY distinct finite-Sharpe
            # trial from gen 0, regardless of the gp_engine min_trades ramp. The
            # prior `min_trades >= _target_min_trades` (MEDIUM-6) guard meant
            # recording did not start until the ramp completed (~gen 30); combined
            # with early-stopped runs this undercounted the DSR trial count N by
            # ~30x (observed dsr_n_trials ~248 < pop_size, vs the true thousands),
            # making SR* far too lenient. A backtest that RAN is a multiple-testing
            # trial whether or not it later cleared min_trades — so N must count it.
            # The VARIANCE estimate V remains protected against invalid short
            # samples by `evolution_n_v_from_records`'s `n_days >= min_days` filter
            # (pbo.py), which is the correct place for the V-validity filter; the
            # min_trades-ramp threshold is NOT a Sharpe-validity criterion and was
            # never the right gate for either N or V.
            if (self._evo_trial_records is not None and data is self.data
                    and self._clean_terminal_data is None):
                _record_trial(
                    self._evo_trial_records,
                    _canonical_signature(entry_tree, exit_tree, size_tree,
                                         delta_tree),
                    result.sharpe, result.n_days,
                )
            # Phase-1 sentinel check: vectorized_backtest sets sharpe=-1e6
            # for churning/max-hold/clock-exit. Catch before Phase-2 gates.
            if result.sharpe <= -1e5:
                # evaluator hard-degeneracy sentinel (churning / clock exit)
                fitness_vec = _graded_infeasible(len(self.objectives), 1.0)
            elif result.total_trades < self.min_trades:
                # too few trades: grade by the shortfall so the GP climbs toward the
                # min-trades floor instead of sitting on a flat 1e6 sentinel (#4).
                fitness_vec = _graded_infeasible(
                    len(self.objectives),
                    1.0 - result.total_trades / max(self.min_trades, 1))
            # exit_util < 10% gate REMOVED (dead code): subsumed by the
            # exit_util < 20% gate at the end of the cascade. Both require
            # trades >= 10. The 10% threshold never independently catches
            # anything the 20% threshold wouldn't also catch.
            elif result.entry_fire_rate_flat > 0.35:
                # Tautology gate (#6 fix 2026-06-02: uses the FLAT-bar fire rate, not the
                # raw bar-level rate). A DAY-SELECTIVE hold strategy fires on most BARS of
                # the days it trades but is flat-and-idle on the days it skips, so its raw
                # bar-level rate is high yet its flat-bar rate (entries among bars where it
                # is actually able to enter) is low. The old raw-rate gate hard-sentineled
                # those winners — the dominant profitable 0DTE pattern (positive control
                # proved a Sharpe ~+12.8 planted winner was killed at raw fire_rate 0.42).
                # entry_fire_rate_flat > 0.35 = "enters on most opportunities when flat" =
                # effectively unconditional (beta, not signal). True always-enter churners
                # still score ~1.0 here; a day-selective hold scores low and now passes.
                # Grade by how far above 0.35 -> gradient toward selectivity (#4).
                fitness_vec = _graded_infeasible(
                    len(self.objectives), (result.entry_fire_rate_flat - 0.35) / 0.65)
            elif result.entry_fire_rate < 0.02:
                # Near-never gate: entry fires on <2% of bars. Too rare to
                # produce meaningful trade statistics. Grade by the shortfall
                # below 0.02 -> gradient toward firing more often (#4).
                fitness_vec = _graded_infeasible(
                    len(self.objectives), (0.02 - result.entry_fire_rate) / 0.02)
            elif (not _skip_random_gate
                  and self._random_entry_sharpe is not None
                  and self.random_entry_margin > 0
                  and result.sharpe <= self._random_entry_sharpe + self.random_entry_margin + 0.50 * result.entry_fire_rate_flat):
                # Random-entry null gate: strategy doesn't beat random entry
                # by a fire-rate-scaled margin. Skipped for val/test scoring.
                # Margin ramps from 0.0→0.30 over first 40% of generations
                # (nursery period) to give GP gradient before full stringency.
                # LOW L2 (2026-06-02 audit): the fire-rate surcharge uses the FLAT-bar
                # fire rate, not the raw bar-level rate, to AGREE with the #6 tautology
                # gate. A day-selective hold strategy fires on most BARS of the days it
                # trades (high raw entry_fire_rate) but is flat on the days it skips, so
                # its raw rate is NOT a selectivity signal — surcharging the null bar by
                # the raw rate penalized exactly the winners #6 was fixed to admit.
                # entry_fire_rate_flat (entries among bars where it can actually enter)
                # is the consistent selectivity measure across both gates.
                # Grade by the Sharpe shortfall below the null threshold (1.0-Sharpe
                # scale) -> gradient toward beating random entry (#4).
                _rand_thr = (self._random_entry_sharpe + self.random_entry_margin
                             + 0.50 * result.entry_fire_rate_flat)
                fitness_vec = _graded_infeasible(
                    len(self.objectives), (_rand_thr - result.sharpe) / 1.0)
            # Selectivity gate (conditional_sharpe_gap) REMOVED (2026-05-19):
            # CSG < 0.0 combined with fire_rate <= 0.35 created an impossible
            # constraint — ALL strategies with fr<=0.35 had negative CSG.
            # GP got sentinel fitness for every individual. The random-entry
            # gate (nursery-ramped) already serves the same purpose. CSG is
            # still COMPUTED and available for post-hoc analysis but no longer
            # a hard gate.
            elif (result.exit_utilization < 0.20 and result.total_trades >= 10
                  and result.sharpe <= 0.0):
                # Exit-utilization gate (#7 fix 2026-06-02: only gates UNPROFITABLE
                # vestigial-exit strategies). <20% signal-driven exits = vestigial exit
                # tree — BUT for 0DTE credit, holding to close (session/settlement exit)
                # is a LEGITIMATE, often-optimal exit, so a PROFITABLE hold-to-close (the
                # dominant 0DTE pattern; the positive control proved exit_util=0 winners
                # with Sharpe ~+12.8 were killed here) must NOT be sentineled. Only an
                # UNprofitable (sharpe<=0) low-exit-util strategy is a true vestigial-exit
                # degeneracy. (Overfit in-sample winners are caught downstream by DSR +
                # cross-fold persistence + OOS val.)
                #
                # GATE-ATTRIBUTION FIX (2026-06-04): grade by SHARPE, not exit_util. The
                # old (0.20-exit_util)/0.20 violation was FLAT (==1.0) for EVERY hold-to-
                # close strategy (exit_util=0), so NSGA-III had no gradient and the
                # population collapsed (0-2/128 feasible) — the dominant 0DTE-credit
                # structure (hold-to-EOD-settlement) was sentineled on a flat landscape
                # while churny SIGNAL-exit LOSERS (exit_util>=0.20, Sharpe ~-24) passed.
                # For a hold-to-close strategy the distance-to-feasible is its SHARPE (it
                # must reach >0 to clear this gate), so grade by -sharpe/3.0: a -0.38
                # strategy now DOMINATES a -2.0, giving the GP a gradient toward
                # profitability among hold-to-close strategies. /3.0 spans the recoverable
                # band (random-entry baseline ~-0.7; below ~-3 is hopeless → max violation).
                # Clipped to [0,1] per the graded-infeasible contract (a feasible
                # individual, objectives <= +5, still dominates ALL infeasibles).
                fitness_vec = _graded_infeasible(
                    len(self.objectives), min(1.0, max(0.0, -result.sharpe / 3.0)))
            elif (not _skip_regime_gate
                  and self.regime_gate_enabled
                  and result.total_trades >= 10
                  and _regime_gate_fails(
                      result, terminal_data,
                      regime_terciles=self._regime_terciles,
                      random_entry_regime_sharpes=self._random_entry_regime_sharpes,
                      random_entry_regime_se=self._random_entry_regime_se,
                      margin_k=self.regime_gate_margin_k)):
                # Per-regime noise-aware gate (Ang & Bekaert 2002): reject strategies
                # worse than random entry in any regime (tercile-based).
                # MH5: skipped on val/test (_skip_regime_gate) — the train-fit
                # terciles/baselines are invalid on a regime-shifted OOS window.
                # Hard-degenerate per-regime failure: maximal violation (#4).
                fitness_vec = _graded_infeasible(len(self.objectives), 1.0)
            else:
                from layer2.grammar import node_count
                tn = node_count(entry_tree) + node_count(exit_tree) + node_count(size_tree)
                if delta_tree is not None:
                    tn += node_count(delta_tree)
                ite = _count_ite(entry_tree) + _count_ite(exit_tree)
                fitness_vec = self._fitness_from_result(result, total_nodes=tn, ite_count=ite)
        except Exception as exc:
            import sys
            _exc_cls = type(exc).__name__
            if not hasattr(self, '_logged_exc_classes'):
                self._logged_exc_classes = set()
            if _exc_cls not in self._logged_exc_classes:
                self._logged_exc_classes.add(_exc_cls)
                print(f"[VectorizedFitnessEvaluator] Backtester exception ({_exc_cls}): {exc}",
                      file=sys.stderr)
            fitness_vec = _graded_infeasible(len(self.objectives), 1.0)
        fitness_vec = np.where(
            np.isfinite(fitness_vec), fitness_vec, FAILED_FITNESS_SENTINEL
        )
        return fitness_vec

    def _fitness_from_result(self, result: BacktestResult,
                            total_nodes: int = 0,
                            ite_count: int = 0) -> np.ndarray:
        """Project BacktestResult onto the objective vector."""
        tc_score = trade_count_score(result.total_trades, result.n_days)

        parsimony_penalty = max(0, total_nodes - 10) ** 2 * 0.005
        parsimony_penalty += ite_count * 0.05
        # Normalize drawdown by avg position size so GP cannot exploit
        # small sizing to minimize drawdown. dd/size makes the objective
        # independent of sizing decisions.
        _adj_dd = result.max_drawdown_uncapped / max(result.avg_position_size, 0.05)
        # UNCAPPED (2026-06-01): the legacy result.max_drawdown was censored at 1.0,
        # making a -1x and a -8x drawdown identical — uncapped/avg_size exposes true
        # depth while keeping the sizing-exploit fix (dd per unit size).
        # SOFT drawdown penalty: push high-drawdown strategies DOWN the return axis —
        # no new objective, no Pareto attractor (cf. the removed exit_util objective).
        # Mild within-template rank pressure; absolute ruin is gated at the QC stage.
        # HIGH-4 fix (2026-06-01 holistic review, REVISED): apply the penalty to BOTH
        # return axes (neg_sharpe AND neg_sortino). RATIONALE: penalizing neg_sharpe
        # ONLY made the penalty Pareto-INCONSISTENT — a high-drawdown strategy could
        # dodge the demotion entirely on the UNPENALIZED neg_sortino axis and re-enter
        # the non-dominated set (the penalty created NEW non-dominated points rather
        # than uniformly demoting). Sharpe and Sortino are ~0.99 correlated, so adding
        # the SAME penalty to both costs ~2x effective weight along essentially one
        # axis, but it is DOMINANCE-PRESERVING: a penalized point is dominated on both
        # return axes by an otherwise-equal low-drawdown sibling, which is the intended
        # behaviour. Dominance consistency > exact penalty magnitude here. (The G9
        # champion sort applies the same formula once on val — selection, not search.)
        _dd_penalty = DRAWDOWN_PENALTY_WEIGHT * max(_adj_dd - DRAWDOWN_FREE_LEVEL, 0.0)
        derived = {
            "neg_sharpe": -result.sharpe + parsimony_penalty + _dd_penalty,
            "neg_sortino": -result.sortino + parsimony_penalty + _dd_penalty,
            "max_drawdown": _adj_dd,
            "neg_trade_count_score": -tc_score,
            "neg_win_rate": -result.win_rate,
            "neg_exit_utilization": -result.exit_utilization,
            "neg_conditional_sharpe_gap": -result.conditional_sharpe_gap,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "win_rate": result.win_rate,
            "psr": result.psr,
        }
        return np.array(
            [derived[obj] for obj in self.objectives], dtype=np.float64
        )

    def _tree_hash(self, entry: Node, exit_: Node, size: Node,
                   delta: "Optional[Node]" = None,
                   stop_mult: "Optional[float]" = None) -> str:
        """Cache key for tree triple/quad + data + evaluation params.

        Includes cost_multiplier and min_trades so cached fitness from one
        proxy configuration doesn't leak into a different configuration
        (e.g., re-evaluating old strategies with calibrated proxy). Also
        includes the evolved stop_mult gene: two individuals with identical
        trees but different stop levels are different strategies and must NOT
        share a cache slot (rounded to 3 dp to match Individual.signature()).
        """
        base = (
            f"D:{self._data_fingerprint}|"
            f"CM:{self.cost_multiplier}|MT:{self.min_trades}|"
            f"E:{to_str(entry)}|X:{to_str(exit_)}|S:{to_str(size)}"
        )
        if delta is not None:
            base += f"|DT:{to_str(delta)}"
        if stop_mult is not None:
            base += f"|SM:{float(stop_mult):.3f}"
        return base

    @property
    def cache_hit_rate(self) -> float:
        total = self._eval_count + self._cache_hit_count
        return self._cache_hit_count / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "evaluations": self._eval_count,
            "cache_hits": self._cache_hit_count,
            "cache_size": len(self._cache),
            "cache_hit_rate": self.cache_hit_rate,
        }
