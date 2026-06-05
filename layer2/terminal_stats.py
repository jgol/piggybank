"""Normalization statistics for REAL terminals — SINGLE SOURCE OF TRUTH.

Computed from L1 Parquet training split (dates <= 2024-09-30) on the CNN
checkpoint at raw_data/local_store/l1_cnn.parquet.

All bypass scalar terminals are normalized to approximately N(0,1) using:
    normalized = (raw - center) / scale

For most terminals: center=mean, scale=std (standard z-score).
For heavy-tailed terminals (VIXSpot, VIXTermSlope, ATM_IV, RawSpread):
    center=median, scale=IQR/1.35 (robust z-score per Model QA audit).

EphReal [-1,1] after normalization covers +/- 1 scale-unit of every
terminal, making GT(terminal, EphReal(x)) meaningful for all terminals.

These constants are FROZEN. To regenerate, run:
    python -m layer2.terminal_stats
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Frozen normalization statistics
# ---------------------------------------------------------------------------
# Computed from training split (date <= 2024-09-30) of l1_cnn.parquet.
# Format: {terminal_name: (center, scale, method)}
# method is "robust" (median, IQR/1.35) or "standard" (mean, std).

TERMINAL_NORM_STATS: Dict[str, Tuple[float, float, str]] = {
    # Heavy-tailed terminals — robust normalization (median, IQR/1.35)
    # 2026-06-02 (M5 clamp-saturation fix). The raw ATM_IV is now WINSORIZED at
    # 0.80 in generate_minute_parquet.py (the BS-solver 5.0 cap tail is not real
    # ATM vol). The prior robust scale (0.0486) mapped raw IV ~0.50 to z=+8.2, so
    # the evaluator's [-5,+5] OOD clamp saturated 2.5% of all bars (upper-tail
    # discrimination loss — the GP could not tell genuinely-high IV from capped).
    # Re-derived on the winsorized train split (date <= 2024-09-27): center =
    # median(winsorized IV > 0.001), scale chosen so the 0.80 winsor cap maps to
    # +4.5 — i.e. INSIDE the ±5 clamp with headroom. Result: 0.0% of bars clamp
    # (target <0.5%); raw IV=0.50→z=2.54, IV=0.15→z=0.26, IV=0.08→z=-0.20 (normal-
    # range discrimination fully preserved). The evaluator applies a purely affine
    # (raw-center)/scale and ignores the method field, so center/scale are the only
    # levers. See tests/test_data_torpedo_fixes.py::test_atm_iv_clamp_saturation_*.
    "ATM_IV":          (0.110291, 0.153269, "robust"),  # winsor@0.80; cap→+4.5; clamp 0.0%
    "VIXSpot":         (16.1562, 4.5718, "robust"),    # median=16.16, IQR/1.35=4.57
    "VIXTermSlope":    (-0.6875, 1.3079, "robust"),    # median=-0.69, IQR/1.35=1.31
    "RawSpread":       (0.01651, 0.008138, "robust"),  # 2026-06-01: spike-cleaned parquet (train<=2024-09-27)

    # Standard terminals — z-score (mean, std)
    "RealizedVol30m":  (0.000259, 0.000148, "standard"),  # 2026-06-01: double-sqrt removed + partial-window warmup
    "MinutesToClose":  (202.0, 116.913, "standard"),    # 2026-06-02 H3: bar-0 MTC=0 artifact repaired (→404); train mean=202.0, std=116.9
    "DeltaSpread1":    (-0.000046, 0.008379, "standard"),  # 2026-06-01: recomputed from spike-cleaned RawSpread
    "DeltaSpread5":    (-0.000202, 0.009458, "standard"),  # 2026-06-01: recomputed from spike-cleaned RawSpread
    "BarOfDay":        (202.0, 116.91, "standard"),      # 0-404 range for 1-min bars (405/day)

    # Probe scalars (Channel 2) — standard normalization
    "PredRV15":        (0.0147, 0.0033, "standard"),    # mean=0.0147, std=0.0033
    "PredRV30":        (0.0151, 0.0034, "standard"),    # mean=0.0151, std=0.0034
    "PredSpread":      (0.0005, 0.0115, "standard"),    # mean=~0, std=0.012
    # PredRegime replaced with binary regime indicators (values 0 or 1).
    # Normalization: center=0.5, scale=0.5 maps {0,1} to {-1, +1} in normalized space.
    # GP can use GT(RegimeIsHigh, EphReal(0.0)) as a clean boolean check.
    "RegimeAboveLow":  (0.5, 0.5, "standard"),     # 1 if regime >= 1 (not LOW_VOL)
    "RegimeIsHigh":    (0.5, 0.5, "standard"),     # 1 if regime >= 2 (HIGH_VOL_*)
    "RegimeIsPremium": (0.5, 0.5, "standard"),     # 1 if regime == 3 (HIGH_VOL_PREMIUM)

    # new probe scalars — computed from l1_cnn_v2.parquet (2026-05-16)
    "PredGammaAccel":      (0.0, 0.0002, "standard"),   # gamma/theta ratio change — very small scale
    "PredSmileConvexity":  (0.0094, 0.0297, "standard"),# IV butterfly spread change
    "PredJump":            (0.0247, 0.1272, "standard"), # P(|ret_30|>1%) ~ 2.5% base rate
    "PredFlowToxicity":    (9.8716, 1.0408, "standard"), # absolute flow magnitude

    # Synthesized terminals — computed from minute Parquet training split.
    # SessionReturn: cumulative intraday SPX return.
    "SessionReturn":   (0.0, 0.006, "standard"),     # empirical std=0.00595

    # Inter-day memory terminals (constant within day, updated at session open).
    # Use robust normalization (median, IQR/1.35) for VIX and RV (heavy-tailed).
    "SPXReturn3d":     (0.0021, 0.0118, "robust"),   # 3-day cumulative SPX return
    "VIXMean5d":       (16.4203, 4.5243, "robust"),  # 5-day rolling mean VIX
    "RV5d":            (0.1108, 0.0533, "robust"),    # 5-day realized vol (annualized)
    "IVRVGap5d":       (0.0197, 0.0328, "robust"),   # 5-day (mean IV - RV5d); positive = IV rich

    # Put-call skew: IV(25Δput) - IV(25Δcall). Robust normalization.
    # Computed from bars where both IV_25Dp and IV_25Dc are valid (>0.001).
    # Bars with missing grid IVs set PutCallSkew=0.0 (not included in stats).
    "PutCallSkew":     (0.0107, 0.0080, "robust"),

    # Session position: (price - session_low) / (session_high - session_low).
    # Range [0, 1] where 0 = at session low, 1 = at session high.
    "SessionPosition": (0.545, 0.334, "standard"),

    # 5-bar smoothed terminals: computed from forward-filled RAW values then normalized.
    # Robust stats (median, IQR/1.35) computed from training split, excluding zero-
    # padded bars. Smoothing barely reduces scale for 1-min data (high autocorrelation).
    "ATM_IV_5m":       (0.110840, 0.153147, "robust"),  # 2026-06-02 M5: winsor@0.80 base, cap→+4.5, clamp 0.0%
    "RealizedVol30m_5m": (0.000258, 0.000147, "standard"), # 2026-06-01: tracks single-Parkinson base
    "RawSpread_5m":    (0.016592, 0.005177, "robust"),  # 2026-06-01: spike-cleaned base

    # VIXChange: day-over-day VIX change. VIXSpot is daily-constant, so
    # Delta(VIXSpot, k) always returns 0 within a day. VIXChange gives GP
    # the inter-day VIX dynamics directly (overnight vol spike/collapse).
    "VIXChange":       (0.0, 1.5, "standard"),         # mean~0, typical overnight change ±1-2 pts

    # OvernightGap: log(today_open / prev_close). Computed from minute Parquet.
    "OvernightGap":    (0.0, 0.0057, "standard"),      # empirical std=0.00567

    # GridReliability: fraction of 11 IV grid points with valid data [0, 1].
    # Computed from minute Parquet training split.
    "GridReliability":  (0.935, 0.176, "standard"),     # empirical mean=0.935, std=0.176

    # ThetaUrgency: 1/sqrt(max(MinutesToClose, 1.0)). Captures nonlinear
    # theta decay for 0DTE options where theta scales as 1/sqrt(T).
    # The GP cannot construct this from existing terminals due to analytic
    # quotient (not true division), normalization destroying raw MTC scale,
    # and 4-node cost (27% of budget). Sufficiency restoration (Koza 1992).
    #
    # 2026-06-01 RE-DERIVATION (clamp-saturation fix). Raw ThetaUrgency is a
    # ONE-SIDED heavy upper tail (median=0.0705, max=1.0 at MTC=1, min=0.0498
    # at MTC=403). The prior robust stat (median-center, IQR/1.35=0.0314 scale)
    # mapped MTC=1 to z=+29.6, so the evaluator's OOD clamp [-5,+5] saturated
    # 5.19% of all bars and 100% of bars with MTC<10 — destroying ALL
    # end-of-day discrimination (every close-region bar pinned to +5). The
    # evaluator applies a purely affine (raw-center)/scale (it ignores the
    # method field), so the only lever is center/scale. Re-derived as
    # center=mean(raw), scale chosen so the raw max (1.0) maps to +4.5 — i.e.
    # INSIDE the ±5 clamp with headroom.
    #
    # 2026-06-02 (H3 bar-0 MTC repair). The bar-0 MinutesToClose=0 artifact made
    # ThetaUrgency=1/sqrt(max(0,1))=1.0 (the +4.5 clamp-region MAX) at every
    # OPEN; Lag/Delta at bar 1 read that garbage and the GP hunted spurious
    # "trade-at-open" rules. With bar-0 MTC repaired to the true open value (~404)
    # the open's ThetaUrgency is now 0.0498 (the MINIMUM, z≈-0.24), and the raw
    # mean drops 0.100529→0.098183 (882 bar-0 values left the 1.0 spike). Stat
    # re-derived on the H3-repaired train split (date <= 2024-09-27, 204,930
    # bars): center=mean(raw)=0.098183, scale=(1.0-mean)/4.5=0.200404 so the raw
    # max (1.0, still at MTC=1 last bars — unaffected by H3) maps to +4.5. Result:
    # 0.0000% of bars clamp (target <0.5%); the final-10-minute gradient is fully
    # preserved and monotone (MTC 1→+4.50, 2→+3.03, 3→+2.40, 5→+1.74, 8→+1.27,
    # 10→+1.09 — each a distinct, EphReal-reachable value). See
    # tests/test_terminal_stats.py::test_theta_urgency_clamp_rate_below_one_percent
    # and tests/test_data_torpedo_fixes.py::test_theta_urgency_at_open_not_max.
    "ThetaUrgency":     (0.098183, 0.200404, "standard"),  # H3-repaired; mean-center, raw max(1.0)→+4.5; clamp 0.0%
}

# Terminals that do NOT get normalized (non-REAL, structural, or special)
# - SPXClose: NOT a grammar terminal (confirmed by spec)
# - AbstainFlag: removed from grammar (spec item 4)
# - EphReal/EphInt: ephemeral constants, not data terminals
# - Regime/Side literals: categorical, not REAL
# - Embedding terminals (EMB_*): vector-valued, not scalar
# - UnrealizedProfitPct: position-aware exit terminal (Phase-2,
# docs/viable_run_plan_2026_06_03.md). INTENTIONALLY RAW (absent from
# TERMINAL_NORM_STATS): it is already a natural bounded FRACTION of max
# profit (≈0 at entry, +1 at max profit, negative when losing), so a z-score
# would destroy the literal threshold semantics. Keeping it raw makes
# GT(UnrealizedProfitPct, EphReal(0.5)) mean exactly "captured ≥ 50% of max
# profit" — and EphReal's sampling range (-3..3, grammar.py) richly covers
# the [0,1] profit band and the [-1,0] loss-cut band. It is computed LIVE
# per-bar by the evaluator (position-dependent), not read from the parquet,
# so there is nothing to fit a (center, scale) from. Codegen mirrors this:
# it sets s["UnrealizedProfitPct"] directly (bypassing _normalize), keeping
# proxy↔QC parity on the raw fraction.

# Set of terminal names that have normalization stats
NORMALIZED_TERMINALS = frozenset(TERMINAL_NORM_STATS.keys())


# ---------------------------------------------------------------------------
# Robust (median / IQR) scaling — ITEM #4, docs/viable_run_plan_2026_06_03.md
# ---------------------------------------------------------------------------
# Phase-4 normalization is robust scaling: center=median, scale=IQR/1.349, fit
# TRAIN-only per fold (no look-ahead). Peer-grounded: RevIN (Kim et al., ICLR
# 2022) — local-window normalization for non-stationary series; Huber (1981) —
# robust statistics for the heavy tails financial vol carries (Cont 2001).
#
# Why robust over the per-fold MEAN/STD that the pipeline used before: over the
# fold-1 expanding train window (18.9% of which is the 2022 bear) the heavy
# upper tail inflates STD by exactly 2x (measured robust/meanstd scale ratio =
# 0.506 on l1_minute_scalars.parquet fold-1), HALVING within-regime IV
# discrimination — the threshold entries need that discrimination. Robust scaling
# dampens the tail directly: the daily ATM_IV inter-quartile gap recovers from
# 0.611 sigma (mean/std) to 1.207 sigma (robust), a 1.97x recovery.
#
# The 1.349 divisor is the normal-consistent IQR->sigma factor (the IQR of a
# standard normal is 2*Phi^-1(0.75) = 1.34898), so on Gaussian data median/IQR
# and mean/std coincide; on heavy-tailed data the robust scale is smaller. (The
# FROZEN constants and the legacy per-method `robust` branch keep the historical
# 1.35 rounding for byte-compat; only the FORCE-ROBUST path uses 1.349.)
#
# Regime-indicator terminals (RegimeAboveLow/IsHigh/IsPremium) and PredRegime are
# FIXED {0,1}->{-1,+1} affine maps, never data-fit — they keep {0.5, 0.5} even in
# force-robust mode.
ROBUST_IQR_TO_SIGMA: float = 1.349  # 2 * Phi^-1(0.75) for N(0,1); normal-consistent
_REGIME_SKIP = frozenset({"RegimeAboveLow", "RegimeIsHigh", "RegimeIsPremium", "PredRegime"})


# Minimum scale floor. Must survive round(., 6): a bare 1e-12 floor rounds to
# exactly 0.0, which then trips normalize()'s `scale < 1e-12` guard and zeroes the
# terminal. 1e-6 is the smallest value that round(., 6) preserves, and is far below
# any real terminal scale (smallest frozen scale is PredGammaAccel=2e-4).
_MIN_SCALE: float = 1e-6


def _fit_one_stat(arr, method: str, robust: bool):
    """Fit a single (center, scale, method) from a 1-D finite array.

    `robust=False` reproduces the historical per-terminal behaviour: use the
    frozen `method` ("robust" -> median, IQR/1.35; "standard" -> mean, std). This
    is the back-compat path (existing callers/tests rely on per-terminal method).

    `robust=True` FORCES robust median / (IQR/1.349) for EVERY continuous terminal
    regardless of its frozen method (ITEM #4 — the per-fold pipeline path).

    DEGENERATE-IQR GUARD (robust path only): some bounded terminals concentrate
    >50% of their mass on a single boundary value (e.g. GridReliability is 1.0 on
    ~94% of bars), so median == p25 == p75 and the IQR is exactly 0 even though the
    terminal has real spread (std=0.175). A 0-IQR scale would (after the floor)
    collapse the whole terminal to a constant 0 in normalized space — destroying a
    live signal. When the IQR is ~0 we fall back to the std as the scale (keeping
    the ROBUST median as the center): robust-center + std-scale is a valid hybrid
    for boundary-degenerate distributions and preserves the terminal's
    discriminative scale. If std is ALSO ~0 the terminal is genuinely constant and
    the _MIN_SCALE floor applies (normalize then returns 0 either way — benign).
    The SAME min-scale floor + NaN-stripping (caller) guards apply in both branches.
    The returned method tag reflects what was actually fit (self-describing fold
    stats); the degenerate fallback is tagged "robust_std_fallback".
    """
    import numpy as np
    if robust:
        center = float(np.median(arr))
        q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
        iqr_scale = (q75 - q25) / ROBUST_IQR_TO_SIGMA
        if iqr_scale > _MIN_SCALE:
            return (round(center, 6), round(iqr_scale, 6), "robust")
        # Degenerate IQR (central 50% is a single value) — keep the robust median
        # center but use std as the scale so a real-spread terminal is not zeroed.
        std_scale = float(np.std(arr))
        scale = max(std_scale, _MIN_SCALE)
        tag = "robust_std_fallback" if std_scale > _MIN_SCALE else "robust"
        return (round(center, 6), round(scale, 6), tag)
    if method == "robust":
        center = float(np.median(arr))
        q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
        scale = max((q75 - q25) / 1.35, _MIN_SCALE)
        return (round(center, 6), round(scale, 6), "robust")
    center = float(np.mean(arr))
    scale = max(float(np.std(arr)), _MIN_SCALE)
    return (round(center, 6), round(scale, 6), "standard")


def normalize(name: str, raw_value: float) -> float:
    """Normalize a raw terminal value to approximately N(0,1).

    Args:
        name: terminal name (e.g., "ATM_IV", "VIXSpot")
        raw_value: the raw value as it appears in L1 Parquet

    Returns:
        Normalized value. If the terminal has no stats, returns raw_value unchanged.
    """
    stats = TERMINAL_NORM_STATS.get(name)
    if stats is None:
        return raw_value
    center, scale, _ = stats
    if scale < 1e-12:
        return 0.0
    return (raw_value - center) / scale


def denormalize(name: str, norm_value: float) -> float:
    """Inverse of normalize: convert normalized value back to raw scale.

    Used for debugging and threshold conversion verification.
    """
    stats = TERMINAL_NORM_STATS.get(name)
    if stats is None:
        return norm_value
    center, scale, _ = stats
    return norm_value * scale + center


def normalize_threshold(name: str, raw_threshold: float) -> float:
    """Convert a raw threshold to normalized space.

    Used by templates.py to convert seed tree constants.
    e.g., ATM_IV > 0.18 becomes ATM_IV_norm > normalize_threshold("ATM_IV", 0.18)
    """
    return normalize(name, raw_threshold)


def recalibrate_seed_thresholds(
    tree: "Node",
    fold_norm_stats: Dict[str, Tuple[float, float, str]],
    skip_terminals: "Optional[set]" = None,
) -> None:
    """Re-normalize EphReal thresholds in seed trees to match per-fold stats.

    Seed trees are constructed at import time using frozen TERMINAL_NORM_STATS.
    When per-fold normalization is active, the data is normalized with different
    center/scale, but the EphReal constants still reflect the frozen stats.
    This function walks the tree and recalibrates in-place.

    For every GT/LT node where one child is a named terminal with norm stats
    and the other is an EphReal:
      1. Denormalize the EphReal value using frozen stats → raw threshold
      2. Re-normalize using fold_norm_stats → corrected threshold
      3. Update the EphReal value in-place

    Args:
        tree: Root node of a seed tree (modified in-place).
        fold_norm_stats: Per-fold normalization stats from compute_norm_stats_from_data().
    """
    from layer2.grammar import FuncNode, TermNode

    if not isinstance(tree, FuncNode):
        return

    # Check if this is a GT or LT comparison with (terminal, EphReal) children
    if tree.name in ("GT", "LT") and len(tree.children) == 2:
        c0, c1 = tree.children
        # Pattern: GT/LT(NamedTerminal, EphReal) or GT/LT(EphReal, NamedTerminal)
        for term_child, eph_child in [(c0, c1), (c1, c0)]:
            if (isinstance(term_child, TermNode) and term_child.name != "EphReal"
                    and isinstance(eph_child, TermNode) and eph_child.name == "EphReal"):
                # BLOCKER fix (2026-06-03): VRP seed thresholds are authored DIRECTLY
                # in fold-normalized space (tagged fold_normalized=True). They are NOT
                # frozen-authored, so the frozen→raw→fold transform below would
                # double-transform them (gutting selectivity). Skip them; legacy
                # frozen-authored seeds (no tag) recalibrate normally.
                if getattr(eph_child, "fold_normalized", False):
                    break
                tname = term_child.name
                # plan #4 LOW fix (2026-06-04 review): under trailing-rolling norm the
                # LEVEL terminals are no longer in the fold-EXPANDING frame, so their
                # frozen thresholds must NOT be recalibrated to expanding stats — leave
                # them as ~z-scores (like the tagged VRP seeds). Caller passes
                # LEVEL_TERMINALS as skip_terminals when norm_mode='trailing_rolling'.
                if skip_terminals and tname in skip_terminals:
                    break
                frozen = TERMINAL_NORM_STATS.get(tname)
                fold = fold_norm_stats.get(tname)
                if frozen and fold:
                    f_center, f_scale, _ = frozen
                    n_center, n_scale, _ = fold
                    if f_scale > 1e-12 and n_scale > 1e-12:
                        # Denormalize from frozen → raw → re-normalize to fold
                        raw = eph_child.value * f_scale + f_center
                        eph_child.value = (raw - n_center) / n_scale
                break  # only one pattern per GT/LT node

    # Recurse into children
    for child in tree.children:
        recalibrate_seed_thresholds(child, fold_norm_stats, skip_terminals=skip_terminals)


def compute_norm_stats_from_data(
    data: "pd.DataFrame",
    robust: bool = False,
) -> Dict[str, Tuple[float, float, str]]:
    """Compute normalization stats from a DataFrame (per-fold alternative to frozen constants).

    For each numeric column that appears in TERMINAL_NORM_STATS, computes stats
    and returns a dict with the same format as TERMINAL_NORM_STATS.

    `robust=False` (default — backward-compatible): match each terminal's frozen
    method (robust -> median, IQR/1.35; standard -> mean, std).

    `robust=True` (ITEM #4): FORCE robust median / (IQR/1.349) for every
    continuous terminal regardless of its frozen method — the heavy-tail-resistant
    per-fold scaling that recovers within-regime IV discrimination (regime
    indicators stay {0.5, 0.5}). The per-fold GP pipeline uses this path.

    This enables per-fold normalization to prevent look-ahead bias:
    Fold 1's training data (to 2024-03-29) should not use stats computed
    from the full period (to 2024-09-30).
    """
    import numpy as np
    stats: Dict[str, Tuple[float, float, str]] = {}
    for col, (_, _, method) in TERMINAL_NORM_STATS.items():
        if col not in data.columns:
            # Use frozen constant if column not in data
            stats[col] = TERMINAL_NORM_STATS[col]
            continue
        if robust and col in _REGIME_SKIP:
            # Fixed {0,1}->{-1,+1} affine maps — never data-fit.
            stats[col] = TERMINAL_NORM_STATS[col]
            continue
        arr = data[col].values.astype(np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            stats[col] = TERMINAL_NORM_STATS[col]
            continue
        stats[col] = _fit_one_stat(arr, method, robust)
    return stats


def compute_norm_stats_from_arrays(
    arrays: "Dict[str, object]",
    robust: bool = False,
) -> Dict[str, Tuple[float, float, str]]:
    """Per-fold norm stats from already-SYNTHESIZED terminal arrays.

    Unlike compute_norm_stats_from_data (which reads raw DataFrame COLUMNS and
    therefore only covers the ~8 base bypass scalars actually present in the
    minute Parquet), this fits stats over the terminal_data dict produced by
    prepare_terminal_data(..., normalize_terminals=False). That dict ALSO holds
    the synthesized terminals (ThetaUrgency, RV5d, VIXChange, SPXReturn3d,
    VIXMean5d, IVRVGap5d, ATM_IV_5m, RealizedVol30m_5m, BarOfDay, SessionReturn,
    the regime indicators, …), so every terminal with frozen stats gets a
    TRAIN-only, per-fold (center, scale) instead of silently falling back to the
    frozen date<=2024-09-30 constants (an M1 per-fold look-ahead leak: for early
    folds the frozen constants span the fold's own validation window).

    For each named terminal in TERMINAL_NORM_STATS that has a finite 1-D array in
    `arrays`, fit a per-fold (center, scale). Inputs MUST be the un-normalized
    (raw-scale) synthesized arrays — pass normalize_terminals=False. Terminals
    absent from `arrays`, non-1-D (typed vectors), or with <10 finite samples
    keep their frozen constant. Binary regime-indicator terminals (constant 0/1
    affine maps) keep their frozen {0.5, 0.5} so {0,1}→{-1,+1} is preserved.

    `robust` (ITEM #4, docs/viable_run_plan_2026_06_03.md):
      - False (default — backward-compatible): fit the SAME robust (median,
        IQR/1.35) or standard (mean, std) METHOD recorded in the frozen constants
        per terminal. Existing callers/tests rely on this per-terminal method.
      - True: FORCE robust median / (IQR/1.349) for EVERY continuous terminal,
        regardless of its frozen method — the heavy-tail-resistant per-fold scaling
        that recovers within-regime IV discrimination (measured: daily ATM_IV IQR
        gap 0.611σ -> 1.207σ on fold-1). This is the path the per-fold GP pipeline
        (run_experiment) uses. Regime indicators still keep {0.5, 0.5}.
    """
    import numpy as np
    stats: Dict[str, Tuple[float, float, str]] = {}
    for name, (_, _, method) in TERMINAL_NORM_STATS.items():
        # Regime indicators are fixed affine {0,1}->{-1,+1} maps, not data-fit; and
        # PredRegime is removed from the float terminal pool (kept as int32). Never
        # re-fit these from data (in EITHER mode).
        if name in _REGIME_SKIP:
            stats[name] = TERMINAL_NORM_STATS[name]
            continue
        val = arrays.get(name)
        if val is None or not isinstance(val, np.ndarray) or val.ndim != 1:
            stats[name] = TERMINAL_NORM_STATS[name]
            continue
        arr = val.astype(np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            stats[name] = TERMINAL_NORM_STATS[name]
            continue
        stats[name] = _fit_one_stat(arr, method, robust)
    return stats


# ---------------------------------------------------------------------------
# Regeneration script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Recompute stats from L1 Parquet and print for manual verification."""
    import numpy as np
    import pandas as pd

    PARQUET_PATH = "raw_data/local_store/l1_cnn.parquet"
    TRAIN_END = "2024-09-30"

    df = pd.read_parquet(PARQUET_PATH)
    train = df[df["date"] <= TRAIN_END]
    print(f"Train split: {len(train)} rows, {train['date'].nunique()} dates")
    print(f"Date range: {train['date'].min()} to {train['date'].max()}")
    print()

    # Bypass scalars
    bypass = ["ATM_IV", "VIXSpot", "VIXTermSlope", "RealizedVol30m",
              "MinutesToClose", "RawSpread", "DeltaSpread1", "DeltaSpread5"]
    # Probe scalars
    probes = ["PredRV15", "PredRV30", "PredSpread", "PredRegime"]

    heavy_tailed = {"ATM_IV", "VIXSpot", "VIXTermSlope", "RawSpread"}

    for col in bypass + probes:
        if col not in train.columns:
            print(f"{col}: NOT IN PARQUET")
            continue
        vals = train[col].dropna()
        mean = vals.mean()
        std = vals.std()
        med = vals.median()
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        robust_std = iqr / 1.35

        if col in heavy_tailed:
            method = "robust"
            center, scale = med, robust_std
        else:
            method = "standard"
            center, scale = mean, std

        print(f'    "{col}": ({center:.4f}, {scale:.4f}, "{method}"),')
        print(f"        # {method}: center={center:.4f}, scale={scale:.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")
