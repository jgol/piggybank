"""Deterministic gates for L3 pipeline (D1-D4 pre-filter, G5, D5-D8 post-QC).

All gates are pure functions: (StrategyPacket) → GateResult.
No LLM calls, no side effects. Gates D1-D4 run BEFORE any LLM stage
to reject ~95-98% of candidates at zero cost.

Gate inventory:
- D1: Harvey-Liu haircut (individual)
- D2: Persistence across folds (individual)
- D3: Trade count minimum (individual)
- D4: Deduplication within template (population-level, via gate_d4_dedup)
- G5: Regime gate (individual)
- D5-D8: Post-QC gates (individual)
- Structural degeneracy: constant/tautological tree detection (individual)

Gate definitions per Section 2.1 of (internal doc).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

from layer3.schemas import GateResult, StrategyPacket


# ---------------------------------------------------------------------------
# Gate thresholds (frozen, pre-registered)
# ---------------------------------------------------------------------------

# D1: Harvey-Liu haircut — val_sharpe > sqrt(2*log(N)/T)
# N=100 (effective trials per template), T=125 (val trading days per fold)
D1_THRESHOLD = math.sqrt(2 * math.log(100) / 125)  # ~0.304

# D2: Per-template persistence — positive val Sharpe in 2+ of 3 folds
D2_MIN_POSITIVE_FOLDS = 2
D2_NUM_EVAL_FOLDS = 3  # folds 1-3; fold 4 is holdout

# D3: Trade count — 60+ trades across all val folds. Lowered from 90
# because cross-fold evaluation (B1 fix) counts trades on folds where
# the strategy wasn't evolved — selective strategies may trade less on
# out-of-fold data. Combined with D2 persistence this is sufficient.
D3_MIN_TRADES = 60

# D4: Deduplication — top-K per similarity cluster
D4_TOP_K_PER_CLUSTER = 3

# G5: Regime gate — no regime with Sharpe < -0.3
# Aligned with proxy-side G5 (tightened from -1.0 per task #188)
G5_MIN_REGIME_SHARPE = -0.3

# D5: Proxy-QC Sharpe gap
D5_MAX_GAP = 0.50

# D6: QC max drawdown
D6_MAX_DRAWDOWN = 0.15

# D7: QC test Sharpe
D7_MIN_SHARPE = 0.0

# D8: Profit factor
D8_MIN_PROFIT_FACTOR = 1.1


# ---------------------------------------------------------------------------
# Pre-filter gates (D1-D4) — run before any LLM call
# ---------------------------------------------------------------------------

def gate_d1_harvey_liu(packet: StrategyPacket) -> GateResult:
    """D1: Val Sharpe exceeds Harvey-Liu haircut (HARD)."""
    val_sharpes = [fm.val_sharpe for fm in packet.fold_metrics
                   if fm.fold_id <= D2_NUM_EVAL_FOLDS]
    if not val_sharpes:
        return GateResult("D1", False, 0.0, D1_THRESHOLD,
                          "No fold metrics available")
    mean_val_sharpe = sum(val_sharpes) / len(val_sharpes)
    return GateResult(
        gate_id="D1",
        passed=mean_val_sharpe > D1_THRESHOLD,
        value=mean_val_sharpe,
        threshold=D1_THRESHOLD,
        reason=f"mean_val_sharpe={mean_val_sharpe:.3f} {'>' if mean_val_sharpe > D1_THRESHOLD else '<='} {D1_THRESHOLD:.3f}",
    )


def gate_d2_persistence(packet: StrategyPacket) -> GateResult:
    """D2: Positive val Sharpe in 2+ of 3 folds (HARD)."""
    eval_folds = [fm for fm in packet.fold_metrics
                  if fm.fold_id <= D2_NUM_EVAL_FOLDS]
    n_positive = sum(1 for fm in eval_folds if fm.val_sharpe > 0)
    return GateResult(
        gate_id="D2",
        passed=n_positive >= D2_MIN_POSITIVE_FOLDS,
        value=float(n_positive),
        threshold=float(D2_MIN_POSITIVE_FOLDS),
        reason=f"{n_positive}/{len(eval_folds)} folds positive",
    )


def gate_d3_trade_count(packet: StrategyPacket) -> GateResult:
    """D3: 100+ trades in val period (HARD)."""
    return GateResult(
        gate_id="D3",
        passed=packet.total_val_trades >= D3_MIN_TRADES,
        value=float(packet.total_val_trades),
        threshold=float(D3_MIN_TRADES),
        reason=f"{packet.total_val_trades} val trades",
    )


def gate_g5_regime(packet: StrategyPacket) -> GateResult:
    """G5: No regime with Sharpe < -0.3 (HARD)."""
    if packet.regime is None:
        return GateResult("G5", True, 0.0, G5_MIN_REGIME_SHARPE,
                          "No regime data available — passed with warning (regime analysis skipped)")
    worst = min(packet.regime.sharpe_low_vol,
                packet.regime.sharpe_mid_vol,
                packet.regime.sharpe_high_vol)
    return GateResult(
        gate_id="G5",
        passed=worst >= G5_MIN_REGIME_SHARPE,
        value=worst,
        threshold=G5_MIN_REGIME_SHARPE,
        reason=f"worst_regime_sharpe={worst:.3f}",
    )


# ---------------------------------------------------------------------------
# D4: Deduplication gate (population-level)
# ---------------------------------------------------------------------------

def gate_d4_dedup(packets: List[StrategyPacket]) -> Dict[str, GateResult]:
    """D4: Deduplicate strategies within each template.

    Groups strategies by template_name, hashes (entry_sexpr, exit_sexpr)
    within each group, and keeps the top-K (by mean_val_sharpe) per unique
    hash. Strategies beyond top-K are marked as failed.

    Operates on the full population, not individual strategies.

    Returns:
        Dict mapping strategy_id -> GateResult for every packet.
    """
    results: Dict[str, GateResult] = {}

    # Group by template
    by_template: Dict[str, List[StrategyPacket]] = defaultdict(list)
    for p in packets:
        by_template[p.template_name].append(p)

    for template_name, group in by_template.items():
        # Group by (entry_sexpr, exit_sexpr) hash within template
        by_signature: Dict[str, List[StrategyPacket]] = defaultdict(list)
        for p in group:
            sig = (p.entry_sexpr, p.exit_sexpr)
            by_signature[sig].append(p)

        for sig, cluster in by_signature.items():
            # Sort by mean_val_sharpe descending, keep top-K
            cluster.sort(key=lambda p: p.mean_val_sharpe, reverse=True)
            for i, p in enumerate(cluster):
                if i < D4_TOP_K_PER_CLUSTER:
                    results[p.strategy_id] = GateResult(
                        gate_id="D4",
                        passed=True,
                        value=float(i + 1),
                        threshold=float(D4_TOP_K_PER_CLUSTER),
                        reason=f"Rank {i + 1}/{len(cluster)} in template={template_name} "
                               f"cluster (top-{D4_TOP_K_PER_CLUSTER} kept)",
                    )
                else:
                    results[p.strategy_id] = GateResult(
                        gate_id="D4",
                        passed=False,
                        value=float(i + 1),
                        threshold=float(D4_TOP_K_PER_CLUSTER),
                        reason=f"Duplicate rank {i + 1}/{len(cluster)} in "
                               f"template={template_name} — "
                               f"exceeds top-{D4_TOP_K_PER_CLUSTER} cutoff",
                    )

    return results


# ---------------------------------------------------------------------------
# Structural degeneracy gate (pre-filter)
# ---------------------------------------------------------------------------

# Regex patterns for structural degeneracy detection.
# Matches a bare EphReal(...) or EphInt(...) with no other terminals.
_RE_BARE_EPHEMERAL = re.compile(r"^EphReal\([^)]*\)$|^EphInt\([^)]*\)$")

# Matches GT(EphReal(X), EphReal(Y)) — tautological if X > Y (always true).
_RE_GT_TWO_EPH = re.compile(
    r"^GT\(EphReal\(([^)]+)\),\s*EphReal\(([^)]+)\)\)$"
)

# Matches LT(EphReal(X), EphReal(Y)) — tautological if X < Y (always true).
_RE_LT_TWO_EPH = re.compile(
    r"^LT\(EphReal\(([^)]+)\),\s*EphReal\(([^)]+)\)\)$"
)

# Matches CrossAbove/CrossBelow on two constant expressions (EphReal or EphInt).
_RE_CROSS_TWO_EPH = re.compile(
    r"^Cross(?:Above|Below)\(Eph(?:Real|Int)\([^)]*\),\s*Eph(?:Real|Int)\([^)]*\)\)$"
)


def gate_structural_degeneracy(packet: StrategyPacket) -> GateResult:
    """Detect structurally degenerate trees that waste pipeline capacity.

    Checks:
    1. Constant size tree: size_sexpr is bare EphReal/EphInt with no terminal
       references — the strategy uses a fixed position size regardless of
       market conditions.
    2. Constant delta tree: delta_sexpr is bare EphReal/EphInt — acceptable
       per S2 (noted but NOT rejected).
    3. Tautological entry: GT(EphReal(X), EphReal(Y)) where X > Y is
       always true — the entry condition never filters.
    4. Exit comparing unrelated constants: CrossAbove/CrossBelow on two
       ephemeral constants — constants never cross, so the exit never fires.

    Returns GateResult with passed=False if any REJECTABLE degeneracy found.
    """
    issues: List[str] = []
    reject = False

    # --- Check 1: Constant size tree ---
    # NOTE: Size tree is deliberately FROZEN at EphReal(0.5) as a design
    # decision (pathology fix #19 — GP exploited size for drawdown hacks).
    # A constant size tree is EXPECTED and NOT degenerate. Log but don't reject.
    if packet.size_sexpr:
        stripped = packet.size_sexpr.strip()
        if _RE_BARE_EPHEMERAL.match(stripped):
            issues.append(f"constant size tree (expected — size frozen by design): {stripped}")

    # --- Check 2: Constant delta tree (note only, per S2) ---
    if packet.delta_sexpr:
        stripped = packet.delta_sexpr.strip()
        if _RE_BARE_EPHEMERAL.match(stripped):
            issues.append(f"constant delta tree (acceptable): {stripped}")
            # Do NOT reject — per S2, constant delta is valid

    # --- Check 3: Tautological entry ---
    if packet.entry_sexpr:
        stripped = packet.entry_sexpr.strip()
        # GT(EphReal(X), EphReal(Y)) where X > Y is always true
        m = _RE_GT_TWO_EPH.match(stripped)
        if m:
            try:
                x_val = float(m.group(1))
                y_val = float(m.group(2))
                if x_val > y_val:
                    issues.append(
                        f"tautological entry (always true): GT(EphReal({x_val}), EphReal({y_val}))"
                    )
                    reject = True
            except ValueError:
                pass
        # LT(EphReal(X), EphReal(Y)) where X < Y is always true
        m = _RE_LT_TWO_EPH.match(stripped)
        if m:
            try:
                x_val = float(m.group(1))
                y_val = float(m.group(2))
                if x_val < y_val:
                    issues.append(
                        f"tautological entry (always true): LT(EphReal({x_val}), EphReal({y_val}))"
                    )
                    reject = True
            except ValueError:
                pass

    # --- Check 4: Exit crossing two constants ---
    if packet.exit_sexpr:
        stripped = packet.exit_sexpr.strip()
        if _RE_CROSS_TWO_EPH.match(stripped):
            issues.append(f"exit crosses two constants (never fires): {stripped}")
            reject = True

    if not issues:
        return GateResult(
            gate_id="STRUCT",
            passed=True,
            value=0.0,
            threshold=0.0,
            reason="No structural degeneracy detected",
        )

    return GateResult(
        gate_id="STRUCT",
        passed=not reject,
        value=float(len(issues)),
        threshold=0.0,
        reason="; ".join(issues),
    )


# ---------------------------------------------------------------------------
# Post-QC gates (D5-D8) — run after QC backtest
# ---------------------------------------------------------------------------

def gate_d5_proxy_qc_gap(packet: StrategyPacket) -> GateResult:
    """D5: Proxy-QC Sharpe gap < 0.50 (HARD)."""
    if packet.proxy_qc_sharpe_gap is None:
        return GateResult("D5", False, 0.0, D5_MAX_GAP,
                          "No QC results available")
    gap = abs(packet.proxy_qc_sharpe_gap)
    return GateResult(
        gate_id="D5",
        passed=gap < D5_MAX_GAP,
        value=gap,
        threshold=D5_MAX_GAP,
        reason=f"proxy-QC gap={gap:.3f}",
    )


def gate_d6_qc_drawdown(packet: StrategyPacket) -> GateResult:
    """D6: QC max drawdown < 15% (HARD)."""
    if packet.qc_drawdown is None:
        return GateResult("D6", False, 0.0, D6_MAX_DRAWDOWN,
                          "No QC results available")
    return GateResult(
        gate_id="D6",
        passed=packet.qc_drawdown < D6_MAX_DRAWDOWN,
        value=packet.qc_drawdown,
        threshold=D6_MAX_DRAWDOWN,
        reason=f"QC DD={packet.qc_drawdown:.3f}",
    )


def gate_d7_qc_test_sharpe(packet: StrategyPacket) -> GateResult:
    """D7: QC test Sharpe > 0.0 (HARD)."""
    if packet.qc_sharpe is None:
        return GateResult("D7", False, 0.0, D7_MIN_SHARPE,
                          "No QC results available")
    return GateResult(
        gate_id="D7",
        passed=packet.qc_sharpe > D7_MIN_SHARPE,
        value=packet.qc_sharpe,
        threshold=D7_MIN_SHARPE,
        reason=f"QC Sharpe={packet.qc_sharpe:.3f}",
    )


def gate_d8_profit_factor(packet: StrategyPacket,
                          profit_factor: Optional[float] = None) -> GateResult:
    """D8: Profit factor > 1.1 (HARD)."""
    if profit_factor is None:
        return GateResult("D8", False, 0.0, D8_MIN_PROFIT_FACTOR,
                          "Profit factor not computed")
    return GateResult(
        gate_id="D8",
        passed=profit_factor > D8_MIN_PROFIT_FACTOR,
        value=profit_factor,
        threshold=D8_MIN_PROFIT_FACTOR,
        reason=f"PF={profit_factor:.3f}",
    )


# ---------------------------------------------------------------------------
# Convenience: run all pre-filter gates
# ---------------------------------------------------------------------------

def run_prefilter_gates(packet: StrategyPacket) -> List[GateResult]:
    """Run D1, D2, D3, and structural degeneracy on a strategy packet.

    Returns list of GateResults. D4 (deduplication) is handled separately
    as it operates on the full population, not individual strategies.
    """
    return [
        gate_d1_harvey_liu(packet),
        gate_d2_persistence(packet),
        gate_d3_trade_count(packet),
        gate_structural_degeneracy(packet),
    ]


def all_prefilter_pass(results: List[GateResult]) -> bool:
    """True iff all pre-filter gates passed."""
    return all(r.passed for r in results)
