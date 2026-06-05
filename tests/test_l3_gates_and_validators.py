"""Tests for L3 pipeline modules: gates, mutation_validator, stages, llm_client."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from layer3.schemas import (
    Decision, FoldMetrics, MutationType, RegimeDecomposition,
    StrategyPacket, TreeMutation, TreeSlot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_packet(
    val_sharpes=(1.0, 0.8, 1.2),
    total_trades=450,
    regime_sharpes=(2.0, 1.0, -0.5),
) -> StrategyPacket:
    """Build a test StrategyPacket with configurable metrics."""
    return StrategyPacket(
        strategy_id="test_001",
        template_name="bull_put_credit",
        condition="scalar-only",
        entry_sexpr="LT(OvernightGap, Mul(VIXChange, BarOfDay))",
        exit_sexpr="GT(MinutesToClose, EphReal(0.5))",
        size_sexpr="Div(EphReal(0.5), ATM_IV)",
        fold_metrics=[
            FoldMetrics(i + 1, 3.0, s, 0.05, total_trades // 3, 0.55)
            for i, s in enumerate(val_sharpes)
        ],
        mean_val_sharpe=sum(val_sharpes) / len(val_sharpes),
        total_val_trades=total_trades,
        regime=RegimeDecomposition(
            sharpe_low_vol=regime_sharpes[0],
            sharpe_mid_vol=regime_sharpes[1],
            sharpe_high_vol=regime_sharpes[2],
            pct_trades_low_vol=0.5,
            pct_trades_mid_vol=0.3,
            pct_trades_high_vol=0.2,
        ),
    )


# ===========================================================================
# 1.1 Gate tests
# ===========================================================================



class TestGates:
    """Deterministic pre-filter gates D1-D4 + G5."""

    def test_d1_passes_above_threshold(self):
        from layer3.gates import gate_d1_harvey_liu
        packet = _make_packet(val_sharpes=(1.0, 0.8, 1.2))
        result = gate_d1_harvey_liu(packet)
        assert result.passed is True

    def test_d1_fails_below_threshold(self):
        from layer3.gates import gate_d1_harvey_liu
        packet = _make_packet(val_sharpes=(0.1, 0.05, 0.15))
        result = gate_d1_harvey_liu(packet)
        assert result.passed is False

    def test_d2_requires_2_positive_folds(self):
        from layer3.gates import gate_d2_persistence
        # 1 positive fold — should fail
        packet = _make_packet(val_sharpes=(0.5, -0.3, -0.1))
        result = gate_d2_persistence(packet)
        assert result.passed is False

    def test_d2_passes_with_2_positive(self):
        from layer3.gates import gate_d2_persistence
        packet = _make_packet(val_sharpes=(0.5, 0.3, -0.1))
        result = gate_d2_persistence(packet)
        assert result.passed is True

    def test_d3_fails_below_100_trades(self):
        from layer3.gates import gate_d3_trade_count
        packet = _make_packet(total_trades=50)
        result = gate_d3_trade_count(packet)
        assert result.passed is False

    def test_d3_passes_above_100_trades(self):
        from layer3.gates import gate_d3_trade_count
        packet = _make_packet(total_trades=150)
        result = gate_d3_trade_count(packet)
        assert result.passed is True

    def test_g5_rejects_bad_regime(self):
        from layer3.gates import gate_g5_regime
        packet = _make_packet(regime_sharpes=(1.0, 0.5, -1.5))
        result = gate_g5_regime(packet)
        assert result.passed is False

    def test_g5_passes_acceptable_regime(self):
        from layer3.gates import gate_g5_regime
        # G5 threshold is -0.3 (aligned with proxy-side per task #188)
        packet = _make_packet(regime_sharpes=(1.0, 0.5, -0.2))
        result = gate_g5_regime(packet)
        assert result.passed is True

    def test_run_prefilter_all_pass(self):
        from layer3.gates import run_prefilter_gates, all_prefilter_pass
        packet = _make_packet()
        results = run_prefilter_gates(packet)
        assert all_prefilter_pass(results)

    def test_run_prefilter_mixed(self):
        from layer3.gates import run_prefilter_gates, all_prefilter_pass
        packet = _make_packet(total_trades=30)  # D3 will fail
        results = run_prefilter_gates(packet)
        assert not all_prefilter_pass(results)

    def test_prefilter_includes_struct_gate(self):
        from layer3.gates import run_prefilter_gates
        packet = _make_packet()
        results = run_prefilter_gates(packet)
        gate_ids = [r.gate_id for r in results]
        assert "STRUCT" in gate_ids


class TestG5Warning:
    """B2: G5 regime gate warning message when regime is None."""

    def test_g5_none_regime_warns(self):
        from layer3.gates import gate_g5_regime
        packet = _make_packet()
        packet.regime = None
        result = gate_g5_regime(packet)
        assert result.passed is True
        assert "warning" in result.reason.lower()
        assert "skipped" in result.reason.lower()


class TestD4Dedup:
    """B3: D4 deduplication gate (population-level)."""

    def test_unique_strategies_all_pass(self):
        from layer3.gates import gate_d4_dedup
        packets = [
            StrategyPacket(strategy_id="s1", template_name="IC", condition="a",
                           entry_sexpr="GT(ATM_IV, EphReal(0.1))", exit_sexpr="X",
                           size_sexpr="Y", mean_val_sharpe=0.5),
            StrategyPacket(strategy_id="s2", template_name="IC", condition="a",
                           entry_sexpr="LT(VIXSpot, EphReal(20))", exit_sexpr="X",
                           size_sexpr="Y", mean_val_sharpe=0.3),
        ]
        results = gate_d4_dedup(packets)
        assert results["s1"].passed is True
        assert results["s2"].passed is True

    def test_duplicates_beyond_top_k_fail(self):
        from layer3.gates import gate_d4_dedup, D4_TOP_K_PER_CLUSTER
        # Create K+1 duplicates (same entry+exit)
        packets = []
        for i in range(D4_TOP_K_PER_CLUSTER + 1):
            packets.append(StrategyPacket(
                strategy_id=f"s{i}", template_name="IC", condition="a",
                entry_sexpr="GT(ATM_IV, EphReal(0.1))", exit_sexpr="LT(X, Y)",
                size_sexpr="Y", mean_val_sharpe=float(D4_TOP_K_PER_CLUSTER - i),
            ))
        results = gate_d4_dedup(packets)
        # First K should pass, last should fail
        for i in range(D4_TOP_K_PER_CLUSTER):
            assert results[f"s{i}"].passed is True
        assert results[f"s{D4_TOP_K_PER_CLUSTER}"].passed is False

    def test_different_templates_independent(self):
        from layer3.gates import gate_d4_dedup
        packets = [
            StrategyPacket(strategy_id="s1", template_name="IC", condition="a",
                           entry_sexpr="GT(ATM_IV, EphReal(0.1))", exit_sexpr="X",
                           size_sexpr="Y", mean_val_sharpe=0.5),
            StrategyPacket(strategy_id="s2", template_name="BPC", condition="a",
                           entry_sexpr="GT(ATM_IV, EphReal(0.1))", exit_sexpr="X",
                           size_sexpr="Y", mean_val_sharpe=0.3),
        ]
        results = gate_d4_dedup(packets)
        # Different templates, both should pass even with same entry+exit
        assert results["s1"].passed is True
        assert results["s2"].passed is True

    def test_empty_population(self):
        from layer3.gates import gate_d4_dedup
        results = gate_d4_dedup([])
        assert results == {}

    def test_ranking_by_sharpe(self):
        from layer3.gates import gate_d4_dedup
        # 4 duplicates, top-3 kept — verify correct ones survive
        packets = [
            StrategyPacket(strategy_id="low", template_name="IC", condition="a",
                           entry_sexpr="E", exit_sexpr="X", size_sexpr="Y",
                           mean_val_sharpe=0.1),
            StrategyPacket(strategy_id="high", template_name="IC", condition="a",
                           entry_sexpr="E", exit_sexpr="X", size_sexpr="Y",
                           mean_val_sharpe=0.9),
            StrategyPacket(strategy_id="mid", template_name="IC", condition="a",
                           entry_sexpr="E", exit_sexpr="X", size_sexpr="Y",
                           mean_val_sharpe=0.5),
            StrategyPacket(strategy_id="lowest", template_name="IC", condition="a",
                           entry_sexpr="E", exit_sexpr="X", size_sexpr="Y",
                           mean_val_sharpe=0.05),
        ]
        results = gate_d4_dedup(packets)
        assert results["high"].passed is True
        assert results["mid"].passed is True
        assert results["low"].passed is True
        assert results["lowest"].passed is False


class TestStructuralDegeneracy:
    """S3: Structural degeneracy gate."""

    def test_clean_strategy_passes(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="clean", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="Div(EphReal(0.5), ATM_IV)",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True
        assert r.gate_id == "STRUCT"

    def test_constant_size_tree_noted_not_rejected(self):
        """Size tree is frozen by design (pathology fix #19) -- note but don't reject."""
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="const_size", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="EphReal(0.5)",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True  # noted, not rejected
        assert "constant size tree" in r.reason

    def test_constant_delta_tree_passes_with_note(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="const_delta", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="Div(EphReal(0.5), ATM_IV)",
            delta_sexpr="EphReal(0.25)",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True  # NOT rejected
        assert "constant delta tree (acceptable)" in r.reason

    def test_tautological_entry_gt_x_gt_y(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="taut", template_name="IC", condition="a",
            entry_sexpr="GT(EphReal(5.0), EphReal(2.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="ATM_IV",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is False
        assert "tautological entry" in r.reason

    def test_gt_x_lt_y_not_flagged_as_tautological(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="false_gt", template_name="IC", condition="a",
            entry_sexpr="GT(EphReal(2.0), EphReal(5.0))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="ATM_IV",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True  # always-false but not always-true

    def test_exit_cross_two_constants_fails(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="cross_const", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="CrossAbove(EphReal(0.5), EphReal(0.3))",
            size_sexpr="ATM_IV",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is False
        assert "exit crosses two constants" in r.reason

    def test_exit_crossbelow_two_constants_fails(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="cross_below", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="CrossBelow(EphReal(1.0), EphReal(0.5))",
            size_sexpr="ATM_IV",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is False
        assert "exit crosses two constants" in r.reason

    def test_normal_cross_passes(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="cross_ok", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="CrossAbove(ATM_IV, EphReal(0.3))",
            size_sexpr="ATM_IV",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True

    def test_multiple_issues_reported(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="multi", template_name="IC", condition="a",
            entry_sexpr="GT(EphReal(5.0), EphReal(2.0))",
            exit_sexpr="CrossAbove(EphReal(0.5), EphReal(0.3))",
            size_sexpr="EphReal(0.5)",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is False
        assert r.value == 3.0  # 3 issues
        assert "tautological" in r.reason
        assert "constant size" in r.reason
        assert "exit crosses" in r.reason

    def test_empty_sexprs_pass(self):
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="empty", template_name="IC", condition="a",
            entry_sexpr="", exit_sexpr="", size_sexpr="",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True

    def test_ephint_constant_size_noted_not_rejected(self):
        """EphInt constant size -- same as EphReal: noted, not rejected."""
        from layer3.gates import gate_structural_degeneracy
        p = StrategyPacket(
            strategy_id="ephint_size", template_name="IC", condition="a",
            entry_sexpr="GT(ATM_IV, EphReal(0.18))",
            exit_sexpr="LT(MinutesToClose, EphReal(30))",
            size_sexpr="EphInt(3)",
        )
        r = gate_structural_degeneracy(p)
        assert r.passed is True  # noted, not rejected
        assert "constant size tree" in r.reason


class TestMutationValidator:
    """Grammar-constrained mutation validation."""

    def test_valid_terminal_swap(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.TERMINAL_SWAP, TreeSlot.ENTRY,
                         from_node="BarOfDay", to_node="ATM_IV")
        valid, _ = validate_mutation(m)
        assert valid is True

    def test_invalid_terminal_rejected(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.TERMINAL_SWAP, TreeSlot.ENTRY,
                         from_node="BarOfDay", to_node="FAKE_TERMINAL")
        valid, reason = validate_mutation(m)
        assert valid is False
        assert "not a valid" in reason

    def test_valid_function_swap(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.FUNCTION_SWAP, TreeSlot.ENTRY,
                         from_node="GT", to_node="LT")
        valid, _ = validate_mutation(m)
        assert valid is True

    def test_cross_type_function_swap_rejected(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.FUNCTION_SWAP, TreeSlot.ENTRY,
                         from_node="GT", to_node="Add")  # BOOL → REAL
        valid, reason = validate_mutation(m)
        assert valid is False
        assert "type mismatch" in reason

    def test_ephreal_in_range(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.EPHEMERAL_PERTURBATION, TreeSlot.ENTRY,
                         from_node="EphReal(0.5)", to_node="EphReal(1.5)")
        valid, _ = validate_mutation(m)
        assert valid is True

    def test_ephreal_out_of_range(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.EPHEMERAL_PERTURBATION, TreeSlot.ENTRY,
                         from_node="EphReal(0.5)", to_node="EphReal(4.0)")
        valid, reason = validate_mutation(m)
        assert valid is False
        assert "out of range" in reason

    def test_ephint_valid_value(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.EPHEMERAL_PERTURBATION, TreeSlot.ENTRY,
                         from_node="EphInt(3)", to_node="EphInt(5)")
        valid, _ = validate_mutation(m)
        assert valid is True

    def test_ephint_invalid_value(self):
        from layer3.mutation_validator import validate_mutation
        m = TreeMutation(MutationType.EPHEMERAL_PERTURBATION, TreeSlot.ENTRY,
                         from_node="EphInt(3)", to_node="EphInt(7)")
        valid, reason = validate_mutation(m)
        assert valid is False
        assert "not in valid set" in reason

    def test_apply_terminal_swap_to_sexpr(self):
        from layer3.mutation_validator import apply_mutation_to_sexpr
        m = TreeMutation(MutationType.TERMINAL_SWAP, TreeSlot.ENTRY,
                         from_node="BarOfDay", to_node="ATM_IV")
        original = "LT(OvernightGap, Mul(VIXChange, BarOfDay))"
        result = apply_mutation_to_sexpr(original, m)
        assert result == "LT(OvernightGap, Mul(VIXChange, ATM_IV))"

    def test_apply_does_not_partial_match(self):
        from layer3.mutation_validator import apply_mutation_to_sexpr
        m = TreeMutation(MutationType.TERMINAL_SWAP, TreeSlot.ENTRY,
                         from_node="ATM_IV", to_node="VIXSpot")
        # Should NOT match ATM_IV inside ATM_IV_5m
        original = "GT(ATM_IV_5m, ATM_IV)"
        result = apply_mutation_to_sexpr(original, m)
        # Should only replace the standalone ATM_IV, not ATM_IV inside ATM_IV_5m
        assert "VIXSpot" in result
        assert "ATM_IV_5m" in result  # _5m version untouched

    def test_wilcoxon_significant(self):
        from layer3.mutation_validator import wilcoxon_test
        # Clear improvement across all folds
        sig, p, imp = wilcoxon_test([0.5, 0.6, 0.4, 0.7], [1.5, 1.8, 1.2, 1.9])
        assert imp > 0.5
        # With only 4 samples, Wilcoxon may or may not be significant at 0.10
        # but improvement should be detected
        assert imp > 0

    def test_wilcoxon_not_significant(self):
        from layer3.mutation_validator import wilcoxon_test
        # No improvement
        sig, p, imp = wilcoxon_test([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert sig is False
        assert p == 1.0

    def test_wilcoxon_insufficient_data(self):
        from layer3.mutation_validator import wilcoxon_test
        # Only 2 samples — can't run Wilcoxon
        sig, p, imp = wilcoxon_test([1.0, 1.0], [1.5, 1.5])
        assert sig is False
        assert p == 1.0


class TestPostQCGates:
    """Post-QC gates D5-D8 (audit finding F21)."""

    def test_d5_passes_small_gap(self):
        from layer3.gates import gate_d5_proxy_qc_gap
        packet = _make_packet()
        packet.proxy_qc_sharpe_gap = 0.3
        result = gate_d5_proxy_qc_gap(packet)
        assert result.passed is True

    def test_d5_fails_large_gap(self):
        from layer3.gates import gate_d5_proxy_qc_gap
        packet = _make_packet()
        packet.proxy_qc_sharpe_gap = 0.8
        result = gate_d5_proxy_qc_gap(packet)
        assert result.passed is False

    def test_d6_passes_low_drawdown(self):
        from layer3.gates import gate_d6_qc_drawdown
        packet = _make_packet()
        packet.qc_drawdown = 0.08
        result = gate_d6_qc_drawdown(packet)
        assert result.passed is True

    def test_d6_fails_high_drawdown(self):
        from layer3.gates import gate_d6_qc_drawdown
        packet = _make_packet()
        packet.qc_drawdown = 0.25
        result = gate_d6_qc_drawdown(packet)
        assert result.passed is False

    def test_d7_passes_positive_sharpe(self):
        from layer3.gates import gate_d7_qc_test_sharpe
        packet = _make_packet()
        packet.qc_sharpe = 0.5
        result = gate_d7_qc_test_sharpe(packet)
        assert result.passed is True

    def test_d7_fails_negative_sharpe(self):
        from layer3.gates import gate_d7_qc_test_sharpe
        packet = _make_packet()
        packet.qc_sharpe = -0.3
        result = gate_d7_qc_test_sharpe(packet)
        assert result.passed is False

    def test_d8_passes_good_pf(self):
        from layer3.gates import gate_d8_profit_factor
        packet = _make_packet()
        result = gate_d8_profit_factor(packet, profit_factor=1.5)
        assert result.passed is True

    def test_d8_fails_low_pf(self):
        from layer3.gates import gate_d8_profit_factor
        packet = _make_packet()
        result = gate_d8_profit_factor(packet, profit_factor=0.9)
        assert result.passed is False


# ---------------------------------------------------------------------------
# P0-5b — L3 consumer rejects a dead-CBOE-VIX backtest
# ---------------------------------------------------------------------------

class TestVixSourceFailedGuard:
    """P0-5b: the generated QC algorithm sets a vix_source_failed runtime statistic
    when CBOE VIX never populated (VIX terminals silently fell back to ATM_IV*100).
    The L3 deploy consumer must DETECT that marker and reject the run rather than
    accept its proxy-scale Sharpe into calibration."""

    def test_detects_marker_in_runtime_statistics(self):
        from layer3.pipeline_v2 import vix_source_failed
        assert vix_source_failed({"Sharpe Ratio": "1.2"}, {"vix_source_failed": "1"})

    def test_detects_marker_merged_into_statistics(self):
        # Defensive: if QC/MCP merges custom stats into `statistics`, still caught.
        from layer3.pipeline_v2 import vix_source_failed
        assert vix_source_failed({"vix_source_failed": "1", "Sharpe Ratio": "0.9"})

    def test_key_and_value_normalization(self):
        from layer3.pipeline_v2 import vix_source_failed
        assert vix_source_failed({"VIX Source Failed": "true"})  # spaces + casing
        assert vix_source_failed({"vix_source_failed": "TRUE"})
        assert vix_source_failed({"vix_source_failed": 1})       # numeric truthy
        # Future-proof: "key present AND not falsy" -> a count value still flags.
        assert vix_source_failed({"vix_source_failed": "2"})

    def test_healthy_run_not_flagged(self):
        from layer3.pipeline_v2 import vix_source_failed
        assert not vix_source_failed({"Sharpe Ratio": "1.5", "Total Orders": "120"}, {})
        assert not vix_source_failed({"vix_source_failed": "0"})  # explicitly healthy
        assert not vix_source_failed({"vix_source_failed": ""})   # present-but-empty
        assert not vix_source_failed({"vix_source_failed": "none"})
        assert not vix_source_failed({}, {}, None)                # empty / None-safe

    def test_complete_handler_wires_the_guard(self):
        # The deploy result handler lives inside an async function; lock the wiring
        # by source so a dead-VIX run is rejected, not silently accepted. The
        # subprocess must forward runtime_stats AND the handler must consult it.
        import inspect
        import layer3.pipeline_v2 as p
        src = inspect.getsource(p)
        assert '"runtime_stats": runtime_stats' in src, (
            "deploy subprocess must forward runtime_stats into the complete event"
        )
        assert 'vix_source_failed(stats, evt.get("runtime_stats", {})' in src, (
            "the complete-event handler must reject runs flagged vix_source_failed"
        )

