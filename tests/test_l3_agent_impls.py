"""Tests for L3 agent implementations (QC Operator, Diagnostician, Mutator).

These test the agent wrappers — the tool handlers, input assembly, and
output parsing. They do NOT make actual LLM calls (that requires API keys).
"""
import json
import pytest


class TestQCOperatorAgent:

    def test_module_importable(self):
        from layer3.agent_qc_operator import interpret_error, diagnose_zero_trades
        assert callable(interpret_error)
        assert callable(diagnose_zero_trades)


class TestDiagnosticianAgent:

    def test_module_importable(self):
        from layer3.agent_diagnostician import assess_strategy, cross_check_assessment
        assert callable(assess_strategy)
        assert callable(cross_check_assessment)

    def test_cross_check_passes_accurate_claims(self):
        from layer3.agent_diagnostician import cross_check_assessment
        from layer3.schemas import StrategyAssessment, TradeRecord

        assessment = StrategyAssessment(
            action="APPROVE",
            technical_gap="Minor gap",
            gap_severity=0.3,
            behavioral_profile="Stable",
            max_consecutive_losses=3,
            tradeability="Good",
            confidence=0.8,
        )
        # gap_severity ~0.3 matches (0.85-0.62)/0.85 ≈ 0.27
        issues = cross_check_assessment(
            assessment,
            trade_log=[],
            proxy_sharpe=0.85,
            qc_sharpe=0.62,
        )
        assert len(issues) == 0, f"Unexpected issues: {issues}"

    def test_cross_check_catches_fabricated_gap_severity(self):
        from layer3.agent_diagnostician import cross_check_assessment
        from layer3.schemas import StrategyAssessment

        assessment = StrategyAssessment(
            action="APPROVE",
            technical_gap="Huge gap",
            gap_severity=0.9,  # claims 90% divergence
            behavioral_profile="Fine",
            tradeability="OK",
            confidence=0.8,
        )
        # Actual gap: (0.85-0.62)/0.85 ≈ 0.27, claimed 0.9 → diff 0.63 > 0.3
        issues = cross_check_assessment(
            assessment,
            trade_log=[],
            proxy_sharpe=0.85,
            qc_sharpe=0.62,
        )
        assert len(issues) >= 1
        assert "gap_severity" in issues[0]

    def test_cross_check_catches_wrong_loss_streak(self):
        from layer3.agent_diagnostician import cross_check_assessment
        from layer3.schemas import StrategyAssessment, TradeRecord

        assessment = StrategyAssessment(
            action="MODIFY",
            technical_gap="Gap",
            gap_severity=0.2,
            behavioral_profile="Bad streaks",
            max_consecutive_losses=15,  # claims 15
            tradeability="Risky",
            confidence=0.5,
        )
        # Actual: 3 consecutive losses (entry_credit < 0 = loss proxy)
        trades = [
            TradeRecord(reason="stop_loss", entry_credit=-10),
            TradeRecord(reason="stop_loss", entry_credit=-10),
            TradeRecord(reason="stop_loss", entry_credit=-10),
            TradeRecord(reason="signal", entry_credit=20),
        ]
        issues = cross_check_assessment(
            assessment,
            trade_log=trades,
            proxy_sharpe=0.5,
            qc_sharpe=0.3,
        )
        assert any("max_consecutive_losses" in i for i in issues)

    def test_diag_input_context_isolation(self):
        """DiagnosticianInput should not accept tree fields."""
        from layer3.schemas import DiagnosticianInput, ProxyMetrics, QCMetrics
        di = DiagnosticianInput(
            strategy_id="test",
            template_name="bpc",
            interpretation="test interp",
            proxy_metrics=ProxyMetrics(),
            qc_metrics=QCMetrics(),
        )
        assert not hasattr(di, "entry_sexpr") or "entry_sexpr" not in di.model_fields


class TestMutatorAgent:

    def test_module_importable(self):
        from layer3.agent_mutator import propose_mutation, validate_mutation
        assert callable(propose_mutation)
        assert callable(validate_mutation)

    def test_validate_valid_replacement(self):
        from layer3.agent_mutator import validate_mutation
        from layer3.schemas import MutationProposal

        proposal = MutationProposal(
            target_tree="entry",
            original_subtree="EphReal(0.18)",
            replacement_subtree="EphReal(0.34)",
            rationale="Scale threshold",
        )
        valid, error = validate_mutation(
            proposal,
            original_trees={"entry_sexpr": "GT(ATM_IV, EphReal(0.18))"},
            condition="scalar-only",
        )
        assert valid, f"Should be valid: {error}"

    def test_validate_rejects_unavailable_terminal(self):
        from layer3.agent_mutator import validate_mutation
        from layer3.schemas import MutationProposal

        proposal = MutationProposal(
            target_tree="entry",
            original_subtree="EphReal(0.18)",
            replacement_subtree="GT(PredRV15, EphReal(0.5))",  # probe not in scalar-only
            rationale="Use probe",
        )
        valid, error = validate_mutation(
            proposal,
            original_trees={"entry_sexpr": "GT(ATM_IV, EphReal(0.18))"},
            condition="scalar-only",
        )
        assert not valid
        assert "PredRV15" in error

    def test_validate_rejects_invalid_sexpr(self):
        from layer3.agent_mutator import validate_mutation
        from layer3.schemas import MutationProposal

        proposal = MutationProposal(
            target_tree="entry",
            original_subtree="EphReal(0.18)",
            replacement_subtree="NOT_VALID_SEXPR(((",
            rationale="Bad",
        )
        valid, error = validate_mutation(
            proposal,
            original_trees={},
            condition="scalar-only",
        )
        assert not valid
        assert "Invalid" in error or "s-expression" in error.lower() or "Error" in error

    def test_mutator_input_context_isolation(self):
        """MutatorInput should not accept QC fields."""
        from layer3.schemas import MutatorInput
        mi = MutatorInput(
            strategy_id="test",
            template_name="bpc",
            condition="scalar-only",
            entry_sexpr="GT(ATM_IV, EphReal(0.1))",
            exit_sexpr="LT(MinutesToClose, EphReal(-1))",
            size_sexpr="EphReal(0.5)",
        )
        assert "qc_metrics" not in mi.model_fields
        assert "trade_log" not in mi.model_fields
