"""Tests for L3 supervisor, HITL, and portfolio manager."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestSupervisorImports:

    def test_process_strategy_importable(self):
        from layer3.supervisor import process_strategy, run_batch
        assert callable(process_strategy)
        assert callable(run_batch)

    def test_constants_defined(self):
        from layer3.supervisor import (
            MAX_MUTATION_ITERATIONS, COST_BUDGET_USD,
            APPROVE_MIN_QC_SHARPE, APPROVE_MIN_QC_TRADES, APPROVE_MAX_QC_DRAWDOWN,
        )
        assert MAX_MUTATION_ITERATIONS == 2
        assert COST_BUDGET_USD == 50.0
        assert APPROVE_MIN_QC_SHARPE == 0.0


class TestHITL:

    @pytest.fixture(autouse=True)
    def temp_approvals_dir(self, monkeypatch):
        import layer3.hitl as mod
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(mod, "APPROVALS_DIR", Path(tmpdir))
            yield Path(tmpdir)

    def test_request_creates_file(self, temp_approvals_dir):
        from layer3.hitl import request_approval
        request_approval("batch1", "strat1", {"strategy_id": "strat1", "sharpe": 0.5})
        path = temp_approvals_dir / "batch1_strat1.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["strategy_id"] == "strat1"

    def test_request_idempotent(self, temp_approvals_dir):
        from layer3.hitl import request_approval
        request_approval("batch1", "strat1", {"v": 1})
        request_approval("batch1", "strat1", {"v": 2})  # should not overwrite
        data = json.loads((temp_approvals_dir / "batch1_strat1.json").read_text())
        assert data["v"] == 1  # first write preserved

    def test_check_existing_decision(self, temp_approvals_dir):
        from layer3.hitl import submit_decision, check_existing_decision
        assert check_existing_decision("b1", "s1") is None
        submit_decision("b1", "s1", "approve", "looks good")
        dec = check_existing_decision("b1", "s1")
        assert dec is not None
        assert dec["decision"] == "approve"

    def test_check_pending_request(self, temp_approvals_dir):
        from layer3.hitl import request_approval, check_pending_request
        assert not check_pending_request("b1", "s1")
        request_approval("b1", "s1", {"data": True})
        assert check_pending_request("b1", "s1")

    def test_list_pending(self, temp_approvals_dir):
        from layer3.hitl import request_approval, submit_decision, list_pending
        request_approval("b1", "s1", {"strategy_id": "s1", "template_name": "bpc"})
        request_approval("b1", "s2", {"strategy_id": "s2", "template_name": "bcc"})
        submit_decision("b1", "s2", "reject")
        pending = list_pending()
        assert len(pending) == 1
        assert pending[0]["strategy_id"] == "s1"

    def test_batch_id_prevents_stale_decisions(self, temp_approvals_dir):
        from layer3.hitl import submit_decision, check_existing_decision
        submit_decision("old_batch", "s1", "approve")
        # New batch should not see old batch's decision
        assert check_existing_decision("new_batch", "s1") is None


class TestPortfolioManager:

    def test_correlation_matrix(self):
        from layer3.portfolio import compute_correlation_matrix
        returns = {
            "s1": np.array([0.01, -0.01, 0.02, -0.02, 0.01]),
            "s2": np.array([0.01, -0.01, 0.02, -0.02, 0.01]),  # identical
            "s3": np.array([-0.01, 0.01, -0.02, 0.02, -0.01]),  # anti-correlated
        }
        corr = compute_correlation_matrix(returns)
        assert abs(corr["s1"]["s2"] - 1.0) < 0.01
        assert abs(corr["s1"]["s3"] - (-1.0)) < 0.01

    def test_correlation_single_strategy(self):
        from layer3.portfolio import compute_correlation_matrix
        corr = compute_correlation_matrix({"s1": np.array([0.01, -0.01])})
        assert corr["s1"]["s1"] == 1.0

    def test_terminal_concentration(self):
        from layer3.portfolio import compute_terminal_concentration
        terminals = {
            "s1": ["ATM_IV", "SessionReturn"],
            "s2": ["ATM_IV", "VIXSpot"],
            "s3": ["ATM_IV", "MinutesToClose"],
        }
        conc = compute_terminal_concentration(terminals)
        assert conc["ATM_IV"] == 3

    def test_directional_balance(self):
        from layer3.portfolio import compute_directional_balance
        templates = {"s1": "bull_put_credit", "s2": "bear_call_credit", "s3": "iron_condor"}
        balance = compute_directional_balance(templates)
        assert "balanced" in balance

    def test_portfolio_drawdown_simulation(self):
        from layer3.portfolio import simulate_portfolio_drawdown
        rng = np.random.RandomState(42)
        returns = {
            "s1": rng.normal(0.001, 0.01, 250),
            "s2": rng.normal(0.001, 0.01, 250),
        }
        sim = simulate_portfolio_drawdown(returns)
        assert "sharpe" in sim
        assert "max_drawdown" in sim
        assert "calmar" in sim
        assert sim["sharpe"] > 0  # positive drift

    def test_build_portfolio_assessment(self):
        from layer3.portfolio import build_portfolio_assessment
        rng = np.random.RandomState(42)
        returns = {"s1": rng.normal(0.001, 0.01, 100), "s2": rng.normal(0.001, 0.01, 100)}
        terminals = {"s1": ["ATM_IV"], "s2": ["ATM_IV"]}
        templates = {"s1": "bull_put_credit", "s2": "bear_call_credit"}
        pa = build_portfolio_assessment(returns, terminals, templates)
        assert pa.n_strategies == 2
        assert pa.portfolio_sharpe > 0
        assert any("ATM_IV" in w for w in pa.diversification_warnings)

    def test_single_strategy_degenerate(self):
        from layer3.portfolio import build_portfolio_assessment
        returns = {"s1": np.array([0.01, -0.01, 0.02])}
        pa = build_portfolio_assessment(returns, {"s1": ["ATM_IV"]}, {"s1": "bpc"})
        assert pa.n_strategies == 1
        assert any("Single-strategy" in w for w in pa.diversification_warnings)

    def test_empty_portfolio(self):
        from layer3.portfolio import build_portfolio_assessment
        pa = build_portfolio_assessment({}, {}, {})
        assert pa.n_strategies == 0
        assert pa.proceed_recommendation is False
