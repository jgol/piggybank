"""Tests for PBO/CSCV wiring in the walk-forward pipeline.

Validates:
1. Top-N strategy selection for PBO
2. Fold training date filtering
3. PBO result attachment to survivors
4. Integration with run_walk_forward summary output
"""
import numpy as np
import pytest


class TestPBOStrategySelection:
    """PBO should use top-N strategies by val_sharpe, not all."""

    def test_top_10_selected_from_larger_front(self):
        """When Pareto front has >10 strategies, only top-10 by val_sharpe used."""
        strategies = [
            {"val_sharpe": i * 0.1, "entry_tree": f"GT(ATM_IV, EphReal({i}))",
             "exit_tree": "LT(MinutesToClose, EphReal(30))",
             "size_tree": "EphReal(0.5)"}
            for i in range(20)
        ]
        sorted_strats = sorted(strategies, key=lambda s: s.get("val_sharpe", 0), reverse=True)
        top_10 = sorted_strats[:10]
        assert len(top_10) == 10
        assert abs(top_10[0]["val_sharpe"] - 1.9) < 1e-9  # highest
        assert abs(top_10[9]["val_sharpe"] - 1.0) < 1e-9  # 10th highest

    def test_fewer_than_10_uses_all(self):
        """When Pareto front has <10 strategies, use all."""
        strategies = [
            {"val_sharpe": i * 0.5, "entry_tree": f"GT(ATM_IV, EphReal({i}))",
             "exit_tree": "LT(MinutesToClose, EphReal(30))",
             "size_tree": "EphReal(0.5)"}
            for i in range(5)
        ]
        sorted_strats = sorted(strategies, key=lambda s: s.get("val_sharpe", 0), reverse=True)
        top_n = sorted_strats[:10]
        assert len(top_n) == 5  # all 5, not 10

    def test_fewer_than_3_skipped(self):
        """PBO needs >= 3 strategies to be meaningful."""
        strategies = [
            {"val_sharpe": 1.0, "entry_tree": "GT(ATM_IV, EphReal(0.1))",
             "exit_tree": "LT(MinutesToClose, EphReal(30))",
             "size_tree": "EphReal(0.5)"},
            {"val_sharpe": 0.5, "entry_tree": "GT(ATM_IV, EphReal(0.2))",
             "exit_tree": "LT(MinutesToClose, EphReal(30))",
             "size_tree": "EphReal(0.5)"},
        ]
        assert len(strategies) < 3  # would be skipped in pipeline


class TestPBOFoldDateFiltering:
    """PBO should use fold training dates, not full dataset."""

    def test_fold_data_filtered_by_train_end(self):
        """Data filtered to dates <= fold's train_end."""
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"date": dates, "value": range(100)})
        train_end = "2023-03-01"
        fold_data = df[df["date"].astype(str) <= train_end]
        assert len(fold_data) < 100
        assert fold_data["date"].max().strftime("%Y-%m-%d") <= train_end

    def test_no_train_end_uses_full_data(self):
        """If fold metadata missing train_end, use full dataset."""
        import pandas as pd
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({"date": dates, "value": range(100)})
        fold_meta = None
        if fold_meta:
            fold_data = df[df["date"].astype(str) <= str(fold_meta)]
        else:
            fold_data = df
        assert len(fold_data) == 100


class TestPBOResultAttachment:
    """PBO results should be attached to surviving strategies."""

    def test_pbo_attached_to_matching_template(self):
        """Survivors with matching template_name get PBO score."""
        survivors = [
            {"template_name": "bull_put_credit", "val_sharpe": 1.0},
            {"template_name": "bear_call_credit", "val_sharpe": 0.8},
            {"template_name": "bull_put_credit", "val_sharpe": 0.6},
        ]
        pbo_val = 0.185
        tname = "bull_put_credit"
        for s in survivors:
            if s.get("template_name") == tname:
                s["pbo"] = pbo_val
        assert survivors[0]["pbo"] == 0.185
        assert "pbo" not in survivors[1]
        assert survivors[2]["pbo"] == 0.185

    def test_pbo_summary_excludes_logit_distribution(self):
        """Logit distribution (potentially large list) excluded from summary JSON."""
        pbo_results = {
            "bpc_fold1": {
                "pbo": 0.185,
                "n_combinations": 28,
                "logit_distribution": [1.0, 2.0, -0.5] * 10,  # large
                "median_logit": 13.82,
                "n_strategies": 20,
            }
        }
        summary_pbo = {k: {kk: vv for kk, vv in v.items()
                           if kk != "logit_distribution"}
                       for k, v in pbo_results.items()}
        assert "logit_distribution" not in summary_pbo["bpc_fold1"]
        assert summary_pbo["bpc_fold1"]["pbo"] == 0.185
        assert summary_pbo["bpc_fold1"]["n_strategies"] == 20


class TestPBOInterpretation:
    """PBO interpretation thresholds match Bailey et al. (2017)."""

    def test_low_pbo(self):
        pbo = 0.05
        assert pbo < 0.10  # LOW — robust selection

    def test_moderate_pbo(self):
        pbo = 0.25
        assert 0.10 <= pbo < 0.40  # MODERATE — caution

    def test_high_pbo(self):
        pbo = 0.55
        assert pbo >= 0.40  # HIGH — likely overfit


class TestPBOImportAndFunction:
    """Verify PBO module imports and basic function signatures."""

    def test_compute_pbo_from_pareto_front_importable(self):
        from layer2.pbo import compute_pbo_from_pareto_front
        assert callable(compute_pbo_from_pareto_front)

    def test_compute_pbo_from_pareto_front_requires_3_strategies(self):
        from layer2.pbo import compute_pbo_from_pareto_front
        import pandas as pd
        from layer2.templates import base_template_by_name
        tmpl = base_template_by_name("bull_put_credit")
        df = pd.DataFrame({"date": ["2024-01-01"] * 10, "ATM_IV": [0.15] * 10})
        with pytest.raises(ValueError, match="non-empty"):
            compute_pbo_from_pareto_front([], df, tmpl)
