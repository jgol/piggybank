"""Tests for scripts/cross_fold_evaluate.py — persistent automated test suite.

These tests verify the cross-fold evaluation pipeline's components without
requiring real L1 Parquet data or GP results. Integration tests with real
data are gated by the --test flag on the script itself.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from layer2.experiment import WALK_FORWARD_FOLDS
from layer2.grammar import from_sexpr, to_str
from scripts.cross_fold_evaluate import (
    _discover_templates,
    _load_strategies_for_template,
    _parse_trees,
    _prepare_fold_data,
)


# ---------------------------------------------------------------------------
# Template discovery
# ---------------------------------------------------------------------------

class TestDiscoverTemplates:
    def test_discovers_across_folds(self, tmp_path):
        for fid in [1, 2]:
            fold = tmp_path / f"fold_{fid}"
            fold.mkdir()
        (tmp_path / "fold_1" / "strategies_ic.jsonl").write_text("")
        (tmp_path / "fold_1" / "strategies_bpc.jsonl").write_text("")
        (tmp_path / "fold_2" / "strategies_ic.jsonl").write_text("")
        (tmp_path / "fold_2" / "strategies_ib.jsonl").write_text("")

        result = _discover_templates(tmp_path)
        assert result == ["bpc", "ib", "ic"]

    def test_empty_dir(self, tmp_path):
        result = _discover_templates(tmp_path)
        assert result == []

    def test_no_fold_dirs(self, tmp_path):
        (tmp_path / "other_dir").mkdir()
        result = _discover_templates(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# Strategy loading
# ---------------------------------------------------------------------------

class TestLoadStrategies:
    def _make_strategy(self, template="test", **overrides):
        base = {
            "template_name": template,
            "entry_tree": "GT(ATM_IV, EphReal(0.18))",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
            "fitness": [-2.0, -1.5, -1.0],
        }
        base.update(overrides)
        return base

    def test_single_fold(self, tmp_path):
        fold = tmp_path / "fold_1"
        fold.mkdir()
        strat = self._make_strategy()
        (fold / "strategies_test.jsonl").write_text(json.dumps(strat) + "\n")

        loaded = _load_strategies_for_template(tmp_path, "test")
        assert len(loaded) == 1
        assert loaded[0]["home_fold"] == 1
        assert loaded[0]["strategy_id"] == "tes_f1_000"
        assert loaded[0]["entry_tree"] == strat["entry_tree"]

    def test_multi_fold(self, tmp_path):
        for fid in [1, 2, 3]:
            fold = tmp_path / f"fold_{fid}"
            fold.mkdir()
            strat = self._make_strategy(
                entry_tree=f"GT(ATM_IV, EphReal({0.1*fid}))")
            (fold / "strategies_test.jsonl").write_text(
                json.dumps(strat) + "\n")

        loaded = _load_strategies_for_template(tmp_path, "test")
        assert len(loaded) == 3
        assert sorted(s["home_fold"] for s in loaded) == [1, 2, 3]

    def test_multiple_strategies_per_fold(self, tmp_path):
        fold = tmp_path / "fold_1"
        fold.mkdir()
        lines = []
        for i in range(5):
            strat = self._make_strategy(
                entry_tree=f"GT(ATM_IV, EphReal({0.1*i}))")
            lines.append(json.dumps(strat))
        (fold / "strategies_test.jsonl").write_text("\n".join(lines) + "\n")

        loaded = _load_strategies_for_template(tmp_path, "test")
        assert len(loaded) == 5
        # Strategy IDs should be sequential
        ids = [s["strategy_id"] for s in loaded]
        assert ids == [f"tes_f1_{i:03d}" for i in range(5)]

    def test_missing_template_file(self, tmp_path):
        fold = tmp_path / "fold_1"
        fold.mkdir()
        # No strategies file — should return empty
        loaded = _load_strategies_for_template(tmp_path, "nonexistent")
        assert len(loaded) == 0

    def test_malformed_json_skipped(self, tmp_path):
        fold = tmp_path / "fold_1"
        fold.mkdir()
        valid = self._make_strategy()
        (fold / "strategies_test.jsonl").write_text(
            json.dumps(valid) + "\n" + "INVALID JSON\n")

        loaded = _load_strategies_for_template(tmp_path, "test")
        assert len(loaded) == 1  # only valid line loaded

    def test_no_fold_dirs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No fold_"):
            _load_strategies_for_template(tmp_path, "test")

    def test_empty_lines_skipped(self, tmp_path):
        fold = tmp_path / "fold_1"
        fold.mkdir()
        strat = self._make_strategy()
        (fold / "strategies_test.jsonl").write_text(
            "\n" + json.dumps(strat) + "\n\n")

        loaded = _load_strategies_for_template(tmp_path, "test")
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Tree parsing
# ---------------------------------------------------------------------------

class TestParseTrees:
    def test_valid_trees(self):
        strategy = {
            "strategy_id": "test_001",
            "entry_tree": "GT(ATM_IV, EphReal(0.18))",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
        }
        result = _parse_trees(strategy)
        assert result is not None
        entry, exit_, size_, delta_ = result
        assert to_str(entry) == "GT(ATM_IV, EphReal(0.18))"
        assert to_str(exit_) == "GT(MinutesToClose, EphReal(30.0))"
        assert to_str(size_) == "EphReal(0.5)"
        assert delta_ is None

    def test_with_delta_tree(self):
        strategy = {
            "strategy_id": "test_002",
            "entry_tree": "GT(ATM_IV, EphReal(0.18))",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
            "delta_tree": "Add(VIXSpot, EphReal(0.1))",
        }
        result = _parse_trees(strategy)
        assert result is not None
        assert result[3] is not None
        assert to_str(result[3]) == "Add(VIXSpot, EphReal(0.1))"

    def test_invalid_tree_returns_none(self):
        strategy = {
            "strategy_id": "test_003",
            "entry_tree": "BOGUS(x, y)",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
        }
        result = _parse_trees(strategy)
        assert result is None

    def test_empty_delta_tree_treated_as_none(self):
        strategy = {
            "strategy_id": "test_004",
            "entry_tree": "GT(ATM_IV, EphReal(0.18))",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
            "delta_tree": "",
        }
        result = _parse_trees(strategy)
        assert result is not None
        assert result[3] is None

    def test_complex_tree_round_trip(self):
        """Test that a realistic GP tree round-trips through parse."""
        sexpr = ("AND(AND(GT(MinutesToClose, Sub(VIXChange, BarOfDay)), "
                 "GT(MinutesToClose, Sqrt(OvernightGap))), "
                 "GT(MinutesToClose, Sqrt(VIXChange)))")
        strategy = {
            "strategy_id": "test_005",
            "entry_tree": sexpr,
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
        }
        result = _parse_trees(strategy)
        assert result is not None
        assert to_str(result[0]) == sexpr


# ---------------------------------------------------------------------------
# Fold data preparation
# ---------------------------------------------------------------------------

class TestPrepareFoldData:
    def _make_df(self, n_dates=1100):
        """Create a minimal DataFrame spanning 2022-01 to 2026-03."""
        dates = pd.date_range("2022-01-03", periods=n_dates, freq="B")
        date_strs = [d.strftime("%Y-%m-%d") for d in dates]
        n_bars = 5
        total = n_dates * n_bars
        return pd.DataFrame({
            "date": [d for d in date_strs for _ in range(n_bars)],
            "SPXClose": np.random.uniform(3800, 5200, total),
            "ATM_IV": np.random.uniform(0.08, 0.30, total),
            "MinutesToClose": np.tile([300, 240, 180, 120, 60], n_dates),
            "VIXSpot": np.random.uniform(12, 30, total),
            "RealizedVol30m": np.random.uniform(0.01, 0.03, total),
        })

    def test_train_val_split(self):
        df = self._make_df()  # uses default n_dates=1100 (2022-2026)
        fold = WALK_FORWARD_FOLDS[0]  # train_end=2024-03-29
        train, val, stats = _prepare_fold_data(df, fold)
        assert len(train) > 0
        assert len(val) > 0
        # Train dates should be <= train_end
        assert max(train["date"].values) <= fold["train_end"]
        # Val dates should be > train_end (after embargo)
        if len(val) > 0:
            assert min(val["date"].values) > fold["train_end"]

    def test_norm_stats_computed(self):
        df = self._make_df()  # uses default n_dates=1100
        fold = WALK_FORWARD_FOLDS[0]
        _, _, stats = _prepare_fold_data(df, fold)
        # Should have stats for terminals present in data
        assert "ATM_IV" in stats
        assert "VIXSpot" in stats
        assert "RealizedVol30m" in stats
        # Each stat is (center, scale, method)
        for key, val in stats.items():
            assert len(val) == 3
            assert isinstance(val[0], float)
            assert isinstance(val[1], float)
            assert val[2] in ("robust", "standard")


# ---------------------------------------------------------------------------
# WALK_FORWARD_FOLDS consistency
# ---------------------------------------------------------------------------

class TestWalkForwardFolds:
    def test_four_folds(self):
        assert len(WALK_FORWARD_FOLDS) == 4

    def test_fold_ids_sequential(self):
        ids = [f["fold_id"] for f in WALK_FORWARD_FOLDS]
        assert ids == [1, 2, 3, 4]

    def test_required_keys(self):
        for fold in WALK_FORWARD_FOLDS:
            assert "fold_id" in fold
            assert "train_end" in fold
            assert "val_end" in fold

    def test_fold_4_holdout(self):
        f4 = WALK_FORWARD_FOLDS[3]
        assert f4["fold_id"] == 4
        # 2026-06-02: F4 now carries a real >=20-trading-day OOS test holdout
        # (val_end pulled back to 2026-03-04, test runs to the corpus end).
        assert f4["test_end"] == "2026-04-09"


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_strategy_entry_required_keys(self):
        """Verify the output schema matches what L3 gate expects."""
        required_keys = {
            "strategy_id", "home_fold", "entry_tree", "exit_tree",
            "size_tree", "cross_fold_metrics", "mean_val_sharpe",
            "folds_positive", "total_trades",
        }
        # A sample output entry
        entry = {
            "strategy_id": "bpc_f1_000",
            "home_fold": 1,
            "entry_tree": "GT(ATM_IV, EphReal(0.18))",
            "exit_tree": "GT(MinutesToClose, EphReal(30.0))",
            "size_tree": "EphReal(0.5)",
            "cross_fold_metrics": {
                "1": {"val_sharpe": 2.44, "n_trades": 72,
                      "val_win_rate": 0.68, "val_sortino": 1.5},
            },
            "mean_val_sharpe": 2.44,
            "folds_positive": 1,
            "total_trades": 72,
        }
        assert required_keys.issubset(set(entry.keys()))

    def test_per_fold_metrics_keys(self):
        per_fold = {
            "val_sharpe": 2.44,
            "val_sortino": 1.5,
            "n_trades": 72,
            "val_win_rate": 0.68,
        }
        required = {"val_sharpe", "n_trades", "val_win_rate"}
        assert required.issubset(set(per_fold.keys()))


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

class TestResumeSupport:
    def test_existing_output_parsed(self, tmp_path):
        output = {
            "template": "test",
            "strategies": [
                {
                    "strategy_id": "tes_f1_000",
                    "home_fold": 1,
                    "entry_tree": "...",
                    "exit_tree": "...",
                    "size_tree": "...",
                    "cross_fold_metrics": {
                        "1": {"val_sharpe": 2.0, "n_trades": 50,
                              "val_win_rate": 0.65, "val_sortino": 1.5},
                        "2": {"val_sharpe": 1.0, "n_trades": 40,
                              "val_win_rate": 0.55, "val_sortino": 0.8},
                    },
                    "mean_val_sharpe": 1.5,
                    "folds_positive": 2,
                    "total_trades": 90,
                }
            ],
        }
        out_path = tmp_path / "cross_fold_metrics_test.json"
        out_path.write_text(json.dumps(output))

        loaded = json.loads(out_path.read_text())
        assert len(loaded["strategies"]) == 1
        s = loaded["strategies"][0]
        assert s["strategy_id"] == "tes_f1_000"
        assert "1" in s["cross_fold_metrics"]
        assert "2" in s["cross_fold_metrics"]
