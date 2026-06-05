"""Tests for L3 pipeline state management."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPipelineStateCreation:

    def test_create_state(self):
        from layer3.state import create_state
        state = create_state(
            batch_id="test_001",
            gp_results_dir="/tmp/results",
            strategy_ids=["s1", "s2", "s3"],
            template_names={"s1": "bpc", "s2": "bcc", "s3": "ic"},
        )
        assert state.batch_id == "test_001"
        assert len(state.ledger) == 3
        assert state.queue == ["s1", "s2", "s3"]
        assert state.completed == []
        assert state.approved == []
        assert state.ledger["s1"].template_name == "bpc"
        assert state.ledger["s1"].status == "pending"

    def test_create_state_empty(self):
        from layer3.state import create_state
        state = create_state("empty", "/tmp", [], {})
        assert len(state.ledger) == 0
        assert state.queue == []


class TestStateSerialization:

    def test_save_and_load(self):
        from layer3.state import create_state, save_state, load_state, STATE_DIR
        with patch.object(Path, '__new__', wraps=Path.__new__):
            import layer3.state as mod
            orig_dir = mod.STATE_DIR
            with tempfile.TemporaryDirectory() as tmpdir:
                mod.STATE_DIR = Path(tmpdir)
                try:
                    state = create_state("round_trip", "/tmp", ["a", "b"], {"a": "bpc", "b": "bcc"})
                    state.qc_slot_1 = "a"
                    state.total_cost_usd = 12.50
                    path = save_state(state)
                    assert path.exists()

                    loaded = load_state("round_trip")
                    assert loaded is not None
                    assert loaded.batch_id == "round_trip"
                    assert loaded.qc_slot_1 == "a"
                    assert loaded.total_cost_usd == 12.50
                    assert loaded.ledger["a"].template_name == "bpc"
                finally:
                    mod.STATE_DIR = orig_dir

    def test_load_nonexistent(self):
        from layer3.state import load_state
        import layer3.state as mod
        orig_dir = mod.STATE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mod.STATE_DIR = Path(tmpdir)
            try:
                assert load_state("nonexistent") is None
            finally:
                mod.STATE_DIR = orig_dir


class TestResumeState:

    def test_interrupted_strategies_requeued(self):
        from layer3.state import create_state, resume_state
        state = create_state("resume_test", "/tmp", ["a", "b", "c", "d"],
                             {"a": "bpc", "b": "bcc", "c": "ic", "d": "ib"})
        # Simulate mid-batch crash
        state.ledger["a"].status = "approved"
        state.ledger["b"].status = "deploying"  # interrupted
        state.ledger["c"].status = "analyzing"  # interrupted
        state.ledger["d"].status = "pending"
        state.completed = ["a"]
        state.approved = ["a"]
        state.queue = ["d"]  # only d was still queued
        state.qc_slot_1 = "b"
        state.qc_slot_2 = "c"

        resumed = resume_state(state)

        # b and c should be re-queued at front
        assert resumed.ledger["b"].status == "pending"
        assert resumed.ledger["c"].status == "pending"
        assert resumed.queue[:2] == ["b", "c"]
        assert "d" in resumed.queue
        # a stays approved
        assert resumed.ledger["a"].status == "approved"
        # Slots cleared
        assert resumed.qc_slot_1 is None
        assert resumed.qc_slot_2 is None

    def test_completed_strategies_not_requeued(self):
        from layer3.state import create_state, resume_state
        state = create_state("no_requeue", "/tmp", ["a", "b"],
                             {"a": "bpc", "b": "bcc"})
        state.ledger["a"].status = "discarded"
        state.ledger["b"].status = "approved"
        state.completed = ["a", "b"]
        state.queue = []

        resumed = resume_state(state)
        assert resumed.queue == []
        assert resumed.ledger["a"].status == "discarded"
        assert resumed.ledger["b"].status == "approved"


class TestUpdateStrategyStatus:

    def test_approve(self):
        from layer3.state import create_state, update_strategy_status
        state = create_state("update_test", "/tmp", ["s1"], {"s1": "bpc"})
        update_strategy_status(state, "s1", "approved", "QC Sharpe +0.6")
        assert state.ledger["s1"].status == "approved"
        assert state.ledger["s1"].final_reason == "QC Sharpe +0.6"
        assert "s1" in state.approved
        assert "s1" in state.completed

    def test_discard(self):
        from layer3.state import create_state, update_strategy_status
        state = create_state("discard_test", "/tmp", ["s1"], {"s1": "bpc"})
        update_strategy_status(state, "s1", "discarded", "compile error")
        assert "s1" in state.discarded
        assert "s1" in state.completed

    def test_review(self):
        from layer3.state import create_state, update_strategy_status
        state = create_state("review_test", "/tmp", ["s1"], {"s1": "bpc"})
        update_strategy_status(state, "s1", "review")
        assert "s1" in state.completed
        assert "s1" not in state.approved
        assert "s1" not in state.discarded

    def test_unknown_strategy_ignored(self):
        from layer3.state import create_state, update_strategy_status
        state = create_state("noop_test", "/tmp", ["s1"], {"s1": "bpc"})
        update_strategy_status(state, "nonexistent", "approved")
        assert "nonexistent" not in state.approved

    def test_no_duplicate_in_completed(self):
        from layer3.state import create_state, update_strategy_status
        state = create_state("dedup_test", "/tmp", ["s1"], {"s1": "bpc"})
        update_strategy_status(state, "s1", "approved")
        update_strategy_status(state, "s1", "approved")  # duplicate
        assert state.completed.count("s1") == 1
        assert state.approved.count("s1") == 1


class TestListStateFiles:

    def test_list_empty(self):
        from layer3.state import list_state_files
        import layer3.state as mod
        from pathlib import Path
        import tempfile
        orig = mod.STATE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mod.STATE_DIR = Path(tmpdir)
            try:
                assert list_state_files() == []
            finally:
                mod.STATE_DIR = orig

    def test_list_with_files(self):
        from layer3.state import create_state, save_state, list_state_files
        import layer3.state as mod
        from pathlib import Path
        import tempfile
        orig = mod.STATE_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mod.STATE_DIR = Path(tmpdir)
            try:
                save_state(create_state("batch_a", "/tmp", [], {}))
                save_state(create_state("batch_b", "/tmp", [], {}))
                files = list_state_files()
                assert "batch_a" in files
                assert "batch_b" in files
                assert len(files) == 2
            finally:
                mod.STATE_DIR = orig
