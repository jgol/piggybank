"""Tests for L3 subagent runner and tool definitions."""
import json
import pytest


class TestToolDefinitions:
    """Verify tool schemas are well-formed."""

    def test_all_roles_have_tools(self):
        from layer3.tools import get_tools_for_role
        for role in ("supervisor", "qc_operator", "diagnostician", "mutator"):
            tools = get_tools_for_role(role)
            assert len(tools) > 0, f"Role {role} has no tools"

    def test_unknown_role_raises(self):
        from layer3.tools import get_tools_for_role
        with pytest.raises(ValueError, match="Unknown role"):
            get_tools_for_role("nonexistent")

    def test_all_tools_have_required_fields(self):
        from layer3.tools import get_tools_for_role
        for role in ("supervisor", "qc_operator", "diagnostician", "mutator"):
            for tool in get_tools_for_role(role):
                assert "name" in tool, f"Tool missing name in {role}"
                assert "description" in tool, f"Tool {tool['name']} missing description"
                assert "input_schema" in tool, f"Tool {tool['name']} missing input_schema"
                assert tool["input_schema"]["type"] == "object"

    def test_final_answer_tools_exist(self):
        from layer3.tools import QC_OPERATOR_TOOLS, DIAGNOSTICIAN_TOOLS, MUTATOR_TOOLS
        op_names = {t["name"] for t in QC_OPERATOR_TOOLS}
        diag_names = {t["name"] for t in DIAGNOSTICIAN_TOOLS}
        mut_names = {t["name"] for t in MUTATOR_TOOLS}
        assert "report_result" in op_names
        assert "submit_assessment" in diag_names
        assert "submit_mutation" in mut_names

    def test_diagnostician_has_no_tree_tools(self):
        """Diagnostician should not have tools to read raw tree s-expressions."""
        from layer3.tools import DIAGNOSTICIAN_TOOLS
        names = {t["name"] for t in DIAGNOSTICIAN_TOOLS}
        assert "read_grammar_spec" not in names
        assert "read_terminal_stats" not in names

    def test_mutator_has_no_qc_tools(self):
        """Mutator should not have tools to read QC metrics."""
        from layer3.tools import MUTATOR_TOOLS
        names = {t["name"] for t in MUTATOR_TOOLS}
        assert "read_qc_metrics" not in names
        assert "read_trade_log" not in names
        assert "read_proxy_metrics" not in names

    def test_tool_schemas_are_valid_json(self):
        from layer3.tools import get_tools_for_role
        for role in ("supervisor", "qc_operator", "diagnostician", "mutator"):
            tools = get_tools_for_role(role)
            # Should be JSON-serializable (required for Anthropic API)
            serialized = json.dumps(tools)
            assert len(serialized) > 0


class TestErrorClassification:
    """Verify deterministic error classification."""

    def test_compile_error(self):
        from layer3.qc_scheduler import classify_error
        error_type, retryable = classify_error("Compilation error on line 45: unexpected indent")
        assert error_type == "COMPILE_ERROR"
        assert retryable is False

    def test_csharp_compile_error(self):
        from layer3.qc_scheduler import classify_error
        error_type, _ = classify_error("error CS1002: ; expected")
        assert error_type == "COMPILE_ERROR"

    def test_runtime_error(self):
        from layer3.qc_scheduler import classify_error
        error_type, retryable = classify_error("Runtime error: Object reference not set to an instance")
        assert error_type == "RUNTIME_ERROR"
        assert retryable is True

    def test_timeout(self):
        from layer3.qc_scheduler import classify_error
        error_type, retryable = classify_error("Request timed out after 120s")
        assert error_type == "TIMEOUT"
        assert retryable is True

    def test_rate_limit(self):
        from layer3.qc_scheduler import classify_error
        error_type, _ = classify_error("Too many requests, please try again later")
        assert error_type == "TIMEOUT"

    def test_unknown_error(self):
        from layer3.qc_scheduler import classify_error
        error_type, retryable = classify_error("Something completely unexpected happened")
        assert error_type == "UNKNOWN"
        assert retryable is True


class TestSchedulerState:
    """Verify scheduler slot management."""

    def test_empty_state(self):
        from layer3.qc_scheduler import SchedulerState
        state = SchedulerState()
        assert state.has_free_slot
        assert not state.has_occupied_slot

    def test_assign_and_free_slot(self):
        from layer3.qc_scheduler import SchedulerState, SlotInfo
        state = SchedulerState()
        info = SlotInfo(strategy_id="s1", project_id=123, backtest_id="bt1")
        state.assign_slot(info)
        assert state.has_occupied_slot
        assert state.has_free_slot  # one slot still free
        assert len(state.occupied_slots()) == 1

        info2 = SlotInfo(strategy_id="s2", project_id=456, backtest_id="bt2")
        state.assign_slot(info2)
        assert not state.has_free_slot  # both occupied
        assert len(state.occupied_slots()) == 2

        state.free_slot("s1")
        assert state.has_free_slot
        assert len(state.occupied_slots()) == 1

    def test_assign_to_full_raises(self):
        from layer3.qc_scheduler import SchedulerState, SlotInfo
        state = SchedulerState()
        state.assign_slot(SlotInfo(strategy_id="s1", project_id=1, backtest_id="b1"))
        state.assign_slot(SlotInfo(strategy_id="s2", project_id=2, backtest_id="b2"))
        with pytest.raises(RuntimeError, match="No free slot"):
            state.assign_slot(SlotInfo(strategy_id="s3", project_id=3, backtest_id="b3"))

    def test_free_nonexistent_slot_is_noop(self):
        from layer3.qc_scheduler import SchedulerState
        state = SchedulerState()
        state.free_slot("nonexistent")  # should not raise


class TestCacheKey:
    """Verify cache key determinism and sensitivity."""

    def test_deterministic(self):
        from layer3.agents import _compute_cache_key
        k1 = _compute_cache_key("sys", [{"name": "t"}], [{"role": "user"}], "model")
        k2 = _compute_cache_key("sys", [{"name": "t"}], [{"role": "user"}], "model")
        assert k1 == k2

    def test_sensitive_to_system(self):
        from layer3.agents import _compute_cache_key
        k1 = _compute_cache_key("sys_a", [], [], "m")
        k2 = _compute_cache_key("sys_b", [], [], "m")
        assert k1 != k2

    def test_sensitive_to_message(self):
        from layer3.agents import _compute_cache_key
        k1 = _compute_cache_key("s", [], [{"content": "a"}], "m")
        k2 = _compute_cache_key("s", [], [{"content": "b"}], "m")
        assert k1 != k2

    def test_sensitive_to_model(self):
        from layer3.agents import _compute_cache_key
        k1 = _compute_cache_key("s", [], [], "sonnet")
        k2 = _compute_cache_key("s", [], [], "opus")
        assert k1 != k2

    def test_returns_hex_string(self):
        from layer3.agents import _compute_cache_key
        k = _compute_cache_key("s", [], [], "m")
        assert len(k) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in k)


class TestAuditLog:

    def test_audit_log_creates_file(self):
        from layer3.agents import _audit_log, AUDIT_LOG_DIR
        import tempfile
        from pathlib import Path
        import layer3.agents as mod
        orig = mod.AUDIT_LOG_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            mod.AUDIT_LOG_DIR = Path(tmpdir)
            try:
                _audit_log({"test": True, "role": "unit_test"})
                log_file = Path(tmpdir) / "agent_calls.jsonl"
                assert log_file.exists()
                import json
                record = json.loads(log_file.read_text().strip())
                assert record["test"] is True
                assert record["role"] == "unit_test"
            finally:
                mod.AUDIT_LOG_DIR = orig


class TestSchedulerLoop:
    """Test the scheduling loop with mock functions."""

    def test_empty_queue(self):
        """Empty queue should complete immediately."""
        import asyncio
        from layer3.qc_scheduler import SchedulerState, run_schedule
        scheduler = SchedulerState()
        results = []
        errors = []

        async def run():
            await run_schedule(
                scheduler,
                deploy_fn=lambda sid: (None, None, None, "should not be called"),
                poll_fn=lambda pid, bid: (None, False, 0.0),
                cleanup_fn=lambda pid: None,
                on_result=lambda sid, stats: results.append(sid),
                on_error=lambda sid, et, msg, r: errors.append(sid),
            )
        asyncio.run(run())
        assert results == []
        assert errors == []

    def test_deploy_error_skips_slot(self):
        """Deploy error should call on_error and not occupy a slot."""
        import asyncio
        from layer3.qc_scheduler import SchedulerState, run_schedule
        scheduler = SchedulerState(queue=["s1"])
        errors = []

        async def bad_deploy(sid):
            return None, None, None, "Compilation error on line 5"

        async def run():
            await run_schedule(
                scheduler,
                deploy_fn=bad_deploy,
                poll_fn=lambda pid, bid: (None, False, 0.0),
                cleanup_fn=lambda pid: None,
                on_result=lambda sid, stats: None,
                on_error=lambda sid, et, msg, r: errors.append((sid, et)),
            )
        asyncio.run(run())
        assert len(errors) == 1
        assert errors[0] == ("s1", "COMPILE_ERROR")
        assert not scheduler.has_occupied_slot

    def test_immediate_completion(self):
        """Strategy that completes on first poll."""
        import asyncio
        from layer3.qc_scheduler import SchedulerState, run_schedule
        scheduler = SchedulerState(queue=["s1"])
        results = []

        async def good_deploy(sid):
            return 100, "c1", "bt1", None

        poll_count = [0]
        async def instant_poll(pid, bid):
            poll_count[0] += 1
            return {"Sharpe Ratio": "1.5"}, True, 1.0

        async def noop_cleanup(pid):
            pass

        async def run():
            await run_schedule(
                scheduler,
                deploy_fn=good_deploy,
                poll_fn=instant_poll,
                cleanup_fn=noop_cleanup,
                on_result=lambda sid, stats: results.append((sid, stats)),
                on_error=lambda sid, et, msg, r: None,
            )
        asyncio.run(run())
        assert len(results) == 1
        assert results[0][0] == "s1"
        assert results[0][1]["Sharpe Ratio"] == "1.5"


class TestAgentImports:
    """Verify agents module imports correctly."""

    def test_task_importable(self):
        from layer3.agents import task, task_sync, FINAL_ANSWER_TOOLS
        assert callable(task)
        assert callable(task_sync)
        assert "submit_assessment" in FINAL_ANSWER_TOOLS
        assert "submit_mutation" in FINAL_ANSWER_TOOLS
        assert "report_result" in FINAL_ANSWER_TOOLS

    def test_final_answer_tools_match_tool_definitions(self):
        """Final answer tool names in agents.py must match actual tool definitions."""
        from layer3.agents import FINAL_ANSWER_TOOLS
        from layer3.tools import QC_OPERATOR_TOOLS, DIAGNOSTICIAN_TOOLS, MUTATOR_TOOLS
        all_tool_names = set()
        for tools in (QC_OPERATOR_TOOLS, DIAGNOSTICIAN_TOOLS, MUTATOR_TOOLS):
            for t in tools:
                all_tool_names.add(t["name"])
        for fa_tool in FINAL_ANSWER_TOOLS:
            if fa_tool == "submit_portfolio_assessment":
                continue  # portfolio manager tools not defined yet
            assert fa_tool in all_tool_names, f"Final answer tool {fa_tool} not in any tool list"
