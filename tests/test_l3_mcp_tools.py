"""Tests for L3 MCP tools (import and signature verification).

Actual MCP calls require the qc_mcp venv with the `mcp` package.
These tests verify the module is importable and functions have
correct signatures without requiring the MCP connection.
"""
import inspect
import pytest


class TestMCPToolsImport:

    def test_module_importable(self):
        import layer3.mcp_tools as mod
        assert mod is not None

    def test_wrapper_functions_exist(self):
        from layer3.mcp_tools import (
            deploy_and_compile, launch_backtest, poll_backtest,
            read_orders, delete_project, list_projects, load_env,
        )
        for fn in (deploy_and_compile, launch_backtest, poll_backtest,
                    read_orders, delete_project, list_projects):
            assert inspect.iscoroutinefunction(fn), f"{fn.__name__} should be async"
        assert callable(load_env)

    def test_ensure_mcp_raises_without_package(self):
        from layer3.mcp_tools import _ensure_mcp
        # In system Python without mcp package, should raise RuntimeError
        try:
            _ensure_mcp()
            # If it succeeds, mcp IS installed — that's fine too
        except RuntimeError as e:
            assert "MCP package not available" in str(e)

    def test_deploy_and_compile_signature(self):
        from layer3.mcp_tools import deploy_and_compile
        sig = inspect.signature(deploy_and_compile)
        params = list(sig.parameters.keys())
        assert "conn" in params
        assert "project_name" in params
        assert "code" in params

    def test_poll_backtest_signature(self):
        from layer3.mcp_tools import poll_backtest
        sig = inspect.signature(poll_backtest)
        params = list(sig.parameters.keys())
        assert "conn" in params
        assert "project_id" in params
        assert "backtest_id" in params

    def test_load_env_does_not_crash(self):
        """load_env should not crash even if .env doesn't exist."""
        from layer3.mcp_tools import load_env
        load_env()  # should be a no-op if env vars already set
