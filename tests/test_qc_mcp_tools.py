"""Regression tests for the QC result-retrieval bugs fixed 2026-05-31.

Two silent failures (no test existed → they shipped):
  1. poll_backtest read top-level `completed`, but QC nests it under
     ["backtest"]["completed"] → infinite "0%" poll on a finished backtest.
  2. read_orders called read_backtest_orders without pagination; the API is
     1-INDEXED (start=0 → progress stub, no orders) → every read returned [].

These use a fake async `conn` (the functions duck-type `conn.call_tool`), so no
Docker/MCP/network is needed.
"""
import asyncio
import json

from layer3.mcp_tools import poll_backtest, read_orders


class _FakeConn:
    """Returns queued JSON strings per tool; records calls for assertions."""
    def __init__(self, by_tool):
        self._by_tool = {k: list(v) for k, v in by_tool.items()}
        self.calls = []

    async def call_tool(self, tool, model):
        self.calls.append((tool, model["model"]))
        q = self._by_tool.get(tool, [])
        return q.pop(0) if q else None


def _run(coro):
    return asyncio.run(coro)


# --- poll_backtest: nested completion -------------------------------------

def test_poll_detects_nested_completion():
    resp = json.dumps({
        "success": True,
        "backtest": {"completed": True, "progress": 1.0, "status": "Completed.",
                     "statistics": {"Sharpe Ratio": "-4.016", "Net Profit": "-34.97%"}},
    })
    conn = _FakeConn({"read_backtest": [resp]})
    stats, completed, progress = _run(poll_backtest(conn, 1, "bt"))
    assert completed is True
    assert progress == 1.0
    assert stats.get("Sharpe Ratio") == "-4.016"


def test_poll_detects_completion_via_status_only():
    # completed flag absent, but status says Completed. — must still complete.
    resp = json.dumps({"success": True,
                       "backtest": {"status": "Completed.",
                                    "statistics": {"Net Profit": "1%"}}})
    stats, completed, _ = _run(poll_backtest(_FakeConn({"read_backtest": [resp]}), 1, "bt"))
    assert completed is True and stats.get("Net Profit") == "1%"


def test_poll_not_completed_returns_progress_not_stats():
    resp = json.dumps({"success": True, "backtest": {"completed": False, "progress": 0.0}})
    stats, completed, progress = _run(poll_backtest(_FakeConn({"read_backtest": [resp]}), 1, "bt"))
    assert completed is False and stats is None and progress == 0.0


def test_poll_top_level_completion_still_works():
    # Defensive: if QC ever returns it at top level, we still accept it.
    resp = json.dumps({"completed": True, "backtest": {"statistics": {"x": "1"}}})
    _, completed, _ = _run(poll_backtest(_FakeConn({"read_backtest": [resp]}), 1, "bt"))
    assert completed is True


# --- read_orders: 1-indexed pagination ------------------------------------

def test_read_orders_paginates_from_one_and_collects_all():
    page1 = json.dumps({"length": 150, "orders": [{"id": i} for i in range(1, 100)], "success": True})  # 99
    page2 = json.dumps({"length": 150, "orders": [{"id": i} for i in range(100, 151)], "success": True})  # 51
    conn = _FakeConn({"read_backtest_orders": [page1, page2]})
    orders = _run(read_orders(conn, 1, "bt"))
    assert len(orders) == 150
    # MUST start at 1 (start=0 returns a stub) and never request start=0.
    starts = [m["start"] for (_t, m) in conn.calls]
    assert starts[0] == 1
    assert 0 not in starts
    assert starts == [1, 100]  # advanced by batch size, not a fixed +100


def test_read_orders_stops_on_empty_batch():
    page1 = json.dumps({"length": 99, "orders": [{"id": i} for i in range(1, 100)], "success": True})
    empty = json.dumps({"length": 99, "orders": [], "success": True})
    conn = _FakeConn({"read_backtest_orders": [page1, empty]})
    assert len(_run(read_orders(conn, 1, "bt"))) == 99


def test_read_orders_handles_progress_stub_gracefully():
    # The start=0 bug shape: a stub with no "orders" key → treated as empty, no crash.
    stub = json.dumps({"progress": 0.0, "status": "...", "success": True})
    assert _run(read_orders(_FakeConn({"read_backtest_orders": [stub]}), 1, "bt")) == []
