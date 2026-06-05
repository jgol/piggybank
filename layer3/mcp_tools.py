"""QC MCP connection and tool wrappers for the L3 pipeline.

Migrated from qc_mcp/utils/mcp_connection.py. The QCMCPConnection class
manages the Docker-based MCP server; the wrapper functions provide
higher-level operations used by the QC Operator agent and scheduler.

Environment variables required:
  QUANTCONNECT_USER_ID — from your QC account
  QUANTCONNECT_API_TOKEN — from your QC account
  DOCKER_PLATFORM (optional) — e.g., "linux/amd64" on Apple Silicon

Docker must be running. The MCP server is `quantconnect/mcp-server`.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Re-export QCMCPConnection from qc_mcp (single source of truth)
# ---------------------------------------------------------------------------
# Rather than copy the class, import from qc_mcp. This avoids code
# duplication while keeping the L3 import path clean.
# If qc_mcp is eventually removed, copy the class here.

import sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Lazy import: QCMCPConnection requires the `mcp` package which is only
# in qc_mcp/bin/python venv. Import fails in system Python, but the
# high-level wrapper functions are still importable for type checking.
# Actual MCP usage requires running under qc_mcp venv.
QCMCPConnection = None
ToolInfo = None

def _ensure_mcp():
    """Import QCMCPConnection on first use. Raises if mcp not installed."""
    global QCMCPConnection, ToolInfo
    if QCMCPConnection is None:
        try:
            from qc_mcp.utils.mcp_connection import QCMCPConnection as _QC, ToolInfo as _TI
            QCMCPConnection = _QC
            ToolInfo = _TI
        except ImportError as e:
            raise RuntimeError(
                "MCP package not available. Run under qc_mcp venv: "
                "PYTHONPATH=qc_mcp qc_mcp/bin/python ..."
            ) from e
    return QCMCPConnection


# ---------------------------------------------------------------------------
# High-level tool wrappers for the QC Operator
# ---------------------------------------------------------------------------

async def deploy_and_compile(
    conn,  # QCMCPConnection (lazy imported)
    project_name: str,
    code: str,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Create project, upload code, compile. Returns (project_id, compile_id, error).

    On success: (project_id, compile_id, None)
    On compile failure: (project_id, None, error_message)
    On creation failure: (None, None, error_message)
    """
    try:
        # Create project
        raw = await conn.call_tool("create_project",
                                   {"model": {"name": project_name, "language": "Py"}})
        data = json.loads(raw) if raw else {}
        projects = data.get("projects", [])
        if not projects:
            return None, None, f"create_project returned no projects: {raw[:200]}"
        pid = projects[0].get("projectId")
        if not pid:
            return None, None, f"No projectId in response: {raw[:200]}"

        # Upload code (read-before-write for QC collaboration lock)
        await conn.call_tool("read_file",
                             {"model": {"projectId": pid, "name": "main.py"}})
        await conn.call_tool("update_file_contents",
                             {"model": {"projectId": pid, "name": "main.py",
                                        "content": code}})

        # Compile
        cr = await conn.call_tool("create_compile",
                                  {"model": {"projectId": pid}})
        cr_data = json.loads(cr) if cr else {}
        compile_id = cr_data.get("compileId", "")

        # Poll compilation (up to 5 minutes)
        for _ in range(30):
            await asyncio.sleep(10)
            try:
                raw = await conn.call_tool("read_compile",
                                          {"model": {"projectId": pid,
                                                     "compileId": compile_id}})
                if not raw:
                    continue
                state_data = json.loads(raw)
            except (json.JSONDecodeError, Exception):
                continue
            state = state_data.get("state", "")
            if state == "BuildSuccess":
                return pid, compile_id, None
            if state == "BuildError":
                errors = state_data.get("errors", [])
                return pid, None, f"Compile failed: {errors[:5]}"

        return pid, None, "Compile timed out (5 min)"

    except Exception as e:
        return None, None, f"deploy_and_compile error: {type(e).__name__}: {e}"


async def launch_backtest(
    conn,  # QCMCPConnection (lazy imported)
    project_id: int,
    compile_id: str,
    backtest_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Launch a backtest. Returns (backtest_id, error)."""
    try:
        raw = await conn.call_tool("create_backtest",
                                   {"model": {"projectId": project_id,
                                              "compileId": compile_id,
                                              "backtestName": backtest_name}})
        bt = json.loads(raw) if raw else {}
        bt_id = (bt.get("backtestId")
                 or bt.get("backtest", {}).get("backtestId", ""))
        if not bt_id:
            return None, f"No backtestId in response: {raw[:200]}"
        return bt_id, None
    except Exception as e:
        return None, f"launch_backtest error: {type(e).__name__}: {e}"


async def poll_backtest(
    conn,  # QCMCPConnection (lazy imported)
    project_id: int,
    backtest_id: str,
) -> Tuple[Optional[dict], bool, float]:
    """Poll a backtest. Returns (stats_dict, completed, progress).

    stats_dict is None if not completed or error.
    """
    try:
        raw = await conn.call_tool("read_backtest",
                                   {"model": {"projectId": project_id,
                                              "backtestId": backtest_id}})
        if not raw:
            return None, False, 0.0
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return None, False, 0.0

    # QC NESTS completion/progress/status under ["backtest"] — the top-level
    # keys are normally ABSENT, so a top-level data.get("completed") poll loops
    # forever at 0% even after the backtest finishes (verified 2026-05-31; cost a
    # wrongly-killed run). Check the nested object (and the status string) too.
    bt = data.get("backtest", {}) if isinstance(data, dict) else {}
    completed = bool(data.get("completed", False) or bt.get("completed", False))
    if not completed:
        completed = str(bt.get("status", "")).strip().lower().startswith("completed")
    progress = float(data.get("progress", 0.0) or bt.get("progress", 0.0) or 0.0)
    if not completed:
        return None, False, progress

    # Extract stats (also nested under ["backtest"]).
    stats = (bt.get("statistics", {})
             or data.get("result", {}).get("Statistics", {})
             or data.get("Statistics", {}))
    return stats, True, 1.0


async def read_orders(
    conn,  # QCMCPConnection (lazy imported)
    project_id: int,
    backtest_id: str,
) -> List[dict]:
    """Read ALL backtest orders for trade-level analysis.

    QC's `read_backtest_orders` is 1-INDEXED and paginated: `start=0` returns a
    bare progress stub with NO `orders` key (looks like zero orders — the old
    bug); `start=1` returns `{length, orders:[~99], success}`. Paginate from
    start=1, <=100 per call, advancing by the batch size until `length` is
    reached (verified 2026-05-31). Without this, every orders read returned [].
    """
    orders: List[dict] = []
    start = 1
    length: Optional[int] = None
    for _ in range(100000):  # safety cap vs runaway pagination
        try:
            raw = await conn.call_tool("read_backtest_orders",
                                       {"model": {"projectId": project_id,
                                                  "backtestId": backtest_id,
                                                  "start": start,
                                                  "end": start + 99}})
            data = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, Exception):
            break
        if isinstance(data, list):  # legacy/unexpected shape
            return data
        if not isinstance(data, dict):
            break
        length = data.get("length", length)
        batch = data.get("orders", []) or []
        if not batch:
            break
        orders.extend(batch)
        start += len(batch)
        if length is not None and len(orders) >= length:
            break
    return orders


async def delete_project(conn: QCMCPConnection, project_id: int) -> bool:
    """Delete a QC project. Returns True on success."""
    try:
        await conn.call_tool("delete_project",
                             {"model": {"projectId": project_id}})
        return True
    except Exception:
        return False


async def list_projects(conn: QCMCPConnection) -> List[dict]:
    """List all QC projects (for crash recovery reconciliation)."""
    try:
        raw = await conn.call_tool("list_projects", {"model": {}})
        data = json.loads(raw) if raw else {}
        return data.get("projects", [])
    except (json.JSONDecodeError, Exception):
        return []


def load_env() -> None:
    """Load environment variables from qc_mcp/.env if not already set."""
    env_path = os.path.join(_project_root, "qc_mcp", ".env")
    if os.path.exists(env_path) and not os.getenv("QUANTCONNECT_USER_ID"):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
