"""Deterministic QC backtest scheduler with 2-slot concurrency.

Manages the deployment queue for QuantConnect backtests. The scheduling
logic is DETERMINISTIC (not LLM) — the QC Operator agent is invoked
only for error interpretation when regex patterns don't match.

QC allows at most 2 simultaneous backtests. The scheduler:
- Launches strategies into free slots
- Polls both slots in parallel (15-second intervals)
- Handles timeouts, compile errors, and API failures
- Classifies errors via regex before falling back to LLM

Architecture:
    qc_scheduler.py (this file) — deterministic scheduling
    agents.py task() — called only for novel error interpretation
    mcp_tools.py — low-level MCP operations
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from layer3.schemas import PipelineState


# ---------------------------------------------------------------------------
# Known error patterns (deterministic classification)
# ---------------------------------------------------------------------------

_COMPILE_ERROR_PATTERNS = [
    re.compile(r"Compilation error", re.IGNORECASE),
    re.compile(r"error CS\d+", re.IGNORECASE),
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"IndentationError", re.IGNORECASE),
    re.compile(r"NameError", re.IGNORECASE),
]

_RUNTIME_ERROR_PATTERNS = [
    re.compile(r"Runtime error", re.IGNORECASE),
    re.compile(r"Object reference not set", re.IGNORECASE),
    re.compile(r"NullReferenceException", re.IGNORECASE),
    re.compile(r"IndexOutOfRangeException", re.IGNORECASE),
    re.compile(r"KeyNotFoundException", re.IGNORECASE),
]

_TRANSIENT_ERROR_PATTERNS = [
    re.compile(r"Request timed out", re.IGNORECASE),
    re.compile(r"5\d\d\s", re.IGNORECASE),  # HTTP 5xx
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"Too many requests", re.IGNORECASE),
]


def classify_error(error_message: str) -> Tuple[str, bool]:
    """Classify an error message deterministically.

    Returns (error_type, retryable).
    Returns ("UNKNOWN", True) if no pattern matches — caller should
    invoke the QC Operator LLM for interpretation.
    """
    for pattern in _COMPILE_ERROR_PATTERNS:
        if pattern.search(error_message):
            return "COMPILE_ERROR", False

    for pattern in _RUNTIME_ERROR_PATTERNS:
        if pattern.search(error_message):
            return "RUNTIME_ERROR", True  # retry once

    for pattern in _TRANSIENT_ERROR_PATTERNS:
        if pattern.search(error_message):
            return "TIMEOUT", True

    return "UNKNOWN", True  # needs LLM interpretation


# ---------------------------------------------------------------------------
# Slot tracking
# ---------------------------------------------------------------------------

@dataclass
class SlotInfo:
    """Tracks one QC backtest slot."""
    strategy_id: str
    project_id: int
    backtest_id: str
    compile_id: str = ""
    launched_at: float = 0.0
    last_progress: float = 0.0
    last_progress_at: float = 0.0
    stagnant_warned: bool = False


@dataclass
class SchedulerState:
    """Deterministic scheduler state."""
    slot_1: Optional[SlotInfo] = None
    slot_2: Optional[SlotInfo] = None
    queue: List[str] = field(default_factory=list)
    results: Dict[str, dict] = field(default_factory=dict)  # strategy_id → result
    errors: Dict[str, dict] = field(default_factory=dict)   # strategy_id → error

    @property
    def has_free_slot(self) -> bool:
        return self.slot_1 is None or self.slot_2 is None

    @property
    def has_occupied_slot(self) -> bool:
        return self.slot_1 is not None or self.slot_2 is not None

    def assign_slot(self, info: SlotInfo) -> None:
        if self.slot_1 is None:
            self.slot_1 = info
        elif self.slot_2 is None:
            self.slot_2 = info
        else:
            raise RuntimeError("No free slot")

    def free_slot(self, strategy_id: str) -> None:
        if self.slot_1 and self.slot_1.strategy_id == strategy_id:
            self.slot_1 = None
        elif self.slot_2 and self.slot_2.strategy_id == strategy_id:
            self.slot_2 = None

    def occupied_slots(self) -> List[SlotInfo]:
        return [s for s in (self.slot_1, self.slot_2) if s is not None]


# ---------------------------------------------------------------------------
# Timeout and polling constants
# ---------------------------------------------------------------------------

from layer3.config import (
    QC_POLL_INTERVAL_S as POLL_INTERVAL_S,
    QC_STAGNANT_WARN_S as STAGNANT_WARN_S,
    QC_BACKTEST_TIMEOUT_S as BACKTEST_TIMEOUT_S,
    QC_COMPILE_TIMEOUT_S as COMPILE_TIMEOUT_S,
    QC_TRANSIENT_RETRY_WAIT_S as TRANSIENT_RETRY_WAIT_S,
    QC_RATE_LIMIT_PAUSE_S as RATE_LIMIT_PAUSE_S,
    QC_MAX_TRANSIENT_RETRIES as MAX_TRANSIENT_RETRIES,
)


# ---------------------------------------------------------------------------
# Scheduling loop
# ---------------------------------------------------------------------------

async def run_schedule(
    scheduler: SchedulerState,
    deploy_fn: "Callable",
    poll_fn: "Callable",
    cleanup_fn: "Callable",
    on_result: "Callable",
    on_error: "Callable",
    interpret_error_fn: "Optional[Callable]" = None,
) -> None:
    """Run the 2-slot scheduling loop until queue and slots are empty.

    This is the deterministic scheduler. It manages WHEN things happen.
    The LLM agent (via interpret_error_fn) is invoked only for novel errors
    that don't match regex patterns.

    Args:
        scheduler: SchedulerState with queue populated.
        deploy_fn: async (strategy_id) → (project_id, compile_id, backtest_id, error).
            Handles: create project, upload, compile, launch backtest.
        poll_fn: async (project_id, backtest_id) → (stats_dict, completed, progress).
        cleanup_fn: async (project_id) → None. Deletes QC project.
        on_result: callback(strategy_id, stats_dict). Called on successful completion.
        on_error: callback(strategy_id, error_type, message, retryable). Called on failure.
        interpret_error_fn: async (error_message) → (error_type, retryable).
            LLM fallback for unknown errors. If None, unknown errors are
            classified as RUNTIME_ERROR + retryable.
    """
    import asyncio
    import sys

    while scheduler.queue or scheduler.has_occupied_slot:
        # Launch into free slots
        while scheduler.has_free_slot and scheduler.queue:
            strategy_id = scheduler.queue.pop(0)
            print(f"  [scheduler] Deploying {strategy_id}...", file=sys.stderr)

            pid, cid, bt_id, error = await deploy_fn(strategy_id)
            if error:
                error_type, retryable = classify_error(error)
                if error_type == "UNKNOWN" and interpret_error_fn:
                    try:
                        error_type, retryable = await interpret_error_fn(error)
                    except Exception:
                        error_type, retryable = "RUNTIME_ERROR", False
                on_error(strategy_id, error_type, error, retryable)
                continue

            slot = SlotInfo(
                strategy_id=strategy_id,
                project_id=pid,
                backtest_id=bt_id,
                compile_id=cid or "",
                launched_at=time.time(),
                last_progress_at=time.time(),
            )
            scheduler.assign_slot(slot)

        # Poll occupied slots
        for slot in scheduler.occupied_slots():
            stats, completed, progress = await poll_fn(slot.project_id, slot.backtest_id)

            if completed and stats is not None:
                print(f"  [scheduler] {slot.strategy_id} completed", file=sys.stderr)
                scheduler.results[slot.strategy_id] = stats
                on_result(slot.strategy_id, stats)
                # Cleanup QC project
                try:
                    await cleanup_fn(slot.project_id)
                except Exception:
                    pass
                scheduler.free_slot(slot.strategy_id)
                continue

            # Track progress for stagnation detection
            now = time.time()
            if progress > slot.last_progress:
                slot.last_progress = progress
                slot.last_progress_at = now

            elapsed = now - slot.launched_at
            stagnant_time = now - slot.last_progress_at

            # Stagnation warning
            if stagnant_time > STAGNANT_WARN_S and not slot.stagnant_warned:
                print(f"  [scheduler] WARNING: {slot.strategy_id} stagnant "
                      f"for {stagnant_time:.0f}s at {progress:.0%}",
                      file=sys.stderr)
                slot.stagnant_warned = True

            # Timeout
            if elapsed > BACKTEST_TIMEOUT_S:
                print(f"  [scheduler] TIMEOUT: {slot.strategy_id} after "
                      f"{elapsed:.0f}s", file=sys.stderr)
                on_error(slot.strategy_id, "TIMEOUT",
                         f"Backtest timed out after {elapsed:.0f}s", True)
                try:
                    await cleanup_fn(slot.project_id)
                except Exception:
                    pass
                scheduler.free_slot(slot.strategy_id)

        # Wait before next poll cycle (only if slots are occupied)
        if scheduler.has_occupied_slot:
            await asyncio.sleep(POLL_INTERVAL_S)


async def reconcile_qc_projects(
    list_projects_fn: "Callable",
    delete_project_fn: "Callable",
    state: "PipelineState",
) -> int:
    """Reconcile QC projects on startup after crash.

    Lists all active QC projects, matches against pipeline ledger.
    Deletes orphaned projects (not in ledger). Returns count deleted.
    """
    projects = await list_projects_fn()
    known_pids = set()
    for entry in state.ledger.values():
        for qc_result in entry.qc_results:
            pid = qc_result.get("project_id")
            if pid:
                known_pids.add(int(pid))

    deleted = 0
    for proj in projects:
        pid = proj.get("projectId")
        if pid and int(pid) not in known_pids:
            # Check if it's an L3 project (by name prefix)
            name = proj.get("name", "")
            if name.startswith("GP_") or name.startswith("L3_"):
                try:
                    await delete_project_fn(int(pid))
                    deleted += 1
                except Exception:
                    pass
    return deleted
