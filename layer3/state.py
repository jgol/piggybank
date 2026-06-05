"""L3 pipeline state management: serialize, resume, and QC reconciliation.

The pipeline state tracks all strategies in a batch through their lifecycle:
pending → deploying → analyzing → modifying → approved/discarded/review.

State is serialized to JSON after every strategy completion so the pipeline
can resume after crashes without re-processing completed strategies.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from layer3.schemas import PipelineState, StrategyLedgerEntry


STATE_DIR = Path(__file__).parent / "state"


def save_state(state: PipelineState) -> Path:
    """Serialize PipelineState to JSON. Returns the path written."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{state.batch_id}.json"
    # Atomic write: write to temp then rename
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2))
    tmp_path.rename(path)
    return path


def load_state(batch_id: str) -> Optional[PipelineState]:
    """Load PipelineState from JSON. Returns None if not found."""
    path = STATE_DIR / f"{batch_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return PipelineState.model_validate(data)


def create_state(
    batch_id: str,
    gp_results_dir: str,
    strategy_ids: List[str],
    template_names: Dict[str, str],
) -> PipelineState:
    """Create a new PipelineState for a batch of strategies.

    Args:
        batch_id: Unique batch identifier (e.g., timestamp).
        gp_results_dir: Path to GP walk-forward results.
        strategy_ids: Ordered list of strategy IDs to process.
        template_names: Map of strategy_id → template_name.
    """
    ledger = {}
    for sid in strategy_ids:
        ledger[sid] = StrategyLedgerEntry(
            strategy_id=sid,
            template_name=template_names.get(sid, "unknown"),
        )
    return PipelineState(
        batch_id=batch_id,
        gp_results_dir=gp_results_dir,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ledger=ledger,
        queue=list(strategy_ids),
    )


def resume_state(state: PipelineState) -> PipelineState:
    """Prepare a loaded state for resumption after crash.

    - Strategies in 'completed', 'approved', 'discarded' stay as-is.
    - Strategies in 'deploying', 'analyzing', 'modifying' are re-queued
      (their in-progress work was lost in the crash).
    - QC slots are cleared (will be reconciled via list_projects).
    """
    interrupted_statuses = {"deploying", "analyzing", "modifying"}
    re_queued = []
    for sid, entry in state.ledger.items():
        if entry.status in interrupted_statuses:
            entry.status = "pending"
            re_queued.append(sid)

    # Re-queue interrupted strategies at the front of the queue
    # (they were already prioritized, don't send them to the back)
    existing_queue = [s for s in state.queue if s not in re_queued]
    state.queue = re_queued + existing_queue

    # Clear QC slots (reconciliation happens separately)
    state.qc_slot_1 = None
    state.qc_slot_2 = None

    return state


def update_strategy_status(
    state: PipelineState,
    strategy_id: str,
    status: str,
    reason: str = "",
) -> None:
    """Update a strategy's status in the ledger."""
    entry = state.ledger.get(strategy_id)
    if entry is None:
        return
    entry.status = status
    if reason:
        entry.final_reason = reason
    # Remove from queue if present
    if strategy_id in state.queue:
        state.queue.remove(strategy_id)
    # Move to appropriate completion list
    if status == "approved":
        if strategy_id not in state.approved:
            state.approved.append(strategy_id)
        if strategy_id not in state.completed:
            state.completed.append(strategy_id)
    elif status == "discarded":
        if strategy_id not in state.discarded:
            state.discarded.append(strategy_id)
        if strategy_id not in state.completed:
            state.completed.append(strategy_id)
    elif status == "review":
        if strategy_id not in state.completed:
            state.completed.append(strategy_id)


def list_state_files() -> List[str]:
    """List all batch IDs with saved state files."""
    if not STATE_DIR.exists():
        return []
    return sorted(
        p.stem for p in STATE_DIR.glob("*.json")
        if not p.name.endswith(".tmp")
    )
