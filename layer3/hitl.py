"""L3 Human-in-the-Loop: file-based interrupt for paper trading approval.

When the Diagnostician approves a strategy, the supervisor triggers HITL:
1. Writes a pending approval JSON with all relevant data
2. Blocks until a decision file appears (or timeout)
3. Returns the human's decision

File naming includes batch_id to prevent stale decisions from previous runs.
CLI helper: python -m layer3.hitl review
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


from layer3.config import HITL_TIMEOUT_HOURS

APPROVALS_DIR = Path(__file__).parent / "pending_approvals"
TIMEOUT_HOURS = HITL_TIMEOUT_HOURS


def request_approval(
    batch_id: str,
    strategy_id: str,
    presentation: dict,
) -> None:
    """Write a pending approval file for human review.

    Does NOT block. The supervisor should call wait_for_decision() after this.
    """
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    path = APPROVALS_DIR / f"{batch_id}_{strategy_id}.json"
    if path.exists():
        return  # idempotent: don't overwrite existing request
    path.write_text(json.dumps(presentation, indent=2, default=str))


def wait_for_decision(
    batch_id: str,
    strategy_id: str,
    timeout_hours: float = TIMEOUT_HOURS,
    poll_interval_s: float = 30.0,
) -> Optional[dict]:
    """Block until a decision file appears or timeout.

    Returns the decision dict, or None on timeout.
    Decision file: {batch_id}_{strategy_id}.decision.json
    """
    decision_path = APPROVALS_DIR / f"{batch_id}_{strategy_id}.decision.json"
    deadline = time.time() + timeout_hours * 3600

    while time.time() < deadline:
        if decision_path.exists():
            try:
                decision = json.loads(decision_path.read_text())
                return decision
            except (json.JSONDecodeError, Exception):
                return None
        time.sleep(poll_interval_s)

    return None  # timeout


def check_existing_decision(
    batch_id: str,
    strategy_id: str,
) -> Optional[dict]:
    """Check if a decision already exists (for resume idempotency).

    Returns the decision dict if found, None otherwise.
    """
    decision_path = APPROVALS_DIR / f"{batch_id}_{strategy_id}.decision.json"
    if decision_path.exists():
        try:
            return json.loads(decision_path.read_text())
        except (json.JSONDecodeError, Exception):
            return None
    return None


def check_pending_request(
    batch_id: str,
    strategy_id: str,
) -> bool:
    """Check if a pending approval request exists (for resume idempotency)."""
    path = APPROVALS_DIR / f"{batch_id}_{strategy_id}.json"
    return path.exists()


def list_pending() -> list:
    """List all pending approvals (no decision file yet)."""
    if not APPROVALS_DIR.exists():
        return []
    pending = []
    for path in APPROVALS_DIR.glob("*.json"):
        if path.name.endswith(".decision.json"):
            continue
        decision_path = path.with_name(path.stem + ".decision.json")
        if not decision_path.exists():
            try:
                data = json.loads(path.read_text())
                pending.append({
                    "file": path.name,
                    "strategy_id": data.get("strategy_id", ""),
                    "template_name": data.get("template_name", ""),
                    "qc_sharpe": data.get("qc_metrics", {}).get("sharpe", "?"),
                })
            except (json.JSONDecodeError, Exception):
                pending.append({"file": path.name, "error": "parse failed"})
    return pending


def submit_decision(
    batch_id: str,
    strategy_id: str,
    decision: str,
    reason: str = "",
) -> None:
    """Write a decision file (used by CLI helper or web UI)."""
    APPROVALS_DIR.mkdir(parents=True, exist_ok=True)
    path = APPROVALS_DIR / f"{batch_id}_{strategy_id}.decision.json"
    path.write_text(json.dumps({
        "decision": decision,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point: python -m layer3.hitl review
# ---------------------------------------------------------------------------

def _cli_review():
    """Interactive CLI for reviewing pending approvals."""
    pending = list_pending()
    if not pending:
        print("No pending approvals.")
        return

    print(f"\n{'='*60}")
    print(f"  {len(pending)} pending approval(s)")
    print(f"{'='*60}\n")

    for i, p in enumerate(pending):
        print(f"  [{i+1}] {p.get('file', '?')}")
        print(f"      Strategy: {p.get('strategy_id', '?')}")
        print(f"      Template: {p.get('template_name', '?')}")
        print(f"      QC Sharpe: {p.get('qc_sharpe', '?')}")
        print()

    for p in pending:
        fname = p.get("file", "")
        if not fname:
            continue
        # Parse batch_id and strategy_id from filename
        parts = fname.replace(".json", "").split("_", 1)
        if len(parts) < 2:
            continue
        batch_id = parts[0]
        strategy_id = parts[1]

        # Show full details
        path = APPROVALS_DIR / fname
        try:
            data = json.loads(path.read_text())
            print(f"\n{'='*60}")
            print(f"  {strategy_id}")
            print(f"{'='*60}")
            print(json.dumps(data, indent=2, default=str)[:2000])
        except Exception:
            print(f"  Could not read {fname}")
            continue

        while True:
            choice = input("\n  [a]pprove / [r]eject / [d]efer / [q]uit: ").strip().lower()
            if choice in ("a", "approve"):
                reason = input("  Reason (optional): ").strip()
                submit_decision(batch_id, strategy_id, "approve", reason)
                print(f"  → APPROVED")
                break
            elif choice in ("r", "reject"):
                reason = input("  Reason: ").strip()
                submit_decision(batch_id, strategy_id, "reject", reason)
                print(f"  → REJECTED")
                break
            elif choice in ("d", "defer"):
                print(f"  → DEFERRED (will timeout in {TIMEOUT_HOURS}h)")
                break
            elif choice in ("q", "quit"):
                return
            else:
                print("  Invalid choice. Try again.")


if __name__ == "__main__":
    _cli_review()
