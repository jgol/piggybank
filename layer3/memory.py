"""L3 persistent memory: JSONL append-only storage for cross-batch learning.

The supervisor's persistent memory accumulates patterns across GP batches:
- terminal_patterns: which terminals transfer well/poorly to QC
- mutation_outcomes: which mutation types succeed/fail
- template_viability: which templates produce viable strategies per fold

Memory is append-only. Each entry includes a confidence score (0-1) and
timestamp. Load context formats the most recent entries weighted by
confidence × recency for injection into the supervisor's system prompt.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


MEMORY_DIR = Path(__file__).parent / "memories"

MEMORY_TYPES = ("terminal_patterns", "mutation_outcomes", "template_viability")


def append(memory_type: str, record: dict) -> None:
    """Append one record to a memory JSONL file.

    Adds timestamp if not present. Flushes immediately.
    """
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unknown memory type: {memory_type}. Must be one of {MEMORY_TYPES}")
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if "timestamp" not in record:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if "confidence" not in record:
        record["confidence"] = 0.5
    path = MEMORY_DIR / f"{memory_type}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_entries(memory_type: str, max_entries: int = 200) -> List[dict]:
    """Load the most recent entries from a memory file.

    Returns up to max_entries, most recent last.
    """
    if memory_type not in MEMORY_TYPES:
        raise ValueError(f"Unknown memory type: {memory_type}")
    path = MEMORY_DIR / f"{memory_type}.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries[-max_entries:]


def load_context(max_per_type: int = 50) -> str:
    """Load all memory types and format as a text block for prompt injection.

    Returns a formatted string with the most recent entries per type,
    weighted by confidence × recency (higher confidence entries shown first
    within each type).
    """
    sections = []
    for mtype in MEMORY_TYPES:
        entries = load_entries(mtype, max_entries=max_per_type)
        if not entries:
            continue
        # Sort by confidence descending (most confident first)
        entries.sort(key=lambda e: e.get("confidence", 0.5), reverse=True)
        header = mtype.replace("_", " ").title()
        lines = [f"## {header} ({len(entries)} entries)"]
        for e in entries[:max_per_type]:
            # Format entry as one-line summary
            conf = e.get("confidence", 0.5)
            ts = e.get("timestamp", "")[:10]  # date only
            # Remove metadata fields, show the rest
            content = {k: v for k, v in e.items()
                       if k not in ("timestamp", "confidence", "batch_id")}
            lines.append(f"  [{conf:.1f}] {ts}: {json.dumps(content)}")
        sections.append("\n".join(lines))
    if not sections:
        return "No persistent memory entries yet."
    return "\n\n".join(sections)


def compact(max_keep: int = 200, min_confidence: float = 0.2) -> Dict[str, int]:
    """Compact memory files: keep most recent max_keep entries per type,
    remove entries with confidence < min_confidence.

    Returns dict of {memory_type: n_removed}.
    """
    removed = {}
    for mtype in MEMORY_TYPES:
        entries = load_entries(mtype, max_entries=100000)  # load all
        before = len(entries)
        # Filter by confidence
        entries = [e for e in entries if e.get("confidence", 0.5) >= min_confidence]
        # Keep most recent
        entries = entries[-max_keep:]
        after = len(entries)
        removed[mtype] = before - after
        if removed[mtype] > 0:
            path = MEMORY_DIR / f"{mtype}.jsonl"
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
            tmp_path.rename(path)
    return removed


def clear(memory_type: Optional[str] = None) -> None:
    """Clear memory files. If memory_type is None, clear all types."""
    types = [memory_type] if memory_type else list(MEMORY_TYPES)
    for mtype in types:
        if mtype not in MEMORY_TYPES:
            raise ValueError(f"Unknown memory type: {mtype}")
        path = MEMORY_DIR / f"{mtype}.jsonl"
        if path.exists():
            path.unlink()
