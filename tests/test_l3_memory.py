"""Tests for L3 persistent memory."""
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def temp_memory_dir(monkeypatch):
    """Redirect memory to a temp directory for each test."""
    import layer3.memory as mod
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(mod, "MEMORY_DIR", Path(tmpdir))
        yield Path(tmpdir)


class TestAppend:

    def test_append_creates_file(self, temp_memory_dir):
        from layer3.memory import append
        append("terminal_patterns", {"terminal": "ATM_IV", "observation": "test"})
        path = temp_memory_dir / "terminal_patterns.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["terminal"] == "ATM_IV"
        assert "timestamp" in record
        assert "confidence" in record

    def test_append_multiple(self, temp_memory_dir):
        from layer3.memory import append
        append("mutation_outcomes", {"strategy_id": "s1", "success": True})
        append("mutation_outcomes", {"strategy_id": "s2", "success": False})
        path = temp_memory_dir / "mutation_outcomes.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_append_preserves_existing_timestamp(self, temp_memory_dir):
        from layer3.memory import append
        append("terminal_patterns", {"terminal": "X", "timestamp": "2026-01-01T00:00:00Z"})
        path = temp_memory_dir / "terminal_patterns.jsonl"
        record = json.loads(path.read_text().strip())
        assert record["timestamp"] == "2026-01-01T00:00:00Z"

    def test_append_invalid_type_raises(self):
        from layer3.memory import append
        with pytest.raises(ValueError, match="Unknown memory type"):
            append("bogus_type", {"data": "x"})

    def test_append_custom_confidence(self, temp_memory_dir):
        from layer3.memory import append
        append("terminal_patterns", {"terminal": "X", "confidence": 0.9})
        record = json.loads((temp_memory_dir / "terminal_patterns.jsonl").read_text().strip())
        assert record["confidence"] == 0.9


class TestLoadEntries:

    def test_load_empty(self, temp_memory_dir):
        from layer3.memory import load_entries
        assert load_entries("terminal_patterns") == []

    def test_load_returns_entries(self, temp_memory_dir):
        from layer3.memory import append, load_entries
        for i in range(5):
            append("template_viability", {"template": f"t{i}"})
        entries = load_entries("template_viability")
        assert len(entries) == 5

    def test_load_max_entries(self, temp_memory_dir):
        from layer3.memory import append, load_entries
        for i in range(20):
            append("terminal_patterns", {"terminal": f"t{i}"})
        entries = load_entries("terminal_patterns", max_entries=5)
        assert len(entries) == 5
        # Should be the last 5
        assert entries[0]["terminal"] == "t15"


class TestLoadContext:

    def test_empty_context(self, temp_memory_dir):
        from layer3.memory import load_context
        ctx = load_context()
        assert "No persistent memory" in ctx

    def test_context_with_entries(self, temp_memory_dir):
        from layer3.memory import append, load_context
        append("terminal_patterns", {"terminal": "ATM_IV", "observation": "1.88x bias"})
        append("mutation_outcomes", {"strategy_id": "s1", "success": True})
        ctx = load_context()
        assert "Terminal Patterns" in ctx
        assert "Mutation Outcomes" in ctx
        assert "ATM_IV" in ctx

    def test_context_sorted_by_confidence(self, temp_memory_dir):
        from layer3.memory import append, load_context
        append("terminal_patterns", {"terminal": "low", "confidence": 0.1})
        append("terminal_patterns", {"terminal": "high", "confidence": 0.9})
        ctx = load_context()
        # High confidence should appear first
        high_pos = ctx.index("high")
        low_pos = ctx.index("low")
        assert high_pos < low_pos


class TestCompact:

    def test_compact_removes_low_confidence(self, temp_memory_dir):
        from layer3.memory import append, load_entries, compact
        append("terminal_patterns", {"terminal": "keep", "confidence": 0.8})
        append("terminal_patterns", {"terminal": "drop", "confidence": 0.1})
        removed = compact(min_confidence=0.2)
        assert removed["terminal_patterns"] == 1
        entries = load_entries("terminal_patterns")
        assert len(entries) == 1
        assert entries[0]["terminal"] == "keep"

    def test_compact_max_keep(self, temp_memory_dir):
        from layer3.memory import append, load_entries, compact
        for i in range(10):
            append("mutation_outcomes", {"idx": i, "confidence": 0.5})
        removed = compact(max_keep=3)
        assert removed["mutation_outcomes"] == 7
        entries = load_entries("mutation_outcomes")
        assert len(entries) == 3
        # Should keep the last 3
        assert entries[0]["idx"] == 7


class TestClear:

    def test_clear_one_type(self, temp_memory_dir):
        from layer3.memory import append, load_entries, clear
        append("terminal_patterns", {"x": 1})
        append("mutation_outcomes", {"y": 2})
        clear("terminal_patterns")
        assert load_entries("terminal_patterns") == []
        assert len(load_entries("mutation_outcomes")) == 1

    def test_clear_all(self, temp_memory_dir):
        from layer3.memory import append, load_entries, clear
        append("terminal_patterns", {"x": 1})
        append("mutation_outcomes", {"y": 2})
        append("template_viability", {"z": 3})
        clear()
        for mtype in ("terminal_patterns", "mutation_outcomes", "template_viability"):
            assert load_entries(mtype) == []
