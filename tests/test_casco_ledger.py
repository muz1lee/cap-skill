from pathlib import Path

import pytest

from casco.exp.ledger import append_entry, read_entries


def test_append_entry_creates_parent_and_preserves_input(tmp_path: Path):
    ledger_path = tmp_path / "nested" / "ledger.jsonl"
    entry = {"run_id": "run-001", "phase": "Batch2", "status": "KEPT"}

    written = append_entry(ledger_path, entry, now=lambda: "2026-06-19T21:30:00+00:00")

    assert entry == {"run_id": "run-001", "phase": "Batch2", "status": "KEPT"}
    assert written == {
        "run_id": "run-001",
        "phase": "Batch2",
        "status": "KEPT",
        "logged_at": "2026-06-19T21:30:00+00:00",
    }
    assert read_entries(ledger_path) == [written]


def test_append_entry_requires_core_fields(tmp_path: Path):
    with pytest.raises(ValueError, match="missing required ledger fields: phase, status"):
        append_entry(tmp_path / "ledger.jsonl", {"run_id": "run-001"})
