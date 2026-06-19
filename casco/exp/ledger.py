"""Append-only JSONL ledger helpers for CaSCo runs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("run_id", "phase", "status")


def _default_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_entry(
    ledger_path: str | Path,
    entry: Mapping[str, Any],
    *,
    now: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Append one JSON object to a ledger file and return the written entry."""
    missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
    if missing:
        raise ValueError(f"missing required ledger fields: {', '.join(missing)}")

    written = dict(entry)
    written.setdefault("logged_at", (now or _default_now)())

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(written, ensure_ascii=False, sort_keys=True) + "\n")
    return written


def read_entries(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL ledger file. Missing files are treated as an empty ledger."""
    path = Path(ledger_path)
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on ledger line {line_number}: {path}") from exc
    return entries
