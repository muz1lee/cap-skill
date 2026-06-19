"""Inventory helpers for CaSCo candidate JSONL files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_candidate_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on candidate line {line_number}: {path}") from exc
    return records


def read_candidate_records(path: str | Path) -> list[dict[str, Any]]:
    """Read candidates from a JSONL file or all JSONL files in a directory."""
    root = Path(path)
    if root.is_dir():
        records: list[dict[str, Any]] = []
        for jsonl_path in sorted(root.glob("*.jsonl")):
            records.extend(_read_candidate_jsonl(jsonl_path))
        return records
    return _read_candidate_jsonl(root)


def summarize_candidate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate records for quick candidate pool inspection."""
    function_names: Counter[str] = Counter()
    imports: Counter[str] = Counter()
    call_names: Counter[str] = Counter()
    code_hashes: set[str] = set()
    task_completed_count = 0
    records_with_functions = 0

    for record in records:
        code_hash = record.get("code_sha256")
        if code_hash:
            code_hashes.add(code_hash)
        if record.get("task_completed"):
            task_completed_count += 1

        functions = record.get("functions", [])
        if functions:
            records_with_functions += 1
        for function in functions:
            name = function.get("name")
            if name:
                function_names[name] += 1

        features = record.get("features", {})
        imports.update(features.get("imports", []))
        call_names.update(features.get("call_names", []))

    return {
        "total_records": len(records),
        "unique_code_count": len(code_hashes),
        "task_completed_count": task_completed_count,
        "records_with_functions": records_with_functions,
        "function_name_counts": dict(sorted(function_names.items())),
        "import_counts": dict(sorted(imports.items())),
        "call_name_counts": dict(sorted(call_names.items())),
    }
