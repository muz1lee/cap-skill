"""Extract CaSCo candidate records from CaP-X run artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from capx.skills.extractor import extract_functions
from casco.exp.summary import parse_summary_file

TRIAL_DIR_PATTERN = re.compile(
    r"^trial_(?P<trial_index>\d+)_sandboxrc_(?P<sandbox_return_code>\d+)_"
    r"reward_(?P<reward>-?\d+(?:\.\d+)?)_taskcompleted_(?P<task_completed>[01])$"
)


def parse_trial_dir_name(name: str) -> dict[str, Any]:
    """Parse the metrics encoded in a CaP-X trial directory name."""
    match = TRIAL_DIR_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid CaP-X trial directory name: {name}")
    return {
        "trial_index": int(match.group("trial_index")),
        "sandbox_return_code": int(match.group("sandbox_return_code")),
        "reward": float(match.group("reward")),
        "task_completed": match.group("task_completed") == "1",
    }


def candidate_records_from_run(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return trial-level candidate records from one CaP-X run directory."""
    path = Path(run_dir)
    summary = parse_summary_file(path / "summaries.txt")
    records: list[dict[str, Any]] = []

    for trial_dir in sorted(path.glob("trial_*")):
        if not trial_dir.is_dir():
            continue
        trial_metrics = parse_trial_dir_name(trial_dir.name)
        code_path = trial_dir / "code.py"
        if not code_path.exists():
            continue
        code = code_path.read_text(encoding="utf-8")
        trial_index = trial_metrics["trial_index"]
        records.append(
            {
                "candidate_id": f"{path.name}:trial_{trial_index:02d}",
                "run_id": path.name,
                "trial_index": trial_index,
                "model": summary["model"],
                "config": summary["config_path"],
                "commit": summary["git_commit"],
                "git_dirty": summary["git_dirty"],
                "sandbox_return_code": trial_metrics["sandbox_return_code"],
                "reward": trial_metrics["reward"],
                "task_completed": trial_metrics["task_completed"],
                "code_path": str(code_path),
                "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                "code": code,
                "functions": extract_functions(code),
            }
        )

    return records


def write_candidate_records_jsonl(
    output_path: str | Path,
    records: Iterable[dict[str, Any]],
) -> int:
    """Write candidate records to JSONL and return the number of records written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
