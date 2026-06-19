"""Build ledger-ready metadata from CaP-X run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from casco.exp.summary import parse_summary_file


def build_ledger_entry_from_run(
    run_dir: str | Path,
    *,
    phase: str,
    status: str,
    stage: str | None = None,
    note: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Create a ledger entry from one CaP-X output directory."""
    path = Path(run_dir)
    summary = parse_summary_file(path / "summaries.txt")
    code_paths = sorted(path.glob("trial_*/code.py"))
    if not code_paths:
        raise ValueError(f"run directory has no trial code artifact: {path}")

    entry: dict[str, Any] = {
        "run_id": path.name,
        "phase": phase,
        "status": status,
        "model": summary["model"],
        "config": summary["config_path"],
        "commit": summary["git_commit"],
        "git_dirty": summary["git_dirty"],
        "output_dir": str(path),
        "code_path": str(code_paths[0]),
        "result": {
            "code_generation_success_rate": summary["code_generation_success_rate"],
            "reward": summary["average_reward"],
            "task_completed": summary["task_completed_count"] > 0,
            "task_completed_count": summary["task_completed_count"],
            "total_trials": summary["total_trials"],
            "elapsed_time_seconds": summary["elapsed_time_seconds"],
        },
    }
    if stage is not None:
        entry["stage"] = stage
    if note is not None:
        entry["note"] = note
    if exit_code is not None:
        entry["result"]["exit_code"] = exit_code
    return entry
