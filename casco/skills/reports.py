"""Build goal-aware reports for CaSCo candidate records."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any

MANIPULATION_CALLS = {
    "close_gripper",
    "goto_pose",
    "goto_pose_interactive_cartesian",
    "open_gripper",
    "sample_grasp_pose",
}


def extract_goal_from_initial_prompt(path: str | Path) -> str | None:
    """Extract the LIBERO task goal from a CaP-X initial prompt artifact."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        return None
    try:
        messages = ast.literal_eval(prompt_path.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError):
        return None

    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        content_items = content if isinstance(content, list) else [content]
        for item in content_items:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = item
            if not isinstance(text, str):
                continue
            match = re.search(r"^Goal:\s*(.+)$", text, flags=re.MULTILINE)
            if match:
                return match.group(1).strip()
    return None


def candidate_archetype(record: dict[str, Any]) -> str:
    """Classify a candidate into a coarse report archetype."""
    features = record.get("features", {})
    if features.get("parse_error"):
        return "invalid_code"
    if record.get("task_completed"):
        return "successful_manipulation"
    call_names = set(features.get("call_names", []))
    if call_names & MANIPULATION_CALLS:
        return "manipulation_attempt"
    return "perception_only"


def _prompt_path_for_candidate(record: dict[str, Any]) -> Path | None:
    code_path = record.get("code_path")
    if not code_path:
        return None
    path = Path(code_path)
    if len(path.parents) < 2:
        return None
    return path.parents[1] / "initial_prompt.txt"


def _reported_candidate(record: dict[str, Any]) -> dict[str, Any]:
    features = record.get("features", {})
    prompt_path = _prompt_path_for_candidate(record)
    goal = extract_goal_from_initial_prompt(prompt_path) if prompt_path is not None else None
    return {
        "candidate_id": record.get("candidate_id"),
        "run_id": record.get("run_id"),
        "trial_index": record.get("trial_index"),
        "goal": goal,
        "archetype": candidate_archetype(record),
        "task_completed": bool(record.get("task_completed")),
        "reward": record.get("reward"),
        "code_path": record.get("code_path"),
        "code_sha256": record.get("code_sha256"),
        "line_count": features.get("line_count"),
        "call_names": list(features.get("call_names", [])),
    }


def build_candidate_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a JSON-serializable report for candidate records."""
    candidates = sorted(
        (_reported_candidate(record) for record in records),
        key=lambda item: (str(item["run_id"]), int(item["trial_index"] or 0)),
    )
    archetype_counts = Counter(candidate["archetype"] for candidate in candidates)
    return {
        "total_records": len(candidates),
        "task_completed_count": sum(1 for candidate in candidates if candidate["task_completed"]),
        "archetype_counts": dict(sorted(archetype_counts.items())),
        "candidates": candidates,
    }
