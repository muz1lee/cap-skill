"""Build outcome-aware motif reports for CaSCo candidate records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from casco.skills.features import analyze_code_features
from casco.skills.inventory import read_candidate_records
from casco.skills.motifs import extract_code_motif
from casco.skills.reports import candidate_archetype

MOTIF_CONSTANT_FIELDS = (
    "has_lift_after_grasp",
    "uses_explicit_target_pose",
    "orientation_constant",
    "z_offset_bucket",
    "object_lookup_style",
)


def _motif_family_key(motif: dict[str, Any]) -> dict[str, Any]:
    return {
        "subgoal_signature": motif.get("subgoal_signature", []),
        "has_lift_after_grasp": motif.get("has_lift_after_grasp"),
        "uses_explicit_target_pose": motif.get("uses_explicit_target_pose"),
        "orientation_constant": motif.get("orientation_constant"),
        "z_offset_bucket": motif.get("z_offset_bucket"),
        "object_lookup_style": motif.get("object_lookup_style"),
    }


def _family_id(family_key: dict[str, Any]) -> str:
    canonical = json.dumps(family_key, sort_keys=True, separators=(",", ":"))
    return "f_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _record_features(record: dict[str, Any]) -> dict[str, Any]:
    features = record.get("features")
    if isinstance(features, dict):
        return features
    return analyze_code_features(str(record.get("code", "")))


def _record_for_archetype(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    enriched["features"] = _record_features(record)
    return enriched


def _parse_task_id(record: dict[str, Any]) -> int | None:
    value = record.get("task_id")
    if isinstance(value, int):
        return value
    run_id = str(record.get("run_id", ""))
    match = re.search(r"(?:task|spatial_)(\d+)", run_id)
    return int(match.group(1)) if match else None


def _parse_sample_index(record: dict[str, Any]) -> int | None:
    value = record.get("sample_index")
    if isinstance(value, int):
        return value
    run_id = str(record.get("run_id", ""))
    match = re.search(r"sample(\d+)", run_id)
    return int(match.group(1)) if match else None


def _failure_stage(archetype: str, motif: dict[str, Any], record: dict[str, Any]) -> str:
    if record.get("task_completed"):
        return "none"
    if archetype == "invalid_code":
        return "invalid_code"
    if archetype == "perception_only":
        return "no_action"
    calls = motif.get("ordered_call_skeleton", [])
    if "close_gripper" not in calls:
        return "grasp"
    if "open_gripper" not in calls:
        return "grasp"
    return "place"


def _candidate_row(record: dict[str, Any]) -> dict[str, Any]:
    code = str(record.get("code", ""))
    motif = extract_code_motif(code)
    family_key = _motif_family_key(motif)
    archetype = candidate_archetype(_record_for_archetype(record))
    return {
        "candidate_id": record.get("candidate_id"),
        "run_id": record.get("run_id"),
        "task_id": _parse_task_id(record),
        "sample_index": _parse_sample_index(record),
        "trial_index": record.get("trial_index"),
        "model": record.get("model"),
        "config": record.get("config"),
        "task_completed": bool(record.get("task_completed")),
        "reward": record.get("reward"),
        "archetype": archetype,
        "motif_id": motif["motif_id"],
        "family_id": _family_id(family_key),
        "code_sha256": record.get("code_sha256"),
        "code_path": record.get("code_path"),
        "failure_stage": _failure_stage(archetype, motif, record),
        "motif": motif,
        "family_key": family_key,
    }


def _constant_splits(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    stable: dict[str, Any] = {}
    unstable: dict[str, list[Any]] = {}
    for field in MOTIF_CONSTANT_FIELDS:
        values = sorted({row["motif"].get(field) for row in rows}, key=str)
        if len(values) == 1:
            stable[field] = values[0]
        else:
            unstable[field] = values
    return stable, unstable


def _motif_row(motif_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    archetype_counts = Counter(row["archetype"] for row in rows)
    successes = [row for row in rows if row["task_completed"]]
    failures = [row for row in rows if not row["task_completed"]]
    stable, unstable = _constant_splits(rows)
    success_count = len(successes)
    perception_count = archetype_counts.get("perception_only", 0)
    n_records = len(rows)
    unique_success_count = len({row.get("code_sha256") for row in successes if row.get("code_sha256")})
    success_rate = success_count / n_records if n_records else 0.0
    perception_rate = perception_count / n_records if n_records else 0.0
    task_ids = {row["task_id"] for row in rows if row["task_id"] is not None}
    return {
        "motif_id": motif_id,
        "n_records": n_records,
        "n_tasks": len(task_ids),
        "success_count": success_count,
        "success_rate": success_rate,
        "perception_only_count": perception_count,
        "manipulation_attempt_count": archetype_counts.get("manipulation_attempt", 0),
        "successful_manipulation_count": archetype_counts.get("successful_manipulation", 0),
        "unique_success_count": unique_success_count,
        "gate0_stable": (
            success_rate >= 0.4
            and perception_rate <= 0.2
            and unique_success_count >= 2
        ),
        "representative_successes": [str(row["candidate_id"]) for row in successes[:3]],
        "representative_failures": [str(row["candidate_id"]) for row in failures[:3]],
        "stable_constants": stable,
        "unstable_constants": unstable,
    }


def _family_row(family_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    archetype_counts = Counter(row["archetype"] for row in rows)
    successes = [row for row in rows if row["task_completed"]]
    failures = [row for row in rows if not row["task_completed"]]
    success_count = len(successes)
    perception_count = archetype_counts.get("perception_only", 0)
    n_records = len(rows)
    unique_success_count = len({row.get("code_sha256") for row in successes if row.get("code_sha256")})
    success_rate = success_count / n_records if n_records else 0.0
    perception_rate = perception_count / n_records if n_records else 0.0
    task_ids = {row["task_id"] for row in rows if row["task_id"] is not None}
    motif_ids = sorted({str(row["motif_id"]) for row in rows})
    return {
        "family_id": family_id,
        "family_key": rows[0]["family_key"],
        "n_records": n_records,
        "n_tasks": len(task_ids),
        "n_motifs": len(motif_ids),
        "motif_ids": motif_ids,
        "success_count": success_count,
        "success_rate": success_rate,
        "perception_only_count": perception_count,
        "manipulation_attempt_count": archetype_counts.get("manipulation_attempt", 0),
        "successful_manipulation_count": archetype_counts.get("successful_manipulation", 0),
        "unique_success_count": unique_success_count,
        "gate0_stable": (
            success_rate >= 0.4
            and perception_rate <= 0.2
            and unique_success_count >= 2
        ),
        "representative_successes": [str(row["candidate_id"]) for row in successes[:3]],
        "representative_failures": [str(row["candidate_id"]) for row in failures[:3]],
    }


def build_motif_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a JSON-serializable motif/outcome report for candidate records."""
    candidates = sorted(
        (_candidate_row(record) for record in records),
        key=lambda row: (str(row["run_id"]), int(row["trial_index"] or 0)),
    )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate["motif_id"]].append(candidate)
        family_groups[candidate["family_id"]].append(candidate)

    archetype_counts = Counter(candidate["archetype"] for candidate in candidates)
    motifs = sorted(
        (_motif_row(motif_id, rows) for motif_id, rows in groups.items()),
        key=lambda row: (row["motif_id"]),
    )
    families = sorted(
        (_family_row(family_id, rows) for family_id, rows in family_groups.items()),
        key=lambda row: (row["family_id"]),
    )
    return {
        "total_records": len(candidates),
        "task_completed_count": sum(1 for candidate in candidates if candidate["task_completed"]),
        "archetype_counts": dict(sorted(archetype_counts.items())),
        "motif_count": len(motifs),
        "gate0_stable_motif_count": sum(1 for motif in motifs if motif["gate0_stable"]),
        "family_count": len(families),
        "gate0_stable_family_count": sum(1 for family in families if family["gate0_stable"]),
        "candidates": candidates,
        "motifs": motifs,
        "families": families,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_motif_report(read_candidate_records(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote motif report to {args.output}")


if __name__ == "__main__":
    main()
