from __future__ import annotations

import json
from pathlib import Path

from casco.skills.inventory import read_candidate_records, summarize_candidate_records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def test_read_candidate_records_accepts_file_or_directory(tmp_path: Path):
    first = {
        "candidate_id": "run_a:trial_01",
        "code_sha256": "aaa",
        "task_completed": True,
        "functions": [{"name": "pick"}],
        "features": {"imports": ["numpy"], "call_names": ["goto_pose"]},
    }
    second = {
        "candidate_id": "run_b:trial_01",
        "code_sha256": "bbb",
        "task_completed": False,
        "functions": [],
        "features": {"imports": [], "call_names": ["open_gripper", "goto_pose"]},
    }
    _write_jsonl(tmp_path / "candidates" / "b.jsonl", [second])
    _write_jsonl(tmp_path / "candidates" / "a.jsonl", [first])

    assert read_candidate_records(tmp_path / "candidates" / "a.jsonl") == [first]
    assert read_candidate_records(tmp_path / "candidates") == [first, second]


def test_summarize_candidate_records_counts_core_traits():
    records = [
        {
            "candidate_id": "run_a:trial_01",
            "code_sha256": "same",
            "task_completed": True,
            "functions": [{"name": "pick"}],
            "features": {"imports": ["numpy"], "call_names": ["goto_pose", "pick"]},
        },
        {
            "candidate_id": "run_b:trial_01",
            "code_sha256": "same",
            "task_completed": False,
            "functions": [],
            "features": {"imports": ["numpy"], "call_names": ["goto_pose"]},
        },
    ]

    assert summarize_candidate_records(records) == {
        "total_records": 2,
        "unique_code_count": 1,
        "task_completed_count": 1,
        "records_with_functions": 1,
        "function_name_counts": {"pick": 1},
        "import_counts": {"numpy": 2},
        "call_name_counts": {"goto_pose": 2, "pick": 1},
    }
