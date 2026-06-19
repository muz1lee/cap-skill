from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summarize_candidates_cli_writes_summary_json(tmp_path: Path):
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "run_001:trial_01",
                "code_sha256": "abc",
                "task_completed": False,
                "functions": [],
                "features": {"imports": ["numpy"], "call_names": ["goto_pose"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "candidate_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.skills.summarize_candidates",
            "--input",
            str(candidates_dir),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "total_records": 1,
        "unique_code_count": 1,
        "task_completed_count": 0,
        "records_with_functions": 0,
        "function_name_counts": {},
        "import_counts": {"numpy": 1},
        "call_name_counts": {"goto_pose": 1},
    }
    assert str(output_path) in result.stdout
