from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_motif_report_cli_writes_json_report(tmp_path: Path):
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "motif_report.json"
    input_path.write_text(
        json.dumps(
            {
                "candidate_id": "run_001:trial_01",
                "run_id": "run_001",
                "trial_index": 1,
                "task_completed": False,
                "reward": 0.0,
                "code_sha256": "sha-run-001",
                "code_path": "/tmp/run_001/code.py",
                "code": "poses = get_all_object_poses()\nprint(poses)\n",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.skills.motif_report",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_records"] == 1
    assert report["candidates"][0]["archetype"] == "perception_only"
    assert str(output_path) in result.stdout
