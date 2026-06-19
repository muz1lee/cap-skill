from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_candidates_cli_writes_goal_aware_report(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "gemini-3.5-flash" / "run_001"
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial_dir.mkdir(parents=True)
    code_path = trial_dir / "code.py"
    code_path.write_text("open_gripper()\ngoto_pose(pos, quat)\n", encoding="utf-8")
    (run_dir / "initial_prompt.txt").write_text(
        repr(
            [
                {"role": "system", "content": "generate code"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Intro\nGoal: Pick the bowl next to the plate\nAPIs:",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "records.jsonl").write_text(
        json.dumps(
            {
                "candidate_id": "run_001:trial_01",
                "run_id": "run_001",
                "trial_index": 1,
                "code_path": str(code_path),
                "code_sha256": "abc",
                "reward": 1.0,
                "task_completed": True,
                "features": {
                    "line_count": 2,
                    "call_names": ["open_gripper", "goto_pose"],
                    "parse_error": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "candidate_report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.skills.report_candidates",
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

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["total_records"] == 1
    assert report["task_completed_count"] == 1
    assert report["archetype_counts"] == {"successful_manipulation": 1}
    assert report["candidates"][0]["goal"] == "Pick the bowl next to the plate"
    assert report["candidates"][0]["archetype"] == "successful_manipulation"
    assert str(output_path) in result.stdout
