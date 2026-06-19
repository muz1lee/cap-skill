from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_run(run_dir: Path) -> None:
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "code.py").write_text("open_gripper()\n", encoding="utf-8")
    (run_dir / "summaries.txt").write_text(
        "\n".join(
            [
                "Model: gemini-3.5-flash",
                "Config Path: env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
                "Git Commit: dadd2a7 (Dirty: False)",
                "Total number of trials: 1",
                "Code generation success rate / Average reward / Task completed: ",
                "1.000/0.000/0",
                "Average code blocks: 1.000",
                "Average regenerations: 0.000",
                "Average finishes: 0.000",
                "Elapsed time: 10.00 seconds",
            ]
        ),
        encoding="utf-8",
    )


def test_extract_candidates_cli_writes_jsonl(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "gemini-3.5-flash" / "run_001"
    output_path = tmp_path / "candidates" / "run_001.jsonl"
    _write_run(run_dir)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.skills.extract_candidates",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["candidate_id"] == "run_001:trial_01"
    assert records[0]["code"] == "open_gripper()\n"
    assert str(output_path) in result.stdout
