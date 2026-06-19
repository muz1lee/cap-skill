from __future__ import annotations

import hashlib
from pathlib import Path

from casco.skills.candidates import candidate_records_from_run, parse_trial_dir_name


def test_parse_trial_dir_name_extracts_trial_metrics():
    assert parse_trial_dir_name("trial_12_sandboxrc_0_reward_-1.250_taskcompleted_1") == {
        "trial_index": 12,
        "sandbox_return_code": 0,
        "reward": -1.25,
        "task_completed": True,
    }


def test_candidate_records_from_run_preserve_code_and_provenance(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "gemini-3.5-flash" / "run_001"
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_0.500_taskcompleted_1"
    trial_dir.mkdir(parents=True)
    code = "\n".join(
        [
            "def reusable_pick(name):",
            "    \"\"\"Pick one object by name.\"\"\"",
            "    close_gripper()",
            "",
            "reusable_pick('bowl')",
        ]
    )
    code_path = trial_dir / "code.py"
    code_path.write_text(code, encoding="utf-8")
    (run_dir / "summaries.txt").write_text(
        "\n".join(
            [
                "Model: gemini-3.5-flash",
                "Config Path: env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
                "Git Commit: dadd2a7 (Dirty: False)",
                "Total number of trials: 1",
                "Code generation success rate / Average reward / Task completed: ",
                "1.000/0.500/1",
                "Average code blocks: 1.000",
                "Average regenerations: 0.000",
                "Average finishes: 0.000",
                "Elapsed time: 10.00 seconds",
            ]
        ),
        encoding="utf-8",
    )

    records = candidate_records_from_run(run_dir)

    assert records == [
        {
            "candidate_id": "run_001:trial_01",
            "run_id": "run_001",
            "trial_index": 1,
            "model": "gemini-3.5-flash",
            "config": "env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
            "commit": "dadd2a7",
            "git_dirty": False,
            "sandbox_return_code": 0,
            "reward": 0.5,
            "task_completed": True,
            "code_path": str(code_path),
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "code": code,
            "features": {
                "line_count": 5,
                "nonempty_line_count": 4,
                "function_count": 1,
                "imports": [],
                "call_names": ["close_gripper", "reusable_pick"],
                "uses_conditionals": False,
                "uses_loops": False,
                "has_exception_handler": False,
                "parse_error": None,
            },
            "functions": [
                {
                    "name": "reusable_pick",
                    "signature": "def reusable_pick(name)",
                    "code": "\n".join(
                        [
                            "def reusable_pick(name):",
                            "    \"\"\"Pick one object by name.\"\"\"",
                            "    close_gripper()",
                        ]
                    ),
                    "docstring": "Pick one object by name.",
                }
            ],
        }
    ]
