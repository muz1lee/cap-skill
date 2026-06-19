from __future__ import annotations

import json
from pathlib import Path

from casco.skills.reports import (
    build_candidate_report,
    candidate_archetype,
    extract_goal_from_initial_prompt,
)


def _write_prompt(run_dir: Path, goal: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = [
        {"role": "system", "content": "generate code"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are controlling a robot.\nGoal: {goal}\nAPIs:",
                }
            ],
        },
    ]
    (run_dir / "initial_prompt.txt").write_text(repr(prompt), encoding="utf-8")


def _record(
    run_dir: Path,
    *,
    candidate_id: str,
    run_id: str,
    task_completed: bool,
    reward: float,
    call_names: list[str],
) -> dict:
    trial_dir = run_dir / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    trial_dir.mkdir(parents=True, exist_ok=True)
    code_path = trial_dir / "code.py"
    code_path.write_text("pass\n", encoding="utf-8")
    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "trial_index": 1,
        "code_path": str(code_path),
        "code_sha256": f"sha-{candidate_id}",
        "reward": reward,
        "task_completed": task_completed,
        "features": {"line_count": 1, "call_names": call_names, "parse_error": None},
    }


def test_extract_goal_from_initial_prompt_reads_capx_prompt_repr(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    _write_prompt(run_dir, "Pick the bowl and place it on the plate")

    assert (
        extract_goal_from_initial_prompt(run_dir / "initial_prompt.txt")
        == "Pick the bowl and place it on the plate"
    )


def test_candidate_archetype_separates_perception_attempt_and_success(tmp_path: Path):
    run_dir = tmp_path / "run_001"

    assert (
        candidate_archetype(
            _record(
                run_dir,
                candidate_id="perception",
                run_id="run_001",
                task_completed=False,
                reward=0.0,
                call_names=["get_all_object_poses", "print"],
            )
        )
        == "perception_only"
    )
    assert (
        candidate_archetype(
            _record(
                run_dir,
                candidate_id="attempt",
                run_id="run_001",
                task_completed=False,
                reward=0.0,
                call_names=["sample_grasp_pose", "goto_pose", "open_gripper"],
            )
        )
        == "manipulation_attempt"
    )
    assert (
        candidate_archetype(
            _record(
                run_dir,
                candidate_id="success",
                run_id="run_001",
                task_completed=True,
                reward=1.0,
                call_names=["goto_pose", "close_gripper"],
            )
        )
        == "successful_manipulation"
    )


def test_build_candidate_report_adds_goal_and_counts_archetypes(tmp_path: Path):
    first_run = tmp_path / "outputs" / "gemini-3.5-flash" / "run_a"
    second_run = tmp_path / "outputs" / "gemini-3.5-flash" / "run_b"
    _write_prompt(first_run, "Pick the bowl on the cookies box")
    _write_prompt(second_run, "Pick the bowl next to the plate")
    records = [
        _record(
            first_run,
            candidate_id="run_a:trial_01",
            run_id="run_a",
            task_completed=False,
            reward=0.0,
            call_names=["get_all_object_poses", "print"],
        ),
        _record(
            second_run,
            candidate_id="run_b:trial_01",
            run_id="run_b",
            task_completed=True,
            reward=1.0,
            call_names=["sample_grasp_pose", "goto_pose", "open_gripper"],
        ),
    ]

    report = build_candidate_report(records)

    assert report["total_records"] == 2
    assert report["task_completed_count"] == 1
    assert report["archetype_counts"] == {
        "perception_only": 1,
        "successful_manipulation": 1,
    }
    assert report["candidates"][0]["goal"] == "Pick the bowl on the cookies box"
    assert report["candidates"][0]["archetype"] == "perception_only"
    assert report["candidates"][1]["goal"] == "Pick the bowl next to the plate"
    assert report["candidates"][1]["archetype"] == "successful_manipulation"


def test_build_candidate_report_is_json_serializable(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_prompt(run_dir, "Pick and place")
    report = build_candidate_report(
        [
            _record(
                run_dir,
                candidate_id="run:trial_01",
                run_id="run",
                task_completed=False,
                reward=0.0,
                call_names=[],
            )
        ]
    )

    json.dumps(report, sort_keys=True)
