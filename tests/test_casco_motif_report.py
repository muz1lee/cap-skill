from __future__ import annotations

from casco.skills.motif_report import build_motif_report


SUCCESS_CODE = "\n".join(
    [
        "import numpy as np",
        "bowl_pose = get_object_pose('akita_black_bowl')",
        "plate_pose = get_object_pose('plate')",
        "grasp_pos, grasp_quat = sample_grasp_pose('akita_black_bowl')",
        "goto_pose(grasp_pos, grasp_quat)",
        "close_gripper()",
        "lift_pos = grasp_pos + np.array([0, 0, 0.15])",
        "goto_pose(lift_pos, grasp_quat)",
        "place_pos = plate_pose[:3] + np.array([0, 0, 0.12])",
        "goto_pose(place_pos, np.array([0, 0, 1, 0]))",
        "open_gripper()",
    ]
)

SUCCESS_CODE_WITH_PREOPEN = "\n".join(
    [
        "import numpy as np",
        "all_objects = get_all_object_poses()",
        "open_gripper()",
        "grasp_pos, grasp_quat = sample_grasp_pose('akita_black_bowl')",
        "goto_pose(grasp_pos, grasp_quat)",
        "close_gripper()",
        "lift_pos = grasp_pos + np.array([0, 0, 0.15])",
        "goto_pose(lift_pos, grasp_quat)",
        "plate_pos, _ = get_object_pose('plate')",
        "place_pos = plate_pos + np.array([0, 0, 0.12])",
        "goto_pose(place_pos, np.array([0, 0, 1, 0]))",
        "open_gripper()",
    ]
)

PERCEPTION_CODE = "\n".join(
    [
        "poses = get_all_object_poses()",
        "for name, pose in poses.items():",
        "    print(name)",
    ]
)


def _record(candidate_id: str, *, code: str, task_completed: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "run_id": candidate_id.split(":", 1)[0],
        "trial_index": 1,
        "task_completed": task_completed,
        "reward": 1.0 if task_completed else 0.0,
        "code_sha256": f"sha-{candidate_id}",
        "code_path": f"/tmp/{candidate_id.replace(':', '_')}/code.py",
        "code": code,
    }


def test_build_motif_report_adds_candidate_rows_and_outcome_counts():
    records = [
        _record("run_success:trial_01", code=SUCCESS_CODE, task_completed=True),
        _record("run_fail:trial_01", code=SUCCESS_CODE, task_completed=False),
        _record("run_perception:trial_01", code=PERCEPTION_CODE, task_completed=False),
    ]

    report = build_motif_report(records)

    assert report["total_records"] == 3
    assert report["task_completed_count"] == 1
    assert report["archetype_counts"] == {
        "manipulation_attempt": 1,
        "perception_only": 1,
        "successful_manipulation": 1,
    }
    rows_by_id = {row["candidate_id"]: row for row in report["candidates"]}
    assert rows_by_id["run_success:trial_01"]["failure_stage"] == "none"
    assert rows_by_id["run_fail:trial_01"]["failure_stage"] == "place"
    assert rows_by_id["run_perception:trial_01"]["failure_stage"] == "no_action"

    manipulation_motifs = [
        motif for motif in report["motifs"] if motif["success_count"] == 1
    ]
    assert len(manipulation_motifs) == 1
    motif = manipulation_motifs[0]
    assert motif["n_records"] == 2
    assert motif["success_rate"] == 0.5
    assert motif["perception_only_count"] == 0
    assert motif["manipulation_attempt_count"] == 1
    assert motif["unique_success_count"] == 1
    assert motif["gate0_stable"] is False
    assert motif["representative_successes"] == ["run_success:trial_01"]
    assert motif["representative_failures"] == ["run_fail:trial_01"]


def test_build_motif_report_groups_coarse_families_across_skeleton_variants():
    records = [
        _record("run_success_a:trial_01", code=SUCCESS_CODE, task_completed=True),
        _record("run_success_b:trial_01", code=SUCCESS_CODE_WITH_PREOPEN, task_completed=True),
    ]

    report = build_motif_report(records)

    assert report["motif_count"] == 2
    assert report["family_count"] == 1
    assert report["gate0_stable_family_count"] == 1
    family = report["families"][0]
    assert family["n_records"] == 2
    assert family["success_count"] == 2
    assert family["success_rate"] == 1.0
    assert family["unique_success_count"] == 2
    assert family["gate0_stable"] is True
    assert family["family_key"]["subgoal_signature"] == [
        "lookup",
        "sample_grasp",
        "approach",
        "grasp",
        "lift",
        "target_pose",
        "place",
        "release",
    ]
    assert {candidate["family_id"] for candidate in report["candidates"]} == {family["family_id"]}
