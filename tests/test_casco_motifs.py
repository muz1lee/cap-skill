from casco.skills.motifs import extract_code_motif


def test_extract_code_motif_detects_lift_target_pose_and_place_constants():
    code = "\n".join(
        [
            "import numpy as np",
            "poses = get_all_object_poses()",
            "bowl_pose = get_object_pose('akita_black_bowl')",
            "plate_pose = get_object_pose('plate')",
            "grasp_pos, grasp_quat = sample_grasp_pose('akita_black_bowl')",
            "goto_pose(grasp_pos + np.array([0, 0, 0.10]), grasp_quat)",
            "goto_pose(grasp_pos, grasp_quat)",
            "close_gripper()",
            "lift_pos = grasp_pos + np.array([0, 0, 0.15])",
            "goto_pose(lift_pos, grasp_quat)",
            "place_pos = plate_pose[:3] + np.array([0, 0, 0.12])",
            "goto_pose(place_pos, np.array([0, 0, 1, 0]))",
            "open_gripper()",
            "goto_pose(place_pos + np.array([0, 0, 0.15]), np.array([0, 0, 1, 0]))",
        ]
    )

    motif = extract_code_motif(code)

    assert motif["ordered_call_skeleton"] == [
        "get_all_object_poses",
        "get_object_pose",
        "get_object_pose",
        "sample_grasp_pose",
        "goto_pose",
        "goto_pose",
        "close_gripper",
        "goto_pose",
        "goto_pose",
        "open_gripper",
        "goto_pose",
    ]
    assert motif["subgoal_signature"] == [
        "lookup",
        "sample_grasp",
        "approach",
        "grasp",
        "lift",
        "target_pose",
        "place",
        "release",
        "retreat",
    ]
    assert motif["has_lift_after_grasp"] is True
    assert motif["uses_explicit_target_pose"] is True
    assert motif["orientation_constant"] == "down_quat"
    assert motif["z_offset_bucket"] == "high"
    assert motif["object_lookup_style"] == "explicit"
    assert motif["motif_id"].startswith("m_")
    assert motif["parse_error"] is None


def test_extract_code_motif_identifies_perception_only_loop():
    code = "\n".join(
        [
            "poses = get_all_object_poses()",
            "for name, pose in poses.items():",
            "    if 'bowl' in name:",
            "        print(name, pose)",
        ]
    )

    motif = extract_code_motif(code)

    assert motif["ordered_call_skeleton"] == ["get_all_object_poses"]
    assert motif["subgoal_signature"] == ["lookup"]
    assert motif["object_lookup_style"] == "loop_filter"
    assert motif["has_lift_after_grasp"] is False
    assert motif["z_offset_bucket"] == "none"


def test_extract_code_motif_ignores_pregrasp_open_when_finding_place_and_release():
    code = "\n".join(
        [
            "import numpy as np",
            "all_objects = get_all_object_poses()",
            "open_gripper()",
            "grasp_pos, grasp_quat = sample_grasp_pose(bowl_name)",
            "goto_pose(grasp_pos, grasp_quat)",
            "close_gripper()",
            "lift_pos = grasp_pos + np.array([0, 0, 0.15])",
            "goto_pose(lift_pos, grasp_quat)",
            "plate_pos, _ = get_object_pose(plate_name)",
            "place_pos = plate_pos + np.array([0, 0, 0.08])",
            "goto_pose(place_pos, grasp_quat)",
            "open_gripper()",
            "goto_pose(place_pos + np.array([0, 0, 0.15]), grasp_quat)",
        ]
    )

    motif = extract_code_motif(code)

    assert motif["subgoal_signature"] == [
        "lookup",
        "sample_grasp",
        "approach",
        "grasp",
        "lift",
        "target_pose",
        "place",
        "release",
        "retreat",
    ]
    assert motif["uses_explicit_target_pose"] is True


def test_extract_code_motif_returns_invalid_bucket_for_parse_errors():
    motif = extract_code_motif("def broken(:\n")

    assert motif["motif_id"] == "invalid_code"
    assert motif["parse_error"] is not None
    assert motif["ordered_call_skeleton"] == []
