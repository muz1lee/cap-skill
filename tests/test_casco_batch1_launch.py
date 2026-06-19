from pathlib import Path
import json
import subprocess
import sys

from casco.exp.batch1_launch import (
    Batch1LaunchConfig,
    build_launch_command,
    build_launch_env,
    expected_output_dir,
)


def test_batch1_launch_command_uses_official_gemini_endpoint_without_key_in_argv(tmp_path: Path):
    key_file = tmp_path / ".geminikey"
    key_file.write_text("SECRET_KEY\n", encoding="utf-8")
    config = Batch1LaunchConfig(
        run_id="casco_batch1_smoke_20260619_210257",
        output_root=Path("/mnt/nas/wenqian/cap-skill-data/outputs"),
        key_file=key_file,
        cuda_visible_devices="2",
    )

    command = build_launch_command(config)
    env = build_launch_env(config, base_env={"PATH": "/usr/bin"})

    assert command == [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        "env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml",
        "--model",
        "gemini-3.5-flash",
        "--server-url",
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "--output-dir",
        "/mnt/nas/wenqian/cap-skill-data/outputs/casco_batch1_smoke_20260619_210257",
        "--total-trials",
        "1",
        "--num-workers",
        "1",
    ]
    assert "--api-key" not in command
    assert "SECRET_KEY" not in " ".join(command)
    assert env["OPENAI_API_KEY"] == "SECRET_KEY"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["MUJOCO_GL"] == "osmesa"
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_expected_output_dir_matches_capx_model_path_insertion():
    config = Batch1LaunchConfig(
        run_id="run_001",
        output_root=Path("/mnt/nas/wenqian/cap-skill-data/outputs"),
    )

    assert expected_output_dir(config) == Path(
        "/mnt/nas/wenqian/cap-skill-data/outputs/gemini-3.5-flash/run_001"
    )


def test_batch1_launch_command_can_pin_uv_binary_for_tmux_path_stability():
    config = Batch1LaunchConfig(
        run_id="run_001",
        output_root=Path("/mnt/nas/wenqian/cap-skill-data/outputs"),
        uv_bin="/root/.local/bin/uv",
    )

    assert build_launch_command(config)[0] == "/root/.local/bin/uv"


def test_batch1_launch_cli_dry_run_prints_redacted_plan(tmp_path: Path):
    key_file = tmp_path / ".geminikey"
    key_file.write_text("SECRET_KEY\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "casco.exp.batch1_launch",
            "--run-id",
            "run_001",
            "--output-root",
            str(tmp_path / "outputs"),
            "--key-file",
            str(key_file),
            "--cuda-visible-devices",
            "3",
            "--uv-bin",
            "/root/.local/bin/uv",
            "--dry-run",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )

    plan = json.loads(result.stdout)
    assert plan["command"][0:5] == [
        "/root/.local/bin/uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
    ]
    assert plan["expected_output_dir"] == str(
        tmp_path / "outputs" / "gemini-3.5-flash" / "run_001"
    )
    assert plan["env_overrides"] == {
        "CUDA_VISIBLE_DEVICES": "3",
        "MUJOCO_GL": "osmesa",
        "OPENAI_API_KEY": "<redacted>",
        "PYTHONUNBUFFERED": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }
    assert "SECRET_KEY" not in result.stdout
    assert "SECRET_KEY" not in result.stderr
