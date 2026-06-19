"""Launch configuration helpers for the CaSCo Batch 1 LIBERO smoke path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

BATCH1_CONFIG_PATH = "env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml"
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_OPENAI_COMPAT_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
DEFAULT_OUTPUT_ROOT = Path("/mnt/nas/wenqian/cap-skill-data/outputs")


@dataclass(frozen=True)
class Batch1LaunchConfig:
    run_id: str
    output_root: Path = DEFAULT_OUTPUT_ROOT
    key_file: Path = Path(".geminikey")
    cuda_visible_devices: str = "0"
    config_path: str = BATCH1_CONFIG_PATH
    model: str = GEMINI_MODEL
    server_url: str = GEMINI_OPENAI_COMPAT_URL
    total_trials: int = 1
    num_workers: int = 1
    mujoco_gl: str = "osmesa"
    uv_bin: str = "uv"


def launch_output_base(config: Batch1LaunchConfig) -> Path:
    """Return the output path passed to CaP-X before it inserts the model segment."""
    return config.output_root / config.run_id


def expected_output_dir(config: Batch1LaunchConfig) -> Path:
    """Return the final directory CaP-X writes after model-name path insertion."""
    model_dir = config.model.replace("/", "_")
    return config.output_root / model_dir / config.run_id


def build_launch_command(config: Batch1LaunchConfig) -> list[str]:
    """Build the CaP-X launch argv without putting secrets into process arguments."""
    return [
        config.uv_bin,
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        config.config_path,
        "--model",
        config.model,
        "--server-url",
        config.server_url,
        "--output-dir",
        str(launch_output_base(config)),
        "--total-trials",
        str(config.total_trials),
        "--num-workers",
        str(config.num_workers),
    ]


def build_launch_env(
    config: Batch1LaunchConfig,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for a Batch 1 launch, including secret auth via env."""
    env = dict(os.environ if base_env is None else base_env)
    env["OPENAI_API_KEY"] = config.key_file.read_text(encoding="utf-8").strip()
    env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    env["MUJOCO_GL"] = config.mujoco_gl
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _redacted_env_overrides(config: Batch1LaunchConfig) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": config.cuda_visible_devices,
        "MUJOCO_GL": config.mujoco_gl,
        "OPENAI_API_KEY": "<redacted>",
        "PYTHONUNBUFFERED": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }


def dry_run_plan(config: Batch1LaunchConfig) -> dict[str, object]:
    """Return a JSON-serializable launch plan with secrets redacted."""
    return {
        "command": build_launch_command(config),
        "env_overrides": _redacted_env_overrides(config),
        "expected_output_dir": str(expected_output_dir(config)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--key-file", type=Path, default=Path(".geminikey"))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> Batch1LaunchConfig:
    return Batch1LaunchConfig(
        run_id=args.run_id,
        output_root=args.output_root,
        key_file=args.key_file,
        cuda_visible_devices=args.cuda_visible_devices,
        uv_bin=args.uv_bin,
    )


def main() -> None:
    args = _parse_args()
    config = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(dry_run_plan(config), indent=2, sort_keys=True))
        return

    completed = subprocess.run(
        build_launch_command(config),
        cwd=args.repo_dir,
        env=build_launch_env(config),
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
