"""Sampling plan construction for CaSCo multi-candidate LIBERO probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from casco.exp.batch1_launch import (
    DEFAULT_OUTPUT_ROOT,
    GEMINI_MODEL,
    Batch1LaunchConfig,
    dry_run_plan,
    expected_output_dir,
)

DEFAULT_DATA_ROOT = Path("/mnt/nas/wenqian/cap-skill-data")
DEFAULT_TEMPLATE_PATH = Path("env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml")


@dataclass(frozen=True)
class SamplingPlanConfig:
    suite_name: str
    task_ids: tuple[int, ...]
    samples_per_task: int
    stage: str
    run_group_id: str
    data_root: Path = DEFAULT_DATA_ROOT
    template_path: Path = DEFAULT_TEMPLATE_PATH
    key_file: Path = Path(".geminikey")
    cuda_visible_devices: str = "0"
    model: str = GEMINI_MODEL
    uv_bin: str = "uv"
    mujoco_gl: str = "osmesa"


@dataclass(frozen=True)
class SamplingRunSpec:
    task_id: int
    sample_index: int
    stage: str
    suite_name: str
    run_id: str
    template_path: Path
    config_path: Path
    candidate_jsonl: Path
    expected_output_dir: Path
    launch_config: Batch1LaunchConfig

    def redacted_dict(self) -> dict[str, Any]:
        plan = dry_run_plan(self.launch_config)
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "stage": self.stage,
            "suite_name": self.suite_name,
            "run_id": self.run_id,
            "template_path": str(self.template_path),
            "config_path": str(self.config_path),
            "candidate_jsonl": str(self.candidate_jsonl),
            "expected_output_dir": str(self.expected_output_dir),
            "command": plan["command"],
            "env_overrides": plan["env_overrides"],
        }


def _run_id(config: SamplingPlanConfig, task_id: int, sample_index: int) -> str:
    return (
        f"{config.stage}_{config.suite_name}_task{task_id}_"
        f"sample{sample_index:02d}_{config.run_group_id}"
    )


def build_sampling_plan(config: SamplingPlanConfig) -> list[SamplingRunSpec]:
    """Return deterministic per-sample run specs for a Batch 2.5 probe."""
    specs: list[SamplingRunSpec] = []
    output_root = config.data_root / "outputs"
    if output_root == DEFAULT_OUTPUT_ROOT:
        output_root = DEFAULT_OUTPUT_ROOT
    for task_id in config.task_ids:
        for sample_index in range(config.samples_per_task):
            run_id = _run_id(config, task_id, sample_index)
            launch_config = Batch1LaunchConfig(
                run_id=run_id,
                output_root=output_root,
                key_file=config.key_file,
                cuda_visible_devices=config.cuda_visible_devices,
                config_path=str(config.data_root / "configs" / f"{run_id}.yaml"),
                model=config.model,
                uv_bin=config.uv_bin,
                mujoco_gl=config.mujoco_gl,
            )
            specs.append(
                SamplingRunSpec(
                    task_id=task_id,
                    sample_index=sample_index,
                    stage=config.stage,
                    suite_name=config.suite_name,
                    run_id=run_id,
                    template_path=config.template_path,
                    config_path=Path(launch_config.config_path),
                    candidate_jsonl=config.data_root / "candidates" / f"{run_id}.jsonl",
                    expected_output_dir=expected_output_dir(launch_config),
                    launch_config=launch_config,
                )
            )
    return specs
