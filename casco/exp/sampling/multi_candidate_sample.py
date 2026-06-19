"""Run or dry-run CaSCo Batch 2.5 multi-candidate LIBERO sampling."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from casco.bench.libero_configs import materialize_privileged_minimal_config
from casco.exp.batch1_launch import build_launch_command, build_launch_env, launch_output_base
from casco.exp.ledger import append_entry
from casco.exp.runs import build_ledger_entry_from_run
from casco.skills.candidates import candidate_records_from_run, write_candidate_records_jsonl

from .sample_plan import SamplingPlanConfig, SamplingRunSpec, build_sampling_plan


def _default_run_group_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_plan(path: Path, specs: list[SamplingRunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"runs": [spec.redacted_dict() for spec in specs]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_failed_launch(
    *,
    spec: SamplingRunSpec,
    ledger_path: Path,
    phase: str,
    exit_code: int,
    note: str,
) -> None:
    append_entry(
        ledger_path,
        {
            "run_id": spec.run_id,
            "phase": phase,
            "status": "FAILED",
            "stage": spec.stage,
            "model": spec.launch_config.model,
            "config": str(spec.config_path),
            "output_dir": str(spec.expected_output_dir),
            "result": {"exit_code": exit_code},
            "note": note,
        },
    )


def run_sampling_plan(
    specs: list[SamplingRunSpec],
    *,
    repo_dir: Path,
    ledger_path: Path,
    phase: str,
    stop_on_failure: bool = False,
) -> list[dict[str, object]]:
    """Execute a sampling plan sequentially and record candidates/ledger entries."""
    results: list[dict[str, object]] = []
    for spec in specs:
        materialize_privileged_minimal_config(
            template_path=spec.template_path,
            output_path=spec.config_path,
            suite_name=spec.suite_name,
            task_id=spec.task_id,
            output_dir=launch_output_base(spec.launch_config),
        )
        completed = subprocess.run(
            build_launch_command(spec.launch_config),
            cwd=repo_dir,
            env=build_launch_env(spec.launch_config),
            check=False,
        )
        result: dict[str, object] = {
            "run_id": spec.run_id,
            "task_id": spec.task_id,
            "sample_index": spec.sample_index,
            "exit_code": completed.returncode,
            "expected_output_dir": str(spec.expected_output_dir),
        }
        if spec.expected_output_dir.exists():
            try:
                records = candidate_records_from_run(spec.expected_output_dir)
                write_candidate_records_jsonl(spec.candidate_jsonl, records)
                entry = build_ledger_entry_from_run(
                    spec.expected_output_dir,
                    phase=phase,
                    status="KEPT" if completed.returncode == 0 else "FAILED",
                    stage=spec.stage,
                    exit_code=completed.returncode,
                )
                append_entry(ledger_path, entry)
                result["candidate_jsonl"] = str(spec.candidate_jsonl)
                result["candidate_count"] = len(records)
            except Exception as exc:  # pragma: no cover - exercised on remote failures.
                _append_failed_launch(
                    spec=spec,
                    ledger_path=ledger_path,
                    phase=phase,
                    exit_code=completed.returncode,
                    note=f"artifact parsing failed: {exc}",
                )
                result["error"] = str(exc)
        else:
            _append_failed_launch(
                spec=spec,
                ledger_path=ledger_path,
                phase=phase,
                exit_code=completed.returncode,
                note="launch produced no expected output directory",
            )
            result["error"] = "missing expected output directory"

        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if completed.returncode != 0 and stop_on_failure:
            break
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-ids", required=True, nargs="+", type=int)
    parser.add_argument("--samples-per-task", required=True, type=int)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-group-id", default=None)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/nas/wenqian/cap-skill-data"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml"),
    )
    parser.add_argument("--key-file", type=Path, default=Path(".geminikey"))
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--repo-dir", type=Path, default=Path("."))
    parser.add_argument("--ledger", type=Path, default=Path("/mnt/nas/wenqian/cap-skill-data/ledger.jsonl"))
    parser.add_argument("--mujoco-gl", default="osmesa")
    parser.add_argument("--phase", default="batch2_5_multisample")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> SamplingPlanConfig:
    return SamplingPlanConfig(
        suite_name=args.suite,
        task_ids=tuple(args.task_ids),
        samples_per_task=args.samples_per_task,
        stage=args.stage,
        run_group_id=args.run_group_id or _default_run_group_id(),
        data_root=args.data_root,
        template_path=args.template,
        key_file=args.key_file,
        cuda_visible_devices=args.cuda_visible_devices,
        model=args.model,
        uv_bin=args.uv_bin,
        mujoco_gl=args.mujoco_gl,
    )


def main() -> None:
    args = _parse_args()
    specs = build_sampling_plan(_config_from_args(args))
    if args.dry_run:
        output = args.plan_output or args.data_root / "logs" / "batch2_5_sampling_plan.json"
        _write_plan(output, specs)
        print(f"wrote sampling plan to {output}")
        return

    if args.plan_output:
        _write_plan(args.plan_output, specs)
    run_sampling_plan(
        specs,
        repo_dir=args.repo_dir,
        ledger_path=args.ledger,
        phase=args.phase,
        stop_on_failure=args.stop_on_failure,
    )


if __name__ == "__main__":
    main()
