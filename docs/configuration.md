# Configuration Reference

## CLI flags

Override any YAML config field from the command line:

```bash
uv run --no-sync --active capx/envs/launch.py \
    --config-path <config.yaml> \
    --model google/gemini-3.1-pro-preview \
    --server-url http://127.0.0.1:8110/chat/completions \
    --temperature 1.0 \
    --total-trials 100 \
    --num-workers 12 \
    --record-video True
```

| Flag                | Default                                  | Description                           |
| ------------------- | ---------------------------------------- | ------------------------------------- |
| `--config-path`     | *(required)*                             | Path to YAML task config              |
| `--model`           | `google/gemini-3.1-pro-preview`          | Model name                            |
| `--server-url`      | `http://127.0.0.1:8110/chat/completions` | LLM endpoint                          |
| `--temperature`     | `1.0`                                    | Sampling temperature                  |
| `--total-trials`    | from YAML                                | Number of evaluation trials           |
| `--num-workers`     | from YAML                                | Parallel worker count                 |
| `--web-ui`          | `False`                                  | Launch interactive web UI             |
| `--use-oracle-code` | `False`                                  | Run human-written reference solutions |
| `--exp-name`        | *(unset)*                                | Optional campaign name, inserted as a top-level directory under `outputs/` (before the model). Lets you group a batch of related runs. |
| `--timestamp`       | `False`                                  | Append a `YYYYMMDD_HHMMSS` subdirectory to the output path so repeated runs don't overwrite each other. Off by default for backward compatibility. |

## YAML config format

```yaml
# env_configs/my_task/my_task.yaml
env:
  _target_: capx.envs.tasks.my_robot.my_task.MyTaskCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: my_sim_env
    privileged: false
    apis:
      - FrankaControlApi

record_video: true
output_dir: ./outputs/my_task
trials: 100
num_workers: 12

# Optional experiment-tracking fields (can also be set via CLI flags)
exp_name: null          # e.g. "ablation_skill_lib_v3" — inserted above {model}/
timestamp: false        # set true to append a YYYYMMDD_HHMMSS subdirectory (avoid overwriting prior runs)
```

The `_target_` keys enable Hydra-style lazy instantiation via `capx.envs.configs.instantiate()`.

### Output directory layout

Each run writes artifacts to a path of the form:

```
{outputs_root}/[{exp_name}/]{model_tag}/{config_stem}/[{timestamp}/]
    summaries.txt           # human-readable summary
    metadata.json           # run-level structured metadata
    trial_XX_sandboxrc_*_reward_*_taskcompleted_*/
        code.py
        summary.txt
        video_*.mp4
        metadata.json       # trial-level structured metadata
        ...
```

- `{outputs_root}` and `{config_stem}` come from the YAML `output_dir` field.
- `{exp_name}` layer is present only when `exp_name` is explicitly set (backward compatible).
- `{model_tag}` is derived from `--model` (slashes replaced with underscores); Qwen proxy models append a `__think-{level}` suffix derived from `--reasoning-effort`.
- `{timestamp}` is `YYYYMMDD_HHMMSS`, captured once per launch. Appears only when `--timestamp` is passed on the CLI (or `timestamp: true` is set in YAML); otherwise the directory ends at `{config_stem}/` and repeated runs overwrite each other.

#### `metadata.json` fields

Run-level (`{run_root}/metadata.json`) includes `run_id`, `run_timestamp`, `exp_name`, `model`, `reasoning_effort`, `temperature`, `config_path`, `config_stem`, `output_dir`, `git_commit`, `git_dirty`, `total_trials`, `num_workers`, `task_completed`, `success_rate`, `average_reward`, `average_code_blocks`, `average_regenerations`, `average_finishes`, `elapsed_sec`, `started_at`, `finished_at`, and an `agent_flags` sub-object (`use_img_differencing`, `use_visual_feedback`, `use_video_differencing`, `use_wrist_camera`, `use_parallel_ensemble`, `use_multimodel`, `use_oracle_code`).

Trial-level (`{run_root}/trial_*/metadata.json`) includes `trial`, `sandbox_rc`, `reward`, `task_completed`, `num_responses`, `has_raw_code`, `has_ensemble`, `has_multiturn_ensemble`, and `num_visual_feedback_imgs`.

To aggregate runs for analysis, glob `outputs/**/metadata.json` and load with `pandas.json_normalize`, which sidesteps any future path-layout changes.

### Perception servers (api_servers)

YAML configs can include an `api_servers` section that **auto-launches** perception servers when the evaluation starts:

```yaml
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1

  - _target_: capx.serving.launch_contact_graspnet_server.main
    port: 8115
    host: 127.0.0.1

  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
    robot: panda_description
    target_link: panda_hand
```

The launcher automatically:
- Skips servers whose port is already in use (e.g. started externally)
- Waits for all servers to be ready before running trials
- Terminates all servers on exit

If you prefer to manage servers separately (e.g. for sharing across multiple eval runs), use `launch_servers.py`:

```bash
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

| Profile | Servers | GPU Required |
|---------|---------|-------------|
| `default` | SAM3 (8114) + ContactGraspNet (8115) + PyRoKi (8116) | Yes (~5 GB VRAM) |
| `full` | default + OWL-ViT (8118) + SAM2 (8113) | Yes (~14 GB VRAM) |
| `minimal` | PyRoKi (8116) only | No (CPU-only) |

## Adding new LLM providers

CaP-X queries language models through a local proxy server that exposes an OpenAI-compatible `/chat/completions` endpoint. A single unified proxy (`capx/serving/llm_proxy_server.py`) dispatches to one of several upstream providers based on the requested model name, so all callers keep hitting the same URL (default `http://localhost:8110/chat/completions`).

### Unified proxy (recommended)

Upstream routes are declared in [capx/serving/llm_routes.yaml](../capx/serving/llm_routes.yaml). Ship with two routes out of the box: OpenRouter and a Qwen proxy (DashScope backend). A route whose key file is missing is skipped at startup with a warning, so you only need the keys for providers you actually use.

1. Drop in whichever API keys you have:
   ```bash
   echo "sk-or-v1-your-openrouter-key"   > .openrouterkey   # OpenRouter
   echo "sk-your-qwen-proxy-key"         > .qwenkey         # Qwen DashScope
   ```
2. Start the unified proxy:
   ```bash
   uv run --no-sync --active capx/serving/llm_proxy_server.py \
       --config capx/serving/llm_routes.yaml --port 8110
   ```
3. Reference models by the IDs the routes expect:
   - OpenRouter: any `openrouter/<vendor>/<model>` name (e.g. `openrouter/google/gemini-2.5-pro-preview`). The proxy strips the `openrouter/` prefix before forwarding.
   - Qwen DashScope (6 models supported out of the box):
     - `qwen3.5-plus`
     - `qwen3.5-27b`
     - `qwen3.5-35b-a3b`
     - `qwen3.5-122b-a10b`
     - `qwen3.5-397b-a17b`
     - `qwen3-vl-235b-a22b-instruct`

   For Qwen models the proxy automatically injects `backend=dashscope` into each upstream request; callers can still override it by sending an explicit `backend` field.

   **Thinking control:** Qwen thinking (DashScope `enable_thinking`) is driven by the existing `--reasoning-effort` flag. Default `medium` means thinking ON; pass `--reasoning-effort off` to disable it. The output directory for Qwen runs is automatically suffixed with `__think-<level>` (e.g. `qwen3.5-35b-a3b__think-medium`) so different thinking configs do not overwrite each other. Non-Qwen models are unaffected.

### Legacy OpenRouter-only launch (still supported)

The old entrypoint continues to work for backwards compatibility with existing scripts:

```bash
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

Under the hood this delegates to the unified proxy with a single OpenRouter-only route. Pass `--config capx/serving/llm_routes.yaml` to get the multi-provider behaviour from the legacy entrypoint.

### Option B: vLLM (local models)

```bash
uv run python -m capx.serving.vllm_server --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8080 --tensor-parallel-size 4
```

### Option C: Custom providers

Add a new entry under `routes:` in [capx/serving/llm_routes.yaml](../capx/serving/llm_routes.yaml) pointing at any OpenAI-compatible upstream. Each route supports:

- `prefix_match` / `model_match`: which incoming model names hit this route.
- `strip_prefix`: prefix removed before the upstream call.
- `base_url` / `key_file`: upstream endpoint and key.
- `default_headers`: extra HTTP headers (e.g. `HTTP-Referer`).
- `default_extra_body`: non-OpenAI fields merged into every request (e.g. `backend: dashscope`).

Then register the new model name in `capx/llm/client.py` (e.g. by adding it to `QWEN_DASHSCOPE_MODELS` or creating a new list and extending `is_local_proxy_model`) so `query_model` routes it through the proxy URL.

> **Note:** `.openrouterkey` and `.qwenkey` are git-ignored. Never commit API keys to the repository.
