# 自定义改动记录

本文件记录在原仓库基础上所做的较大功能修改，便于溯源和同步上游时参考。

---

## 1. SAM3 支持本地权重加载 - 2026.4.22

**背景**：原实现只能从 HuggingFace 下载 `facebook/sam3` 权重，需要 HF 访问权限；在无法联网或已预先下好权重的环境下不可用。

**改动范围**：

- `capx/serving/launch_sam3_server.py`
  - `main(...)` 新增参数 `checkpoint_path: str | None = None`。
  - 权重解析优先级：**YAML/CLI `checkpoint_path` > 环境变量 `SAM3_CHECKPOINT_PATH` > HF 默认下载**。
  - 若解析到路径但文件不存在，直接 `FileNotFoundError`，不做静默回退。
  - 使用本地路径时调用 `build_sam3_image_model(checkpoint_path=..., load_from_HF=False)`。
- `README.md`
  - SAM3 小节增加"使用本地权重"说明（env var + YAML 字段两种用法）。
- 未修改第三方代码 `capx/third_party/sam3`（`build_sam3_image_model` 原生已支持 `checkpoint_path`）。
- 未批量修改 env_configs 下的 YAML；需要时在对应 `api_servers` 条目里加 `checkpoint_path` 字段即可。

**使用方式**：

```bash
export SAM3_CHECKPOINT_PATH=/mnt/oss/wangcx/checkpoints/sam3/sam3.pt
```

或在 YAML 的 `api_servers` 条目中：

```yaml
- _target_: capx.serving.launch_sam3_server.main
  device: cuda
  port: 8114
  host: 127.0.0.1
  checkpoint_path: /mnt/oss/wangcx/checkpoints/sam3/sam3.pt
```

**兼容性**：未设置 env var 且 YAML 未写该字段时，行为与改动前完全一致（走 HF 下载）。手动 `launch_servers.py` 和 eval 自动拉起两条路径都支持。运行时只使用 `sam3.pt`，`config.json` 无需本地准备。

---

## 2. 统一 LLM 代理 + 新增 Qwen (DashScope) 系列 - 2026.4.22

**背景**：原 `openrouter_server.py` 只能对接单一 OpenRouter 上游；想新增 Qwen proxy（`http://39.106.150.148:5018/v1`，DashScope 后端）时，改 `api_key/base_url` 不够，因为 DashScope 模型必须在请求体里带 `backend=dashscope`，而原 server 的严格 Pydantic schema 会丢弃未知字段。

**改动范围**：

- 新增 `capx/serving/llm_proxy_server.py`：多路由 FastAPI 代理，按 `model` 名分发到不同上游。`ChatRequest` 用 `ConfigDict(extra="allow")` 透传 `backend` / `enable_thinking` / `chat_template_kwargs` 等字段到上游的 `extra_body`；支持 SSE 流式、`/health`；key 文件缺失时**打 warning 跳过该路由**（全部路由都不可用才抛错），符合"不静默回退"。
- 新增 `capx/serving/llm_routes.yaml`：两条默认路由
  - `openrouter`（前缀匹配 `openrouter/`，strip prefix，保留原 `HTTP-Referer` / `X-Title` headers）
  - `qwen-dashscope`（白名单 6 个 DashScope 模型，`default_extra_body.backend=dashscope`）
- 重写 `capx/serving/openrouter_server.py` 为薄 shim：原 CLI（`--key-file/--api-key/--base-url/--port/--async-client`）保持不变，内部委托给 `llm_proxy_server`；新增 `--config` 可切到多路由模式。所有 `scripts/*.sh` / README / docs 里的旧调用零感知。
- 编辑 `capx/llm/client.py`：新增 `QWEN_DASHSCOPE_MODELS`、`is_qwen_proxy_model`、`is_local_proxy_model`；`query_model` 和 `query_model_streaming` 都会把 Qwen 模型路由到 `http://localhost:8110/chat/completions`；新增 Qwen payload 分支（不写 `backend`，由代理注入）。端口/`OPENROUTER_SERVER_URL` 常量不变。
- 编辑 `docs/configuration.md`："Adding new LLM providers" 小节重写为统一代理用法，列出 6 个 Qwen 模型名，并保留 legacy 入口说明。
- 编辑 `.gitignore`：新增 `.qwenkey`。
- `pyproject.toml` 无需改动（`pyyaml` 已存在）。

**Qwen 支持的模型名**：`qwen3.5-plus` / `qwen3.5-27b` / `qwen3.5-35b-a3b` / `qwen3.5-122b-a10b` / `qwen3.5-397b-a17b` / `qwen3-vl-235b-a22b-instruct`。

**使用方式**：

```bash
echo "sk-xxx-qwen-key"       > .qwenkey
echo "sk-or-v1-xxx"          > .openrouterkey   # 可选
uv run --no-sync --active capx/serving/llm_proxy_server.py \
    --config capx/serving/llm_routes.yaml --port 8110
# 业务侧直接用 --model qwen3.5-plus 即可，无需改 --server-url
```

**兼容性**：端口仍 `8110`；`python -m capx.serving.openrouter_server --key-file .openrouterkey --port 8110` 依旧可用；`client.py` 公开 API（`is_openrouter_model`、`query_model`、`ModelQueryArgs` 等）签名未变。

**已知限制 / 后续可做**：~~Qwen DashScope 返回的思考链在 `message.reasoning_content`（非 `reasoning`），当前 `client.py` 的 reasoning 提取未适配这一字段，需要时再加。~~ 已在下方 §3 中修复。

---

## 3. Qwen Thinking 控制 + reasoning_content 兼容 - 2026.4.22

**背景**：Qwen DashScope 后端支持 `enable_thinking` 开关，但之前调用链路里没有控制入口；同时 DashScope 返回的思考链字段是 `reasoning_content`（不是 `reasoning`），业务层拿不到。

**改动范围**：

- `capx/llm/client.py`
  - 新增 `normalize_reasoning_effort(effort) -> (bool, str)` 工具函数，鲁棒处理大小写/类型/None：off/none/minimal/""/false/0 → off；low/medium/high/on/true/1 → on；未知值 `ValueError`。
  - `query_model` / `query_model_streaming` 的 Qwen 分支：根据 `args.reasoning_effort` 判断是否在 payload 里加 `"enable_thinking": True`。
  - reasoning 提取统一改为 `msg.get("reasoning") or msg.get("reasoning_content")`，覆盖非流式、流式 JSON fallback、SSE delta 三条路径。
- `capx/envs/runner.py`
  - `_setup_output_dir`：Qwen 模型输出目录自动加 `__think-<level>` 后缀（如 `qwen3.5-35b-a3b__think-medium`），非 Qwen 模型不变。

**使用方式**：

```bash
# thinking ON（默认 medium = ON）
uv run ... --model qwen3.5-35b-a3b
# -> outputs/qwen3.5-35b-a3b__think-medium/...

# thinking OFF
uv run ... --model qwen3.5-35b-a3b --reasoning-effort off
# -> outputs/qwen3.5-35b-a3b__think-off/...

# thinking HIGH
uv run ... --model qwen3.5-35b-a3b --reasoning-effort high
# -> outputs/qwen3.5-35b-a3b__think-high/...
```

**兼容性**：非 Qwen 模型（GPT/Claude/OpenRouter/OSS/vLLM）的行为和输出目录完全不变。`--reasoning-effort` 默认值仍为 `medium`。

---

## 4. 实验组织：exp_name + timestamp + metadata.json - 2026.4.22

**背景**：原输出路径只有 `outputs/{model_tag}/{config_stem}/` 两级，存在三个痛点：(1) 同一 `(model, config)` 重复跑会**直接覆盖**之前的结果；(2) 没有 campaign 级别的分组，批量跑消融 / 对比多个 agent system 时不好收敛；(3) 只有纯文本 `summaries.txt`，没有结构化数据供 `pandas` 聚合分析。

**改动范围**：

- `capx/envs/launch.py`
  - `LaunchArgs` 新增两个字段：
    - `exp_name: str | None = None` — 可选 campaign 名，非空时作为最外层目录插入。
    - `timestamp: bool = False` — 是否在输出路径末尾追加时间戳子目录（默认不加，显式打开以便区分多次 run）。
  - tyro 自动暴露为 `--exp-name <name>` / `--timestamp`（关闭可用 `--no-timestamp`，但那是默认行为）。
- `capx/utils/launch_utils.py`
  - `_load_config` 的 `merged_config` 里合并 `exp_name` / `timestamp`（沿用"CLI 优先、YAML 补位"模式）。
  - `_save_trial_artifacts` 在 trial 目录末尾额外写 **trial-level** `metadata.json`（字段见下）。
  - `_print_and_save_summary` 在 `summaries.txt` 之后额外写 **run-level** `metadata.json`（字段见下）。
- `capx/envs/runner.py`
  - `_setup_output_dir` 重写：
    - 保留原有 `model_tag` 构造（含 Qwen `__think-{level}` 后缀，§3）。
    - 如 `config["exp_name"]` 非空，在 `model_tag` 之前再插一层。
    - 仅当 `config["timestamp"]` 为真时，在末尾追加 `time.strftime("%Y%m%d_%H%M%S")` 子目录（**一次 launch 内只生成一次**，回写到 `config["run_timestamp"]`，保证所有 worker 共享同一个 timestamp）。
- `docs/configuration.md`
  - CLI flags 表格补 `--exp-name` / `--timestamp`。
  - YAML 示例补 `exp_name` / `timestamp` 两个可选字段。
  - 新增 "Output directory layout" 小节：画出完整目录树，列出 run / trial metadata.json 的所有字段。

> **命名变更（2026.4.24）**：最初字段叫 `no_timestamp`（默认 `True`，即默认不加时间戳），tyro 下想开启时间戳需要写反人类直觉的 `--no-no-timestamp`。改为正向命名 `timestamp: bool = False` 后，开启用 `--timestamp`、关闭用 `--no-timestamp`，语义直观。**默认行为也从原来的"默认加时间戳"改成了"默认不加时间戳"**——这是一个 breaking change，见下方兼容性小节。

**最终路径结构**：

```
outputs/
  [{exp_name}/]                 # 仅当显式设置时出现
    {model_tag}/                # Qwen 含 __think-{level} 后缀
      {config_stem}/
        [{YYYYMMDD_HHMMSS}/]    # 默认关闭；显式 --timestamp 开启
          summaries.txt         # 原有（人类可读）
          metadata.json         # 新增：run-level 结构化汇总
          trial_XX_sandboxrc_*_reward_*_taskcompleted_*/
            code.py             # 原有
            summary.txt         # 原有
            video_*.mp4         # 原有
            metadata.json       # 新增：trial-level 结构化明细
            ...
```

**metadata.json 字段**：

- **Run-level**（`{run_root}/metadata.json`）：`run_id`、`run_timestamp`、`exp_name`、`model`、`reasoning_effort`、`temperature`、`visual_differencing_model`、`config_path`、`config_stem`、`output_dir`（绝对路径）、`git_commit`、`git_dirty`、`total_trials`、`num_workers`、`task_completed`、`success_rate`、`average_reward`、`average_code_blocks`、`average_regenerations`、`average_finishes`、`elapsed_sec`、`started_at`、`finished_at`、`agent_flags`（嵌套对象，含 `use_img_differencing` / `use_visual_feedback` / `use_video_differencing` / `use_wrist_camera` / `use_parallel_ensemble` / `use_multimodel` / `use_oracle_code`）。
- **Trial-level**（`{run_root}/trial_*/metadata.json`）：`trial`、`sandbox_rc`、`reward`、`task_completed`、`num_responses`、`has_raw_code`、`has_ensemble`、`has_multiturn_ensemble`、`num_visual_feedback_imgs`。

**使用方式**：

```bash
# 1) 默认行为：固定路径（与最早的原始行为一致，反复跑会覆盖）
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
    --model qwen3.5-35b-a3b
# -> outputs/qwen3.5-35b-a3b__think-medium/franka_robosuite_cube_stack_main_use_server/

# 2) 想区分多次 run（不互相覆盖）：显式开启 timestamp
uv run ... --model qwen3.5-35b-a3b --timestamp
# -> outputs/qwen3.5-35b-a3b__think-medium/franka_robosuite_cube_stack_main_use_server/20260422_153012/

# 3) 为一批消融实验起一个 campaign 名（可与 --timestamp 组合）
uv run ... --model qwen3.5-35b-a3b --exp-name ablation_skill_lib_v3 --timestamp
# -> outputs/ablation_skill_lib_v3/qwen3.5-35b-a3b__think-medium/.../20260422_153012/
```

YAML 侧（可选，CLI 优先级更高）：

```yaml
exp_name: ablation_skill_lib_v3
timestamp: true
```

**分析建议**：将来写跨 run 对比脚本时，**不要靠解析路径拿维度信息**，直接 glob `outputs/**/metadata.json` + `pandas.json_normalize` 即可得到扁平表格。`agent_flags` 嵌套子对象会被 `json_normalize` 展开为 `agent_flags.use_img_differencing` 这种列名，方便 pivot。路径结构未来若再调整（例如拆 `task` / `method` config），只要 metadata.json schema 不变，分析代码无需改。

**兼容性**：

- **默认不设 `exp_name` 且不加 `--timestamp` 时，路径层级与最早的原始行为完全一致**（`outputs/{model_tag}/{config_stem}/`），反复跑会覆盖。如需保留历史 run，显式加 `--timestamp`。
- `scripts/regression_test.sh` 只 `grep` stdout 里的 `success/reward/completed` 三元组，不 hard-code 路径，**不回归**。
- `record_video=true` 的视频路径由 `capx/envs/trial.py:_get_trial_dir` 基于 `config["output_dir"]` 拼接，自动跟随新路径，无需改。
- **2026.4.22 ~ 2026.4.24 期间的老用法**（`no_timestamp` 字段、默认会追加 timestamp）已移除；若有脚本写过 `--no-timestamp` 旧 flag 或 YAML 里的 `no_timestamp: true/false`，需改为 `--timestamp` / `timestamp: true/false`。由于默认值翻转，原本依赖"默认带时间戳"收敛结果的脚本需要显式加 `--timestamp`。
- 老的分析脚本若硬编码 `ls outputs/{model}/{config}/trial_*` 现在反而能匹配（默认不再有 timestamp 中间层）；用了 `--timestamp` 的 run 需改成 `outputs/{model}/{config}/**/trial_*` 或走 `glob **/metadata.json` 的新方式。

**未来扩展留坑**（暂未实现，按需再加）：

- 把 `agent method` 从 monolithic YAML 里拆出来（像 `--task foo --method bar` 这种组合），让 agent system 变成一等公民的维度。
- 接 wandb / mlflow（目前 `metadata.json` 已足够做离线分析；若接线上 tracker，建议在 `_print_and_save_summary` 里再 `log_run(metadata)` 一下即可）。
- run-level `metadata.json` 增加 `per_trial` 字段（列出每个 trial 的 reward / completed），省掉聚合时还要再 glob trial-level metadata 的一步。
