# CaSCo on CaP-X 仓库规则

本仓库是 CaSCo 研究在 CaP-X 上的实现与实验执行仓。开始改代码或跑实验前先读本文件；项目交接真源是父目录的 `_CODEX_HANDOFF.md`。

## 当前工作线

这是“研究先行、未来迁移”线。先保证 CaP-X + LLM + simulator 的研究链路可复现，再做 CaSCo 方法层；不要把它当成公司工程部署线，也不要为了未来真机迁移提前扩大当前 Batch 1 范围。

## 分支与代码边界

- 工作分支：`wenqian/casco-mvp`
- 参考分支：`dev-repo` 只作为 wangcx 复现版的取文件来源，不在其上开发，不直接 merge。
- 我们自己的方法层新代码放 `casco/`。
- 绝不改 `capx/skills/`。
- 改 CaP-X 内部文件必须有明确必要性、记录原因，并保持可回滚。
- 每个可验证里程碑 commit + push。

## 服务器与路径

- 唯一服务器入口：`ssh -p 1023 root@101.132.143.105`
- 绝对不要使用 `ssh -p 1024 root@101.132.143.105`。
- 远端代码路径：`/mnt/data/wenqian/cap-skill`
- 远端 venv：`/mnt/nas/wenqian/envs/cap-skill-libero-venv`
- 远端数据与日志：`/mnt/nas/wenqian/cap-skill-data/`
- 不要使用 `/mnt/workspace/wenqian`；不要修改 `/mnt/data/wenqian` 下旧的 12 月 STA 文件。

## Batch 1 真源

Batch 1 只要求跑通一个 privileged LIBERO trial：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0_privileged_minimal.yaml \
    --model gemini-3.5-flash \
    --server-url https://generativelanguage.googleapis.com/v1beta/openai/chat/completions \
    --api-key "$(cat .geminikey)"
```

Batch 1 不需要 qwen、OpenRouter、wangcx proxy、SAM3、GraspNet 或 VLM。只需要 PyRoKi server。

## 密钥安全

- 远端 Gemini key 放仓库根的 `.geminikey`，权限 `600`。
- `.geminikey`、`.qwenkey`、`.openrouterkey`、`.env` 不得提交。
- 不在日志、终端输出、commit message、文档中写出任何 key。
- 不把 Gemini key 发给非 Google 官方 endpoint，尤其不要发给 `39.106.150.148:5018`。

## 环境与实验记录

- 使用干净 venv：`/mnt/nas/wenqian/envs/cap-skill-libero-venv`，不要复用 wangcx 的 `.venv-libero`。
- 安装或 trial 失败时，先保留日志并做 root-cause 分析，不要连续猜改。
- 长跑任务放 tmux，Batch 1 session 名称为 `casco_batch1`。
- 用 GPU 前运行 `nvidia-smi`，选空闲卡并设置 `CUDA_VISIBLE_DEVICES`。
- 每个有意义 run append `/mnt/nas/wenqian/cap-skill-data/ledger.jsonl`。
- 关键进度写 `/mnt/nas/wenqian/cap-skill-data/logs/`。
- 输出目录应不可变，不覆盖旧 run。
