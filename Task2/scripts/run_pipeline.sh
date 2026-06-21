#!/usr/bin/env bash
# 题目二主流程：数据准备 → B-only 训练 → ABC 训练 → D 零样本评测
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

bash scripts/prepare_data.sh
bash scripts/train_env_b.sh
echo "等待 B-only 训练完成后运行: bash scripts/train_env_abc.sh && bash scripts/eval_zero_shot_d.sh"
