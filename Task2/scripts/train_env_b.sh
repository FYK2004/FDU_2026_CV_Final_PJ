#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh

PYTHON="$(task2_python)"

# 数据准备（若 splitB 已存在会跳过下载/转换）
if [[ ! -f data/calvin-lerobot/splitB/meta/info.json ]]; then
  "${PYTHON}" src/download_splits.py --splits B
  "${PYTHON}" src/convert_splits.py --splits B
fi
"${PYTHON}" src/fix_dataset_features.py --splits splitB
"${PYTHON}" src/prepare_splits.py

LOG="${ROOT}/outputs/train_env_b.log"
mkdir -p outputs/train
: > "${LOG}"

nohup "${PYTHON}" -u src/train_act.py --split env_b_only >> "${LOG}" 2>&1 &
echo "后台训练已启动 PID=$!"
echo "日志: ${LOG}"
echo "权重: outputs/train/act_env_b_only/checkpoints/"
