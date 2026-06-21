#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh

PYTHON="$(task2_python)"

# 下载 splitA/C（splitB 第一步已完成）
for split in A C; do
  if [[ ! -f "data/calvin-lerobot/split${split}/meta/info.json" ]]; then
    echo "下载 split${split} ..."
    "${PYTHON}" src/download_splits.py --splits "${split}"
    "${PYTHON}" src/convert_splits.py --splits "${split}"
    "${PYTHON}" src/fix_dataset_features.py --splits "split${split}"
  fi
done

# 合并 A+B+C
if [[ ! -f data/calvin-lerobot/splitABC_merged/meta/info.json ]]; then
  "${PYTHON}" src/merge_splits.py
  "${PYTHON}" src/fix_dataset_features.py --splits splitABC_merged
fi

"${PYTHON}" src/prepare_splits.py

if [[ -d outputs/train/act_env_abc/checkpoints ]]; then
  echo "已有 checkpoint: outputs/train/act_env_abc/checkpoints/，跳过训练" >&2
  exit 0
fi

LOG="${ROOT}/outputs/train_env_abc.log"
mkdir -p outputs/train
: > "${LOG}"

nohup "${PYTHON}" -u src/train_act.py --split env_abc >> "${LOG}" 2>&1 &
echo "后台训练已启动 PID=$!"
echo "日志: ${LOG}"
echo "权重: outputs/train/act_env_abc/checkpoints/"
