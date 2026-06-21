#!/usr/bin/env bash
# 题目二数据准备：下载/合并 split → video → episode 划分
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh

PYTHON="$(task2_python)"
if [[ ! -x "${PYTHON}" ]]; then
  echo "缺少 task2-lerobot 环境，请先: bash scripts/setup.sh" >&2
  exit 1
fi

need_split() {
  [[ ! -f "data/calvin-lerobot/split${1}/meta/info.json" ]]
}

for s in B D; do
  if need_split "${s}"; then
    "${PYTHON}" src/download_splits.py --splits "${s}"
    "${PYTHON}" src/convert_splits.py --splits "${s}"
  fi
  "${PYTHON}" src/fix_dataset_features.py --splits "split${s}"
done

if [[ ! -f data/calvin-lerobot/splitABC_merged/meta/info.json ]]; then
  for s in A C; do
    if need_split "${s}"; then
      "${PYTHON}" src/download_splits.py --splits "${s}"
      "${PYTHON}" src/convert_splits.py --splits "${s}"
    fi
    "${PYTHON}" src/fix_dataset_features.py --splits "split${s}"
  done
  "${PYTHON}" src/merge_splits.py
  "${PYTHON}" src/fix_dataset_features.py --splits splitABC_merged
fi

for s in splitB splitABC_merged splitD; do
  "${PYTHON}" src/convert_to_video.py --splits "${s}" --num-workers 16 --vcodec h264 --replace-source
done

"${PYTHON}" src/prepare_splits.py
echo "数据准备完成 → data/calvin-lerobot/ + data/splits/"
