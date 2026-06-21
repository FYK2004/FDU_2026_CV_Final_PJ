#!/usr/bin/env bash
# 从 wander.mp4 抽取报告用关键帧 → report/figures/fusion/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VIDEO="${ROOT}/Task1/outputs/fusion/wander.mp4"
OUT="${ROOT}/report/figures/fusion"
mkdir -p "${OUT}"

if [[ ! -f "${VIDEO}" ]]; then
  echo "缺少融合视频: ${VIDEO}" >&2
  exit 1
fi

# 默认均匀取 4 帧；可通过参数指定帧号，如: bash extract_fusion_frames.sh 0 45 90 119
FRAMES=("${@:-0 30 60 90}")
for idx in "${FRAMES[@]}"; do
  out="${OUT}/frame_$(printf '%03d' "${idx}").png"
  ffmpeg -y -loglevel error -i "${VIDEO}" -vf "select=eq(n\\,${idx})" -vframes 1 "${out}"
  echo "→ ${out}"
done

echo "完成。在 main.tex 中引用 figures/fusion/frame_XXX.png"
