#!/usr/bin/env bash
# 题目一主流程：A/B/C 重建 → 背景 → B/C 2DGS → 融合视频
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

STEPS=(run_object_a run_object_b run_object_c run_background run_object_bc_2dgs run_fusion_2dgs)
for step in "${STEPS[@]}"; do
  echo ""
  echo "========== ${step}.sh =========="
  bash "scripts/${step}.sh"
done
echo ""
echo "完成 → outputs/fusion/wander.mp4"
