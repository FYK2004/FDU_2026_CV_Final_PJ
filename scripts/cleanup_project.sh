#!/usr/bin/env bash
# 清理项目中间产物，保留主要实验结果与复现所需代码/配置。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

log() { echo "[cleanup] $*"; }

rm_rf() {
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      sz=$(du -sh "$p" 2>/dev/null | cut -f1)
      rm -rf "$p"
      log "删除 ${p} (${sz})"
    fi
  done
}

log "=== Task1: 背景 traj 可视化 ==="
rm_rf Task1/outputs/background/bicycle/train/traj

log "=== Task1: object_a colmap 重复图像 ==="
rm_rf Task1/outputs/object_a/colmap
rm_rf Task1/outputs/object_b/colmap Task1/outputs/object_c/colmap
rm_rf Task1/outputs/object_a/train/input.ply

log "=== Task1: threestudio 中间 checkpoint ==="
rm_rf Task1/external/threestudio/outputs/magic123-coarse-sd
rm_rf Task1/external/threestudio/outputs/magic123-refine-sd/rgba.png-a_teddy_bear@20260619-140232
rm_rf Task1/external/threestudio/outputs/magic123-refine-sd/rgba.png-a_teddy_bear@20260619-140722
rm_rf Task1/external/threestudio/outputs/magic123-refine-sd/rgba.png-a_teddy_bear@20260619-141210
REFINE_OK="Task1/external/threestudio/outputs/magic123-refine-sd/rgba.png-a_teddy_bear@20260619-141554"
if [[ -d "${REFINE_OK}/ckpts" ]]; then
  rm_rf "${REFINE_OK}/ckpts"
fi

log "=== Task2: 测试 run 与中间 checkpoint ==="
rm_rf Task2/outputs/train/act_env_b_test
for run in act_env_b_only act_env_abc; do
  for ck in 010000 020000 030000 040000; do
    rm_rf "Task2/outputs/train/${run}/checkpoints/${ck}"
  done
  rm_rf "Task2/outputs/train/${run}/checkpoints/050000/training_state"
  rm_rf "Task2/outputs/train/${run}/wandb"
done

log "=== Task2: 冗余数据 split ==="
rm_rf Task2/data/calvin-lerobot/splitD_old
rm_rf Task2/data/calvin-lerobot/splitA
rm_rf Task2/data/calvin-lerobot/splitC
rm_rf Task2/data/calvin-lerobot/splitA_video
rm_rf Task2/data/_hf_raw/.cache

log "=== 日志 / wandb / 缓存 ==="
rm_rf Task2/outputs/wandb Task2/outputs/*.log report/wandb
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
rm_rf dog_clean.ply

log "=== 保留清单 ==="
cat <<EOF
Task1  fusion/wander.mp4 + A/B/C/background 最终点云与 mesh
Task2  act_env_*/checkpoints/050000/pretrained_model/
Task2  eval/zero_shot_env_d/summary.json + chunk_l1_by_horizon.png
Task2  data: splitB, splitABC_merged, splitD
report/figures/
EOF

log "=== 清理后体积 ==="
du -sh Task1 Task2 report 2>/dev/null || true
