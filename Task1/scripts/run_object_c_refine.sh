#!/usr/bin/env bash
# 物体 C：Magic123 refine（从 coarse 初始化，优化后脑等不可见区域几何）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh
source scripts/env.sh
load_config

PYTHON="$(task1_threestudio_python)"
TS="${EXTERNAL_THREESTUDIO}"
RGBA="${ROOT}/${OBJECT_C_RGBA}"
GPU="${OBJECT_C_GPU}"
PROMPT="${OBJECT_C_PROMPT}"
SD_MODEL="${OBJECT_C_SD_MODEL}"
REFINE_STEPS="${OBJECT_C_REFINE_STEPS}"
COARSE_TRIAL="${OBJECT_C_COARSE_TRIAL:-}"
LOG="${ROOT}/${OBJECT_C_OUT}/train_refine.log"

REFINE_CFG_SRC="${ROOT}/${OBJECT_C_REFINE_CONFIG}"
if [[ ! -f "${REFINE_CFG_SRC}" ]]; then
  REFINE_CFG_SRC="${TS}/configs/magic123-refine-car.yaml"
fi

if [[ -z "${COARSE_TRIAL}" ]]; then
  COARSE_TRIAL="$(ls -td "${TS}/outputs/magic123-coarse-sd"/* 2>/dev/null | head -1 || true)"
fi
COARSE_CKPT="${COARSE_TRIAL}/ckpts/last.ckpt"
[[ -f "${COARSE_CKPT}" ]] || {
  echo "[C] 缺少 coarse checkpoint: ${COARSE_CKPT}"
  echo "    请先运行: bash scripts/run_object_c.sh  (或仅 coarse 段)"
  exit 1
}

mkdir -p "${TS}/configs" "${ROOT}/${OBJECT_C_OUT}"
cp -f "${REFINE_CFG_SRC}" "${TS}/configs/magic123-refine-hifa.yaml"
cp -f "${ROOT}/${OBJECT_C_INPUT}" "${RGBA}" 2>/dev/null || true

cd "${TS}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[C] Magic123 refine | steps=${REFINE_STEPS} | from=${COARSE_CKPT}" | tee "${LOG}"
set +e
"${PYTHON}" launch.py --config configs/magic123-refine-hifa.yaml --train --gpu "${GPU}" \
  seed="${OBJECT_C_SEED:-0}" \
  data.image_path="${RGBA}" \
  data.height=128 \
  data.width=128 \
  data.random_camera.height=128 \
  data.random_camera.width=128 \
  data.random_camera.eval_height=256 \
  data.random_camera.eval_width=256 \
  system.prompt_processor.prompt="${PROMPT}" \
  "system.prompt_processor.pretrained_model_name_or_path=${SD_MODEL}" \
  "system.guidance.pretrained_model_name_or_path=${SD_MODEL}" \
  system.geometry_convert_from="${COARSE_CKPT}" \
  trainer.max_steps="${REFINE_STEPS}" \
  trainer.precision=32 2>&1 | tee -a "${LOG}"
set -e

REFINE_TRIAL="$(ls -td outputs/magic123-refine-sd/* 2>/dev/null | head -1 || true)"
if [[ -z "${REFINE_TRIAL}" ]]; then
  REFINE_TRIAL="$(ls -td outputs/magic123-hifa-refine-sd/* 2>/dev/null | head -1 || true)"
fi
[[ -n "${REFINE_TRIAL}" && -f "${REFINE_TRIAL}/ckpts/last.ckpt" ]] || {
  echo "[C] refine 未生成 checkpoint"
  exit 1
}
echo "[C] refine 完成 → ${REFINE_TRIAL}"
