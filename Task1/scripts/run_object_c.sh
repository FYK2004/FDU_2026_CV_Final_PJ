#!/usr/bin/env bash
# 物体 C：Magic123 coarse → refine → mesh 导出（refine 改善后脑等不可见区域）
# 参考: https://github.com/threestudio-project/threestudio#magic123-
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh
source scripts/env.sh
load_config

PYTHON="$(task1_threestudio_python)"
TS="${EXTERNAL_THREESTUDIO}"
OUT="${ROOT}/${OBJECT_C_OUT}"
RGBA="${ROOT}/${OBJECT_C_RGBA}"
INPUT="${ROOT}/${OBJECT_C_INPUT}"
GPU="${OBJECT_C_GPU}"
PROMPT="${OBJECT_C_PROMPT}"
SD_MODEL="${OBJECT_C_SD_MODEL}"
COARSE_CFG="${ROOT}/${OBJECT_C_COARSE_CONFIG}"
COARSE_STEPS="${OBJECT_C_COARSE_STEPS}"
LOG="${OUT}/train_coarse.log"
SKIP_COARSE="${SKIP_COARSE:-0}"
SKIP_REFINE="${SKIP_REFINE:-0}"

mkdir -p "${OUT}" "${TS}/configs"
cp -f "${ROOT}/external/threestudio/configs/magic123-coarse-hifa.yaml" "${TS}/configs/" 2>/dev/null || true
cp -f "${INPUT}" "${RGBA}"

cd "${TS}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

COARSE_TRIAL=""
if [[ "${SKIP_COARSE}" != "1" ]]; then
  echo "[C] Magic123 coarse (官方 HiFA) | steps=${COARSE_STEPS}" | tee "${LOG}"
  set +e
  "${PYTHON}" launch.py --config "${COARSE_CFG}" --train --gpu "${GPU}" \
    seed="${OBJECT_C_SEED:-0}" \
    data.image_path="${RGBA}" \
    system.prompt_processor.prompt="${PROMPT}" \
    "system.prompt_processor.pretrained_model_name_or_path=${SD_MODEL}" \
    "system.guidance.pretrained_model_name_or_path=${SD_MODEL}" \
    trainer.max_steps="${COARSE_STEPS}" 2>&1 | tee -a "${LOG}"
  set -e
  COARSE_TRIAL="$(ls -td outputs/magic123-coarse-sd/* 2>/dev/null | head -1)"
else
  COARSE_TRIAL="$(ls -td outputs/magic123-coarse-sd/* 2>/dev/null | head -1)"
  echo "[C] 跳过 coarse，使用已有 trial: ${COARSE_TRIAL}"
fi

if [[ "${SKIP_REFINE}" != "1" ]]; then
  OBJECT_C_COARSE_TRIAL="${TS}/${COARSE_TRIAL}" bash "${ROOT}/scripts/run_object_c_refine.sh"
  REFINE_TRIAL="$(ls -td "${TS}/outputs/magic123-refine-sd"/* 2>/dev/null | head -1 || true)"
  if [[ -z "${REFINE_TRIAL}" ]]; then
    REFINE_TRIAL="$(ls -td "${TS}/outputs/magic123-hifa-refine-sd"/* 2>/dev/null | head -1 || true)"
  fi
  export OBJECT_C_MESH_TRIAL="${REFINE_TRIAL}"
  export OBJECT_C_MESH_BACKUP="model_refine.obj"
  export OBJECT_C_EXPORT_RESOLUTION="${OBJECT_C_REFINE_EXPORT_RESOLUTION:-128}"
  export OBJECT_C_EXPORT_THRESHOLD="${OBJECT_C_REFINE_EXPORT_THRESHOLD:-25.0}"
  bash "${ROOT}/scripts/run_object_c_export_mesh.sh"
else
  echo "[C] 需要 refine，请勿设置 SKIP_REFINE=1" >&2
  exit 1
fi

echo "[C] 完成 → ${OUT}/model.obj"
echo "[C] 若已更新 mesh，请重训 2DGS 并融合:"
echo "      bash scripts/run_object_bc_2dgs.sh  # 可设 SKIP_RENDER=0 仅 object_c"
echo "      bash scripts/run_fusion_2dgs.sh --c-iteration 10000"
