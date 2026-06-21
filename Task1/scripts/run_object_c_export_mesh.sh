#!/usr/bin/env bash
# 物体 C：从任意 Magic123 trial checkpoint 用 mesh-exporter 导出 model.obj
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh
source scripts/env.sh
load_config

PYTHON="$(task1_threestudio_python)"
TS="${EXTERNAL_THREESTUDIO}"
OUT="${ROOT}/${OBJECT_C_OUT}"
GPU="${OBJECT_C_GPU}"
TEXTURE_SIZE="${OBJECT_C_TEXTURE_SIZE:-2048}"
ISO_THRESHOLD="${OBJECT_C_EXPORT_THRESHOLD:-${OBJECT_C_COARSE_EXPORT_THRESHOLD:-35.0}}"
ISO_METHOD="${OBJECT_C_EXPORT_METHOD:-${OBJECT_C_COARSE_EXPORT_METHOD:-mc-cpu}}"
ISO_RESOLUTION="${OBJECT_C_EXPORT_RESOLUTION:-${OBJECT_C_COARSE_EXPORT_RESOLUTION:-256}}"
OUTLIER_TH="${OBJECT_C_OUTLIER_THRESHOLD:-0.02}"
MESH_TRIAL="${OBJECT_C_MESH_TRIAL:-}"
BACKUP_NAME="${OBJECT_C_MESH_BACKUP:-}"

if [[ -z "${MESH_TRIAL}" ]]; then
  MESH_TRIAL="$(ls -td "${TS}/outputs/magic123-refine-sd"/* 2>/dev/null | head -1 || true)"
  if [[ -z "${MESH_TRIAL}" ]]; then
    MESH_TRIAL="$(ls -td "${TS}/outputs/magic123-hifa-refine-sd"/* 2>/dev/null | head -1 || true)"
  fi
  if [[ -z "${MESH_TRIAL}" ]]; then
    MESH_TRIAL="$(ls -td "${TS}/outputs/magic123-coarse-sd"/* 2>/dev/null | head -1 || true)"
  fi
fi
[[ -n "${MESH_TRIAL}" && -f "${MESH_TRIAL}/ckpts/last.ckpt" ]] || {
  echo "[C] 未找到 trial checkpoint: ${MESH_TRIAL}"
  exit 1
}

mkdir -p "${OUT}"
cp -f "${ROOT}/${OBJECT_C_INPUT}" "${ROOT}/${OBJECT_C_RGBA}" 2>/dev/null || true

cd "${TS}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[C] mesh-exporter 导出 | trial=${MESH_TRIAL}"
echo "[C] texture=${TEXTURE_SIZE}"

EXPORT_ARGS=(
  --config "${MESH_TRIAL}/configs/parsed.yaml"
  --export --gpu "${GPU}"
  resume="${MESH_TRIAL}/ckpts/last.ckpt"
  system.exporter_type=mesh-exporter
  system.exporter.fmt=obj-mtl
  system.exporter.texture_size="${TEXTURE_SIZE}"
  system.exporter.texture_format=png
)
if [[ "${MESH_TRIAL}" != *magic123-refine-sd* && "${MESH_TRIAL}" != *magic123-hifa-refine-sd* ]]; then
  echo "[C] coarse 导出: threshold=${ISO_THRESHOLD} method=${ISO_METHOD} res=${ISO_RESOLUTION}"
  EXPORT_ARGS+=(
    "system.geometry.isosurface_threshold=${ISO_THRESHOLD}"
    "system.geometry.isosurface_method=${ISO_METHOD}"
    "system.geometry.isosurface_resolution=${ISO_RESOLUTION}"
    system.geometry.isosurface_remove_outliers=true
    "system.geometry.isosurface_outlier_n_faces_threshold=${OUTLIER_TH}"
  )
fi

"${PYTHON}" launch.py "${EXPORT_ARGS[@]}"

export_dir="$(find "${MESH_TRIAL}/save" -type d -name '*-export' 2>/dev/null | sort | tail -1)"
[[ -n "${export_dir}" ]] || { echo "[C] 导出失败"; exit 1; }

cp -f "${export_dir}"/model.* "${OUT}/"
cp -f "${export_dir}"/texture_* "${OUT}/" 2>/dev/null || true
if [[ -n "${BACKUP_NAME}" ]]; then
  cp -f "${OUT}/model.obj" "${OUT}/${BACKUP_NAME}"
fi
echo "[C] 导出完成 → ${OUT}/model.obj + texture_kd.png"
