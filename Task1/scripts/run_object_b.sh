#!/usr/bin/env bash
# 物体 B：threestudio DreamFusion SDS → obj+UV mesh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh
source scripts/env.sh

PYTHON="$(task1_threestudio_python)"
TS="${EXTERNAL_THREESTUDIO}"

eval "$(python3 -c "
import yaml
from pathlib import Path
b = yaml.safe_load(open('${ROOT}/config/task1.yaml'))['object_b']
print(f\"PROMPT={b['prompt']!r}\")
print(f'MAX_STEPS={b[\"max_steps\"]}')
print(f'GPU={b[\"gpu\"]}')
print(f'OUT={Path(\"${ROOT}\") / b[\"output_dir\"]}')
print(f'CFG={Path(\"${ROOT}\") / b[\"config\"]}')
")"

mkdir -p "${OUT}" "${TS}/configs"
cp -f "${CFG}" "${TS}/configs/" 2>/dev/null || true

cd "${TS}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

echo "[B] DreamFusion | prompt=${PROMPT} steps=${MAX_STEPS}"
"${PYTHON}" launch.py --config "$(basename "${CFG}")" --train --gpu "${GPU}" \
  system.prompt_processor.prompt="${PROMPT}" \
  trainer.max_steps="${MAX_STEPS}"

TRIAL="$(ls -td outputs/dreamfusion-sd/* 2>/dev/null | head -1)"
[[ -n "${TRIAL}" && -f "${TRIAL}/ckpts/last.ckpt" ]] || { echo "[B] 训练失败"; exit 1; }

echo "[B] 导出 mesh"
"${PYTHON}" launch.py --config "${TRIAL}/configs/parsed.yaml" --export --gpu "${GPU}" \
  resume="${TRIAL}/ckpts/last.ckpt" \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj-mtl \
  system.exporter.texture_size=2048 \
  system.exporter.texture_format=png

EXPORT_DIR="$(find "${TRIAL}/save" -type d -name '*-export' 2>/dev/null | sort | tail -1)"
[[ -n "${EXPORT_DIR}" ]] || { echo "[B] 导出失败"; exit 1; }
cp -f "${EXPORT_DIR}"/model.* "${OUT}/"
cp -f "${EXPORT_DIR}"/texture_* "${OUT}/" 2>/dev/null || true
echo "[B] 完成 → ${OUT}/model.obj"
