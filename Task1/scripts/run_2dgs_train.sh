#!/usr/bin/env bash
# COLMAP + 2DGS 训练（物体 A / 背景 bicycle）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

KEY="${1:?用法: run_2dgs_train.sh object_a|background}"
source scripts/common.sh

ENV_NAME="${TASK1_2DGS_ENV:-task1-2dgs}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

DGS="${ROOT}/external/2d-gaussian-splatting"
eval "$(python3 -c "
import yaml
from pathlib import Path
c = yaml.safe_load(open('${ROOT}/config/task1.yaml'))
key = '${KEY}'
if key == 'background':
    bg = c['background']
    print(f'SOURCE={Path(\"${ROOT}\") / bg[\"data_root\"] / bg[\"scene\"]}')
    print(f'MODEL={Path(\"${ROOT}\") / bg[\"output_dir\"] / bg[\"scene\"] / \"train\"}')
    print(f'ITERS={bg[\"train\"][\"iterations\"]}')
else:
    o = c[key]
    print(f'SOURCE={Path(\"${ROOT}\") / o[\"output_dir\"] / \"source\"}')
    print(f'MODEL={Path(\"${ROOT}\") / o[\"output_dir\"] / \"train\"}')
    print(f'ITERS={o[\"train\"][\"iterations\"]}')
    print(f'IMAGES={Path(\"${ROOT}\") / o[\"images_dir\"]}')
")"

if [[ "${KEY}" == "object_a" ]]; then
  mkdir -p "${SOURCE}/input"
  find "${SOURCE}/input" -type l -delete 2>/dev/null || true
  shopt -s nullglob
  for f in "${IMAGES}"/*; do
    ln -sf "$(realpath "${f}")" "${SOURCE}/input/$(basename "${f}")"
  done
  if [[ ! -d "${SOURCE}/sparse" ]]; then
    echo "=== COLMAP: ${KEY} ==="
    (cd "${DGS}" && python convert.py -s "${SOURCE}")
  fi
fi

if [[ ! -d "${SOURCE}/sparse" ]]; then
  echo "缺少 COLMAP sparse: ${SOURCE}" >&2
  exit 1
fi

echo "=== 2DGS 训练 ${KEY} (${ITERS} iters) ==="
(
  cd "${DGS}"
  python train.py -s "${SOURCE}" -m "${MODEL}" --iterations "${ITERS}" --port 6011
)
echo "完成 → ${MODEL}/point_cloud/iteration_${ITERS}/"
