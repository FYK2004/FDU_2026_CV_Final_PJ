#!/usr/bin/env bash
# 汉堡 / 泰迪熊：mesh 多视角渲染 → 2DGS 训练（与 object_a 相同表示）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

source scripts/common.sh
load_config

ENV_NAME="${TASK1_2DGS_ENV:-llm-26}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

DGS="${ROOT}/external/2d-gaussian-splatting"

render_object() {
  local key="$1"
  local mesh="${ROOT}/outputs/${key}/model.obj"
  if [[ ! -f "${mesh}" ]]; then
    echo "[warn] 跳过 ${key}，缺少 mesh: ${mesh}"
    return 1
  fi
  echo "=== [1/2] 多视角渲染 ${key} ==="
  python src/render_mesh_colmap.py --config config/task1.yaml --object "${key}"
}

train_object() {
  local key="$1"
  local port="$2"
  local iter
  iter="$(python3 -c "import yaml; c=yaml.safe_load(open('config/task1.yaml')); print(c['${key}'].get('train_2dgs',{}).get('iterations',10000))")"
  local src="${ROOT}/outputs/${key}/colmap"
  local model="${ROOT}/outputs/${key}/train"
  if [[ ! -f "${src}/transforms_train.json" ]]; then
    echo "[warn] 跳过 ${key} 训练，缺少 ${src}/transforms_train.json"
    return 1
  fi
  if [[ ! -f "${src}/transforms_test.json" ]]; then
    python3 -c "import json; p='${src}/transforms_train.json'; d=json.load(open(p)); json.dump({'camera_angle_x':d['camera_angle_x'],'frames':[]}, open('${src}/transforms_test.json','w'), indent=2)"
  fi
  echo "=== [2/2] 2DGS 训练 ${key} (${iter} iter) ==="
  (
    cd "${DGS}"
    python train.py -s "${src}" -m "${model}" --iterations "${iter}" --white_background --port "${port}"
  )
}

SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

for key in object_b object_c; do
  if [[ "${SKIP_RENDER}" != "1" ]]; then
    render_object "${key}" || true
  fi
  if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ "${key}" == "object_b" ]]; then
    train_object "${key}" 6009 || true
  else
    train_object "${key}" 6010 || true
  fi
  fi
done

echo "=== 扶正点云朝向（与 model.obj 一致） ==="
for key in object_b object_c; do
  iter="$(python3 -c "import yaml; c=yaml.safe_load(open('config/task1.yaml')); print(c['${key}'].get('train_2dgs',{}).get('iterations',10000))")"
  ply="${ROOT}/outputs/${key}/train/point_cloud/iteration_${iter}/point_cloud.ply"
  if [[ -f "${ply}" ]]; then
    python src/fix_gaussian_ply_upright.py --object "${key}" --iteration "${iter}" --no-backup || true
  fi
done

echo "完成。运行融合: bash scripts/run_fusion_2dgs.sh"
