#!/usr/bin/env bash
# 克隆题目一依赖的第三方仓库（不自动安装 CUDA 扩展）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="${ROOT}/external"
mkdir -p "${EXT}"

clone_if_missing() {
  local url="$1" dir="$2" extra="${3:-}"
  if [[ -d "${dir}/.git" ]]; then
    echo "[skip] ${dir} 已存在"
  else
    echo "[clone] ${url} -> ${dir}"
    git clone --recursive ${extra} "${url}" "${dir}"
  fi
}

clone_if_missing "https://github.com/hbb1/2d-gaussian-splatting.git" \
  "${EXT}/2d-gaussian-splatting" "--recursive"

# GitLab simple-knn 常超时，用镜像补全
bash "$(dirname "$0")/fix_submodules.sh"

clone_if_missing "https://github.com/threestudio-project/threestudio.git" \
  "${EXT}/threestudio"

clone_if_missing "https://github.com/guochengqian/Magic123.git" \
  "${EXT}/Magic123"

echo ""
echo "=== 下一步（需 GPU）==="
echo "2DGS:   cd ${EXT}/2d-gaussian-splatting && conda env create -f environment.yml"
echo "        conda activate surfel_splatting"
echo "threestudio: 见 ${EXT}/threestudio/README.md"
echo "Magic123:    见 ${EXT}/Magic123/README.md"
