#!/usr/bin/env bash
# 修复 2DGS 子模块（GitLab simple-knn 不可达时使用 GitHub 镜像）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DGS="${ROOT}/external/2d-gaussian-splatting"

[[ -d "${DGS}" ]] || exit 0

if [[ ! -f "${DGS}/submodules/simple-knn/setup.py" ]]; then
  echo "[fix] simple-knn <- camenduru/simple-knn"
  rm -rf "${DGS}/submodules/simple-knn"
  git clone --depth 1 https://github.com/camenduru/simple-knn.git "${DGS}/submodules/simple-knn"
fi

if [[ ! -f "${DGS}/submodules/diff-surfel-rasterization/setup.py" ]]; then
  echo "[fix] diff-surfel-rasterization"
  rm -rf "${DGS}/submodules/diff-surfel-rasterization"
  git clone --recursive --depth 1 https://github.com/hbb1/diff-surfel-rasterization.git \
    "${DGS}/submodules/diff-surfel-rasterization"
fi
