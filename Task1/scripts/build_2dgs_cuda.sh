#!/usr/bin/env bash
# 编译 2DGS CUDA 扩展（物体 A + 背景场景必需）
# 用法: bash scripts/build_2dgs_cuda.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DGS="${ROOT}/external/2d-gaussian-splatting"
ENV_NAME="${TASK1_2DGS_ENV:-task1-2dgs-cu12}"

# 磁盘满时：在 /tmp 编译（不写入 public 盘）
BUILD_ROOT="${TMPDIR:-/tmp}/task1-2dgs-build"
if df /inspire/hdd/project/fdu-aidake-cfff/public 2>/dev/null | tail -1 | grep -qE '100%|9[0-9]%'; then
  echo "[warn] public 盘空间不足，使用 ${BUILD_ROOT} 编译"
  mkdir -p "${BUILD_ROOT}/submodules"
  cp -a "${DGS}/submodules/diff-surfel-rasterization" "${BUILD_ROOT}/submodules/"
  cp -a "${DGS}/submodules/simple-knn" "${BUILD_ROOT}/submodules/"
  DGS_BUILD="${BUILD_ROOT}"
else
  DGS_BUILD="${DGS}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 清理之前失败安装留下的错误符号链接
rm -rf "${CONDA_PREFIX}/include/cub" \
       "${CONDA_PREFIX}/include/thrust" \
       "${CONDA_PREFIX}/include/cuda" 2>/dev/null || true

# 确保编译工具链与 PyTorch cu121 对齐
conda install -y -c nvidia cuda-nvcc=12.1 cuda-toolkit=12.1 cuda-cccl 2>/dev/null || true
pip install -q "setuptools<70" ninja wheel packaging

# CCCL 头文件路径（cub / thrust / cuda/std）
CCCL="${CONDA_PREFIX}/targets/x86_64-linux/include/cccl"
INC="${CONDA_PREFIX}/include"
mkdir -p "${INC}/cuda"
ln -sfn "${CCCL}/cub" "${INC}/cub"
ln -sfn "${CCCL}/thrust" "${INC}/thrust"
ln -sfn "${CCCL}/cuda/std" "${INC}/cuda/std"
ln -sfn "${CCCL}/cuda/__cccl_config" "${INC}/cuda/__cccl_config"

export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/include"
export CPLUS_INCLUDE_PATH="${CPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
# 编译中间文件写到 /tmp，避免 public 盘满（100% 时会报 Disk quota exceeded）
export TMPDIR="${TMPDIR:-/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-task1}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "nvcc: $(nvcc --version | grep release)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

cd "${DGS_BUILD}/submodules/diff-surfel-rasterization"
python setup.py install
cd "${DGS_BUILD}/submodules/simple-knn"
python setup.py install

python -c "from diff_surfel_rasterization import _C; import simple_knn._C; print('2DGS CUDA extensions OK')"
echo "环境 ${ENV_NAME} 已可用于 2DGS 训练"
