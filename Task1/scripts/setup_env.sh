#!/usr/bin/env bash
# Task1 一键环境安装（conda 环境 + CUDA 扩展 + 辅助 pip）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "========== Task1 环境安装 =========="

# 1) 克隆/修复第三方仓库
bash scripts/setup.sh
bash scripts/fix_submodules.sh

DGS="${ROOT}/external/2d-gaussian-splatting"
TS="${ROOT}/external/threestudio"
M123="${ROOT}/external/Magic123"

# 2) task1-tools：COLMAP + 辅助脚本
if ! conda env list | grep -qE '^task1-tools '; then
  echo "[conda] 创建 task1-tools ..."
  conda env create -f environment/task1-tools.yml
else
  echo "[conda] task1-tools 已存在"
fi
conda run -n task1-tools pip install -q -r requirements.txt

# 3) task1-2dgs：2D Gaussian Splatting
if ! conda env list | grep -qE '^task1-2dgs '; then
  echo "[conda] 创建 task1-2dgs（约 10–20 分钟）..."
  conda env create -f environment/task1-2dgs.yml
else
  echo "[conda] task1-2dgs 已存在"
fi
echo "[pip] 编译安装 2DGS CUDA 扩展..."
bash scripts/build_2dgs_cuda.sh || {
  echo "[warn] 2DGS CUDA 扩展编译失败，请手动: conda activate task1-2dgs-cu12 && bash scripts/build_2dgs_cuda.sh"
}

# 4) task1-threestudio：文本→3D（物体 B）
if ! conda env list | grep -qE '^task1-threestudio '; then
  echo "[conda] 创建 task1-threestudio ..."
  conda create -y -n task1-threestudio python=3.10 pip
fi
echo "[pip] 安装 threestudio 依赖（较久）..."
conda run -n task1-threestudio pip install -q --upgrade pip
conda run -n task1-threestudio pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n task1-threestudio pip install -q ninja
conda run -n task1-threestudio bash -c "cd '${TS}' && pip install -q -e . && pip install -q -r requirements.txt" || {
  echo "[warn] threestudio 部分依赖可能失败，可稍后手动: conda activate task1-threestudio && cd external/threestudio && pip install -e ."
}

# 5) task1-magic123：单图→3D（物体 C，与 Magic123 共用 Stable-DreamFusion 栈）
if ! conda env list | grep -qE '^task1-magic123 '; then
  echo "[conda] 创建 task1-magic123 ..."
  conda create -y -n task1-magic123 python=3.10 pip
fi
echo "[pip] 安装 Magic123 基础依赖..."
conda run -n task1-magic123 pip install -q --upgrade pip
conda run -n task1-magic123 pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda run -n task1-magic123 bash -c "cd '${M123}' && pip install -q -r requirements.txt" || {
  echo "[warn] Magic123 依赖可能需手动补全 nvdiffrast 等，见 external/Magic123/README.md"
}

# 6) 验证
echo ""
echo "========== 环境验证 =========="
conda run -n task1-tools python -c "import yaml, cv2; print('task1-tools OK')"
conda run -n task1-tools colmap -h 2>&1 | head -1 || echo "colmap: check manually"
conda run -n task1-2dgs python -c "import torch; print('task1-2dgs torch', torch.__version__, 'cuda', torch.cuda.is_available())"
conda run -n task1-threestudio python -c "import torch; print('task1-threestudio torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || true
conda run -n task1-magic123 python -c "import torch; print('task1-magic123 torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || true

cat > "${ROOT}/.envrc.example" << 'EOF'
# 使用前: conda activate <env>
# 物体 A / 背景: conda activate task1-2dgs
# 物体 B:         conda activate task1-threestudio && cd external/threestudio
# 物体 C:         conda activate task1-magic123 && cd external/Magic123
# 工具脚本:       conda activate task1-tools && cd final_pj/Task1
EOF

echo ""
echo "安装完成。环境说明见 README.md「环境」一节。"
