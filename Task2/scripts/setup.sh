#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if ! conda env list | grep -qE '^task2-lerobot '; then
  CONDA_SOLVER=classic CONDA_NO_PLUGINS=true conda create -y -n task2-lerobot python=3.10 pip
fi

conda run -n task2-lerobot pip install -q --upgrade pip
conda run -n task2-lerobot pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
conda run -n task2-lerobot pip install -q -r requirements.txt

echo "完成。使用: conda activate task2-lerobot"
echo "下一步: python src/prepare_splits.py && bash scripts/train_env_b.sh"
