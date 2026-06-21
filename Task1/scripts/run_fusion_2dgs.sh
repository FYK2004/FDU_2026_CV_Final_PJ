#!/usr/bin/env bash
# 2DGS 原生场景融合（背景 + A/B/C 全高斯合并）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

source scripts/common.sh
load_config

ENV_NAME="${TASK1_2DGS_ENV:-llm-26}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python src/compose_fusion_2dgs.py --config config/task1.yaml "$@"
