#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
source scripts/common.sh
PYTHON="$(task2_python)"
"${PYTHON}" src/eval_zero_shot.py
echo "评测完成: outputs/eval/zero_shot_env_d/summary.json"
