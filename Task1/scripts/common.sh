#!/usr/bin/env bash
# 从 task1.yaml 导出 shell 变量（需 python3 + pyyaml）
load_config() {
  eval "$(python3 "${ROOT}/src/load_config.py" "${ROOT}/config/task1.yaml")"
}
