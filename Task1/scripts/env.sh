#!/usr/bin/env bash
task1_threestudio_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if command -v conda &>/dev/null; then
    local base py
    base="$(conda info --base 2>/dev/null)" || true
    if [[ -n "${base}" ]]; then
      py="${base}/envs/task1-threestudio/bin/python"
      if [[ -x "${py}" ]]; then
        echo "${py}"
        return
      fi
    fi
  fi
  echo python3
}
