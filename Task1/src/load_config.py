#!/usr/bin/env python3
"""将 task1.yaml 导出为 shell export 语句。"""
import sys
from pathlib import Path

import yaml


def main() -> None:
    cfg_path = Path(sys.argv[1])
    with open(cfg_path, encoding="utf-8") as f:
        c = yaml.safe_load(f)

    root = cfg_path.resolve().parent.parent
    ext = c["external"]
    oc = c["object_c"]
    fusion = c.get("fusion", {})

    lines = [
        f'export ROOT="{root}"',
        f'export EXTERNAL_THREESTUDIO="{root / ext["threestudio"]}"',
        f'export EXTERNAL_MAGIC123="{root / ext["magic123"]}"',
        f'export OBJECT_C_RGBA="{oc["image_rgba"]}"',
        f'export OBJECT_C_INPUT="{oc.get("image_input", oc["image_rgba"])}"',
        f'export OBJECT_C_PROMPT="{oc["prompt"]}"',
        f'export OBJECT_C_OUT="{oc["output_dir"]}"',
        f'export OBJECT_C_GPU="{oc["gpu"]}"',
        f'export OBJECT_C_COARSE_CONFIG="{oc.get("coarse_config", "configs/magic123-coarse-hifa.yaml")}"',
        f'export OBJECT_C_REFINE_CONFIG="{oc.get("refine_config", "configs/magic123-refine-hifa.yaml")}"',
        f'export OBJECT_C_SD_MODEL="{oc.get("sd_model", "runwayml/stable-diffusion-v1-5")}"',
        f'export OBJECT_C_COARSE_STEPS="{oc.get("coarse_steps", 5000)}"',
        f'export OBJECT_C_REFINE_STEPS="{oc.get("refine_steps", 5000)}"',
        f'export OBJECT_C_REPAIR_MODE="{oc.get("repair_mode", "off")}"',
        f'export OBJECT_C_TEXTURE_SIZE="{oc.get("texture_size", 2048)}"',
        f'export OBJECT_C_COARSE_EXPORT_THRESHOLD="{oc.get("coarse_export_threshold", "35.0")}"',
        f'export OBJECT_C_COARSE_EXPORT_METHOD="{oc.get("coarse_export_method", "mc-cpu")}"',
        f'export OBJECT_C_COARSE_EXPORT_RESOLUTION="{oc.get("coarse_export_resolution", 256)}"',
        f'export OBJECT_C_REFINE_EXPORT_THRESHOLD="{oc.get("refine_export_threshold", "25.0")}"',
        f'export OBJECT_C_REFINE_EXPORT_RESOLUTION="{oc.get("refine_export_resolution", 128)}"',
        f'export OBJECT_C_OUTLIER_THRESHOLD="{oc.get("outlier_threshold", "0.02")}"',
        f'export OBJECT_C_SEED="{oc.get("seed", 0)}"',
        f'export FUSION_OUT="{fusion.get("output_dir", "outputs/fusion")}"',
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
