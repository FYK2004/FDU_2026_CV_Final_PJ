#!/usr/bin/env python3
"""将 2DGS 点云 PLY 旋转到与 mesh 一致的直立朝向（Y 轴向上）。"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DGS = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(DGS))
sys.path.insert(0, str(ROOT / "src"))

from scene.gaussian_model import GaussianModel  # noqa: E402
from fusion_gaussian_utils import euler_xyz_matrix, matrix_to_quaternion_wxyz, quaternion_multiply_batch  # noqa: E402

# 训练 PLY 默认倒立；绕 X 轴 180° 翻转为直立（薄轴沿 Y，侧视而非底面）
DEFAULT_UPRIGHT_EULER = [180, 0, 0]


def rotate_gaussian_model(model: GaussianModel, euler_deg: list[float]) -> None:
    R = euler_xyz_matrix(euler_deg)
    R_t = torch.tensor(R, dtype=torch.float32, device=model._xyz.device)
    q_global = matrix_to_quaternion_wxyz(R)

    with torch.no_grad():
        model._xyz.copy_((model._xyz @ R_t.T).detach())
        model._rotation.copy_(quaternion_multiply_batch(q_global, model._rotation.detach()))


def fix_ply(
    ply_path: Path,
    euler_deg: list[float] | None = None,
    *,
    backup: bool = True,
    from_backup: bool = True,
) -> Path:
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    euler_deg = list(euler_deg or DEFAULT_UPRIGHT_EULER)
    bak = ply_path.with_suffix(ply_path.suffix + ".bak")

    if from_backup and bak.exists():
        shutil.copy2(bak, ply_path)
        print(f"从备份恢复 -> {ply_path}")
    elif backup and not bak.exists():
        shutil.copy2(ply_path, bak)
        print(f"备份 -> {bak}")

    model = GaussianModel(3)
    model.load_ply(str(ply_path))
    rotate_gaussian_model(model, euler_deg)
    model.save_ply(str(ply_path))
    print(f"已旋转 {ply_path}  rotation_euler={euler_deg}")
    return ply_path


def main() -> None:
    p = argparse.ArgumentParser(description="扶正 2DGS 点云 PLY（汉堡/熊）")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "task1.yaml")
    p.add_argument("--object", choices=["object_b", "object_c", "both"], default="both")
    p.add_argument("--iteration", type=int, default=10000)
    p.add_argument("--ply", type=Path, help="直接指定 PLY 路径")
    p.add_argument("--rotation", nargs=3, type=float, default=DEFAULT_UPRIGHT_EULER)
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--reset-fusion-rotation", action="store_true", help="将 task1.yaml 中 B/C 旋转重置为 [0,0,0]")
    args = p.parse_args()

    targets: list[Path] = []
    if args.ply:
        targets = [args.ply]
    else:
        import yaml

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        keys = ["object_b", "object_c"] if args.object == "both" else [args.object]
        for key in keys:
            ply = (
                ROOT
                / cfg[key]["output_dir"]
                / "train"
                / "point_cloud"
                / f"iteration_{args.iteration}"
                / "point_cloud.ply"
            )
            targets.append(ply)

    for ply in targets:
        fix_ply(ply, args.rotation, backup=not args.no_backup)

    if args.reset_fusion_rotation:
        import yaml

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        for key in ("object_b", "object_c"):
            cfg["fusion"]["placements"][key]["rotation_euler"] = [0, 0, 0]
        args.config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"已重置 {args.config} 中 object_b/object_c rotation_euler -> [0,0,0]")


if __name__ == "__main__":
    main()
