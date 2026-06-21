#!/usr/bin/env python3
"""
2DGS 原生场景融合：背景 + A/B/C 全部合并为高斯，一次光栅化渲染。

物体位姿由 config/task1.yaml 的 fusion.placements 指定。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from argparse import Namespace

ROOT = Path(__file__).resolve().parent.parent
DGS = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(DGS))
sys.path.insert(0, str(ROOT / "src"))

from gaussian_renderer import render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.render_utils import generate_path  # noqa: E402

from fusion_gaussian_utils import (  # noqa: E402
    apply_placement,
    crop_object_a_gaussians,
    crop_object_gaussians,
    gaussian_tensors_from_model,
    merge_gaussian_tensors,
    subset_gaussians,
)


def load_cfg(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_background_gaussians(bg_cfg: dict, iteration: int) -> tuple[GaussianModel, Scene, object]:
    source = ROOT / bg_cfg["data_root"] / bg_cfg["scene"]
    model_path = ROOT / bg_cfg["output_dir"] / bg_cfg["scene"] / "train"
    dataset = Namespace(
        sh_degree=3,
        source_path=str(source),
        model_path=str(model_path),
        images="images",
        resolution=-1,
        white_background=False,
        data_device="cuda",
        eval=False,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, depth_ratio=0.0, debug=False)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return gaussians, scene, pipe


def load_object_gaussians(obj_cfg: dict, iteration: int) -> GaussianModel:
    gaussians = GaussianModel(3)
    ply = (
        ROOT
        / obj_cfg["output_dir"]
        / "train"
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    if not ply.exists():
        raise FileNotFoundError(f"物体点云不存在: {ply}")
    gaussians.load_ply(str(ply))
    return gaussians


def trained_ply_path(cfg: dict, key: str, iteration: int) -> Path | None:
    obj_cfg = cfg[key]
    train_cfg = obj_cfg.get("train_2dgs", {})
    if train_cfg is not None and not train_cfg.get("enabled", True):
        return None
    ply = (
        ROOT
        / obj_cfg["output_dir"]
        / "train"
        / "point_cloud"
        / f"iteration_{iteration}"
        / "point_cloud.ply"
    )
    return ply if ply.exists() else None


def build_trained_object_tensors(
    cfg: dict,
    key: str,
    iteration: int,
    *,
    zero_position: bool = False,
) -> dict[str, torch.Tensor] | None:
    """加载 B/C 训练好的 2DGS，裁剪并放置（与 object_a 相同模式）。"""
    ply = trained_ply_path(cfg, key, iteration)
    if ply is None:
        return None
    obj_g = load_object_gaussians(cfg[key], iteration)
    center = obj_g.get_xyz.mean(dim=0)
    mask = crop_object_gaussians(obj_g, center)
    obj_sub = subset_gaussians(obj_g, mask)
    cropped = obj_sub["xyz"]
    local_extent = (cropped.max(dim=0).values - cropped.min(dim=0).values).max().item()
    local_extent = max(local_extent, 1e-3)
    placement = dict(cfg["fusion"]["placements"][key])
    if zero_position:
        placement["position"] = [0.0, 0.0, 0.0]
    print(
        f"{key}: 训练 2DGS {mask.sum().item()}/{mask.numel()} 高斯 "
        f"(local_extent={local_extent:.3f}, iter={iteration})"
    )
    return apply_placement(
        obj_sub,
        placement,
        center=center,
        normalize_extent=local_extent,
    )


def load_object_a_gaussians(obj_cfg: dict, iteration: int) -> GaussianModel:
    return load_object_gaussians(obj_cfg, iteration)


def build_object_a_tensors(cfg: dict, a_iter: int) -> dict[str, torch.Tensor]:
    a_g = load_object_a_gaussians(cfg["object_a"], a_iter)
    center = a_g.get_xyz.mean(dim=0)
    mask = crop_object_a_gaussians(a_g, center)
    a_sub = subset_gaussians(a_g, mask)
    cropped = a_sub["xyz"]
    local_extent = (cropped.max(dim=0).values - cropped.min(dim=0).values).max().item()
    local_extent = max(local_extent, 1e-3)
    print(
        f"物体 A: 保留 {mask.sum().item()}/{mask.numel()} 高斯 "
        f"(local_extent={local_extent:.3f})"
    )
    return apply_placement(
        a_sub,
        cfg["fusion"]["placements"]["object_a"],
        center=center,
        normalize_extent=local_extent,
    )


def prepare_fusion_cache(
    cfg: dict,
    bg_iter: int,
    a_iter: int,
    b_iter: int | None = None,
    c_iter: int | None = None,
) -> dict:
    """预加载背景 / A / B / C 高斯张量，供逐帧合并。"""
    bg_g, _, _ = load_background_gaussians(cfg["background"], bg_iter)
    sh_degree = bg_g.max_sh_degree
    bg_tensors = gaussian_tensors_from_model(bg_g)
    a_tf = build_object_a_tensors(cfg, a_iter)

    b_iter = b_iter if b_iter is not None else int(cfg["object_b"].get("train_2dgs", {}).get("iterations", 10000))
    c_iter = c_iter if c_iter is not None else int(cfg["object_c"].get("train_2dgs", {}).get("iterations", 10000))

    b_tf = build_trained_object_tensors(cfg, "object_b", b_iter)
    if b_tf is not None:
        print("object_b: 使用训练 2DGS")
    c_tf = build_trained_object_tensors(cfg, "object_c", c_iter)
    if c_tf is not None:
        print("object_c: 使用训练 2DGS")

    if b_tf is None or c_tf is None:
        missing = [k for k, t in [("object_b", b_tf), ("object_c", c_tf)] if t is None]
        raise FileNotFoundError(
            f"缺少 B/C 2DGS 点云，请先运行: bash scripts/run_object_bc_2dgs.sh ({', '.join(missing)})"
        )

    return {
        "sh_degree": sh_degree,
        "bg_tensors": bg_tensors,
        "a_tf": a_tf,
        "b_tf": b_tf,
        "c_tf": c_tf,
    }


def merged_gaussians_for_frame(cache: dict, frame_idx: int, cam=None) -> GaussianModel:
    parts = [cache["bg_tensors"], cache["a_tf"]]
    if cache["b_tf"] is not None:
        parts.append(cache["b_tf"])
    if cache["c_tf"] is not None:
        parts.append(cache["c_tf"])
    return merge_gaussian_tensors(parts, cache["sh_degree"])


def render_full_gaussian_fusion(
    cache: dict,
    scene: Scene,
    pipe,
    gs_dir: Path,
    final_dir: Path,
    num_frames: int,
    width: int | None,
    cameras_json: Path | None = None,
) -> None:
    """单遍 2DGS 渲染：背景 + A + B + C。"""
    gs_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    ref_cams = scene.getTrainCameras()
    traj = generate_path(ref_cams, n_frames=num_frames)
    if width is not None and width > 0:
        ref_w = ref_cams[0].image_width
        scale = width / ref_w
        for cam in traj:
            cam.image_width = int(round(cam.image_width * scale / 2) * 2)
            cam.image_height = int(round(cam.image_height * scale / 2) * 2)
            tanfovx = math.tan(cam.FoVx * 0.5)
            aspect = cam.image_width / cam.image_height
            cam.FoVy = 2 * math.atan(math.tan(cam.FoVx * 0.5) / aspect)

    if cameras_json is not None:
        export_cameras_json(traj, cameras_json)

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    for i, cam in enumerate(tqdm(traj, desc="2DGS 全场景融合")):
        gaussians = merged_gaussians_for_frame(cache, i, cam)
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg_color)
            img = pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            depth = pkg["surf_depth"].detach().squeeze(0).cpu().numpy()
        out_name = f"frame_{i:04d}.png"
        arr = (img * 255).astype(np.uint8)
        Image.fromarray(arr).save(final_dir / out_name)
        Image.fromarray(arr).save(gs_dir / out_name)
        np.save(gs_dir / f"frame_{i:04d}_depth.npy", depth.astype(np.float32))

    n_gauss = merged_gaussians_for_frame(cache, 0).get_xyz.shape[0]
    print(f"合并后高斯总数（含 B/C）: {n_gauss}")


def export_cameras_json(cameras, out_path: Path) -> None:
    records = []
    for i, cam in enumerate(cameras):
        w2v = cam.world_view_transform.detach().cpu().numpy().T
        c2w = np.linalg.inv(w2v)
        records.append(
            {
                "frame": i,
                "width": int(cam.image_width),
                "height": int(cam.image_height),
                "fovx": float(cam.FoVx),
                "fovy": float(cam.FoVy),
                "znear": float(cam.znear),
                "zfar": float(cam.zfar),
                "c2w": c2w.tolist(),
            }
        )
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="2DGS 场景融合（位姿来自 task1.yaml）")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "task1.yaml")
    p.add_argument("--bg-iteration", type=int, default=30000)
    p.add_argument("--a-iteration", type=int, default=10000)
    p.add_argument("--b-iteration", type=int, default=None, help="object_b 训练 2DGS iteration")
    p.add_argument("--c-iteration", type=int, default=None, help="object_c 训练 2DGS iteration")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--width", type=int, default=1280, help="输出宽度，0=原始分辨率")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    fusion = cfg["fusion"]
    b_iter = args.b_iteration
    if b_iter is None:
        b_iter = int(cfg["object_b"].get("train_2dgs", {}).get("iterations", 10000))
    c_iter = args.c_iteration
    if c_iter is None:
        c_iter = int(cfg["object_c"].get("train_2dgs", {}).get("iterations", 10000))

    out_root = ROOT / fusion["output_dir"]
    gs_dir = out_root / "gs_render"
    final_dir = out_root / "render_frames"
    cameras_json = out_root / "fusion_cameras.json"
    num_frames = args.num_frames or fusion["camera"]["num_frames"]
    width = args.width if args.width > 0 else None

    safe_state(True)

    def write_manifest() -> None:
        manifest = {
            "method": "2dgs_gaussian_merge_all",
            "placements_config": str(args.config.resolve()),
            "background_gaussians": str(
                ROOT / cfg["background"]["output_dir"] / cfg["background"]["scene"] / "train"
            ),
            "object_a_gaussians": str(ROOT / cfg["object_a"]["output_dir"] / "train"),
            "object_b_gaussians": str(ROOT / cfg["object_b"]["output_dir"] / "train"),
            "object_c_gaussians": str(ROOT / cfg["object_c"]["output_dir"] / "train"),
            "cameras_json": str(cameras_json),
            "render_output": str(final_dir.resolve()),
        }
        (out_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def encode_video() -> None:
        mp4 = out_root / "wander.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", "24",
            "-i", str(final_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4),
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"视频: {mp4}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[warn] ffmpeg 失败: {e}")

    _, scene, pipe = load_background_gaussians(cfg["background"], args.bg_iteration)
    cache = prepare_fusion_cache(cfg, args.bg_iteration, args.a_iteration, b_iter, c_iter)

    render_full_gaussian_fusion(
        cache, scene, pipe, gs_dir, final_dir, num_frames, width, cameras_json
    )
    write_manifest()
    encode_video()
    print(f"融合帧: {final_dir}")


if __name__ == "__main__":
    main()
