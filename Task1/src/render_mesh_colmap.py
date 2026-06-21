#!/usr/bin/env python3
"""围绕 AIGC mesh 渲染多视角 RGBA，写出 Blender/NeRF 格式数据集供 2DGS 训练。"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mesh_utils import _load_textured_mesh, _sample_surface_colors  # noqa: E402

DGS = ROOT / "external" / "2d-gaussian-splatting"
sys.path.insert(0, str(DGS))
from scene.dataset_readers import storePly  # noqa: E402


def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def orbit_c2w(azimuth: float, elevation: float, radius: float) -> np.ndarray:
    """OpenGL/Blender 相机 c2w：Y 向上，相机 -Z 为观察方向。"""
    ce = math.cos(elevation)
    se = math.sin(elevation)
    ca = math.cos(azimuth)
    sa = math.sin(azimuth)
    cam_pos = np.array([radius * ce * sa, radius * se, radius * ce * ca], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-9
    up_c = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_c
    c2w[:3, 2] = -forward
    c2w[:3, 3] = cam_pos
    return c2w


def generate_views(
    num_azimuth: int,
    elevations_deg: list[float],
    radius: float,
) -> list[np.ndarray]:
    views: list[np.ndarray] = []
    for elev_deg in elevations_deg:
        elev = math.radians(elev_deg)
        for i in range(num_azimuth):
            az = 2.0 * math.pi * i / num_azimuth
            views.append(orbit_c2w(az, elev, radius))
    return views


def write_points3d_ply(mesh_path: Path, out_ply: Path, num_points: int = 8000) -> None:
    mesh = _load_textured_mesh(mesh_path)
    center = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices - center
    count = min(num_points, max(2000, len(mesh.faces) * 2))
    points, face_idx = trimesh.sample.sample_surface(mesh, count)
    colors = _sample_surface_colors(mesh, points, face_idx)
    rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    storePly(str(out_ply), points.astype(np.float64), rgb_u8)


BLENDER_RENDER_TEMPLATE = r'''import bpy, json, math, os
from mathutils import Matrix, Vector
from pathlib import Path

PAYLOAD = json.loads(Path(r"__PAYLOAD_PATH__").read_text(encoding="utf-8"))

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def import_obj(path, name):
    try:
        bpy.ops.wm.obj_import(filepath=path)
    except Exception:
        bpy.ops.import_scene.obj(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
    return obj

def ensure_material(obj, mesh_path):
    mesh_dir = os.path.dirname(mesh_path)
    tex_path = os.path.join(mesh_dir, "texture_kd.png")
    mat = bpy.data.materials.new(name=obj.name + "_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf and os.path.exists(tex_path):
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(tex_path, check_existing=True)
        mat.node_tree.links.new(bsdf.inputs["Base Color"], tex_node.outputs["Color"])
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def setup_lights():
    for old in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(old, do_unlink=True)
    bpy.ops.object.light_add(type="SUN", location=(2, 2, 4))
    sun = bpy.context.active_object
    sun.data.energy = 3.0

def setup_render(w, h):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = w
    sc.render.resolution_y = h
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.eevee.taa_render_samples = 8

def main():
    clear_scene()
    payload = PAYLOAD
    mesh_path = payload["mesh"]
    out_dir = Path(payload["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = import_obj(mesh_path, "asset")
    ensure_material(obj, mesh_path)
    setup_lights()
    w, h = payload["width"], payload["height"]
    fovy = payload["fovy"]
    setup_render(w, h)
    for i, frame in enumerate(payload["frames"]):
        cam_data = bpy.data.cameras.new("Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam_data.type = "PERSP"
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.angle = fovy
        c2w = frame["transform_matrix"]
        cam.matrix_world = Matrix(c2w)
        bpy.context.scene.render.filepath = str(out_dir / frame["file_name"])
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam, do_unlink=True)
        print("render", i, frame["file_name"])

if __name__ == "__main__":
    main()
'''


def run_blender_render(script: Path) -> None:
    cmd = ["xvfb-run", "-a", "blender", "-b", "-P", str(script)]
    print("Blender 多视角:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_dataset(
    mesh_path: Path,
    out_dir: Path,
    train_cfg: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = int(train_cfg.get("resolution", 512))
    num_az = int(train_cfg.get("num_azimuth", 24))
    elevs = [float(x) for x in train_cfg.get("elevations_deg", [-15.0, 15.0, 45.0])]
    radius_scale = float(train_cfg.get("radius_scale", 2.8))
    fov_deg = float(train_cfg.get("fov_deg", 50.0))

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    center = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    extent = float((mesh.bounds[1] - mesh.bounds[0]).max())
    radius = max(extent * radius_scale, 0.5)

    views = generate_views(num_az, elevs, radius)
    frames = []
    for i, c2w in enumerate(views):
        name = f"r_{i:04d}.png"
        frames.append(
            {
                "file_path": f"./r_{i:04d}",
                "file_name": name,
                "transform_matrix": c2w.tolist(),
            }
        )

    payload = {
        "mesh": str(mesh_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "width": res,
        "height": res,
        "fovy": math.radians(fov_deg),
        "frames": frames,
    }
    payload_path = out_dir / "blender_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    script = out_dir / "_blender_render.py"
    script.write_text(
        BLENDER_RENDER_TEMPLATE.replace("__PAYLOAD_PATH__", str(payload_path.resolve())),
        encoding="utf-8",
    )
    run_blender_render(script)

    camera_angle_x = 2.0 * math.atan(math.tan(math.radians(fov_deg) * 0.5) * 1.0)
    transforms = {
        "camera_angle_x": camera_angle_x,
        "frames": [{"file_path": f["file_path"], "transform_matrix": f["transform_matrix"]} for f in frames],
    }
    (out_dir / "transforms_train.json").write_text(
        json.dumps(transforms, indent=2), encoding="utf-8"
    )
    (out_dir / "transforms_test.json").write_text(
        json.dumps({"camera_angle_x": camera_angle_x, "frames": []}, indent=2),
        encoding="utf-8",
    )

    write_points3d_ply(mesh_path, out_dir / "points3d.ply")
    print(f"数据集: {out_dir} ({len(frames)} 视角, radius={radius:.3f})")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="mesh 多视角渲染 → 2DGS 训练数据集")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "task1.yaml")
    p.add_argument("--object", choices=("object_b", "object_c"), required=True)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    obj_cfg = cfg[args.object]
    mesh = ROOT / obj_cfg["output_dir"] / "model.obj"
    if not mesh.exists():
        raise FileNotFoundError(f"缺少 mesh: {mesh}")

    train_cfg = obj_cfg.get("train_2dgs", {})
    out_dir = ROOT / obj_cfg["output_dir"] / "colmap"
    build_dataset(mesh, out_dir, train_cfg)


if __name__ == "__main__":
    main()
