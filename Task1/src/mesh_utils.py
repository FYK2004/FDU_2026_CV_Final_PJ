"""Blender 多视角渲染前的 mesh 加载与纹理采样。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _rebuild_mesh_with_obj_uv(mesh_path: Path, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray] | None:
    verts: list[list[float]] = []
    uvs_raw: list[list[float]] = []
    faces_v: list[list[int]] = []
    faces_vt: list[list[int]] = []

    for line in mesh_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs_raw.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            fv, fvt = [], []
            for part in line.split()[1:]:
                bits = part.split("/")
                fv.append(int(bits[0]) - 1)
                fvt.append(int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1)
            faces_v.append(fv)
            faces_vt.append(fvt)

    if not uvs_raw or not faces_v:
        return None

    verts_arr = np.asarray(verts, dtype=np.float64)
    uvs_arr = np.asarray(uvs_raw, dtype=np.float64)
    new_verts: list[np.ndarray] = []
    new_uvs: list[np.ndarray] = []
    new_faces: list[list[int]] = []
    corner_map: dict[tuple[int, int], int] = {}

    for fv, fvt in zip(faces_v, faces_vt):
        tri: list[int] = []
        for vi, ti in zip(fv, fvt):
            key = (vi, ti)
            if key not in corner_map:
                corner_map[key] = len(new_verts)
                new_verts.append(verts_arr[vi])
                new_uvs.append(uvs_arr[ti] if ti >= 0 else np.zeros(2))
            tri.append(corner_map[key])
        new_faces.append(tri)

    rebuilt = trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    return rebuilt, np.asarray(new_uvs, dtype=np.float64)


def _load_textured_mesh(mesh_path: Path) -> trimesh.Trimesh:
    from PIL import Image

    mesh_path = Path(mesh_path)
    mesh_dir = mesh_path.parent
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"无法加载三角 mesh: {mesh_path}")

    tex_path = mesh_dir / "texture_kd.png"
    visual = mesh.visual
    uv = getattr(visual, "uv", None) if isinstance(visual, trimesh.visual.texture.TextureVisuals) else None

    if uv is None:
        rebuilt = _rebuild_mesh_with_obj_uv(mesh_path, mesh)
        if rebuilt is not None:
            mesh, uv = rebuilt

    if tex_path.exists() and uv is not None:
        material = trimesh.visual.material.SimpleMaterial(image=Image.open(tex_path).convert("RGB"))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return mesh


def _sample_surface_colors(
    mesh: trimesh.Trimesh, points: np.ndarray, face_idx: np.ndarray
) -> np.ndarray:
    colors = np.full((len(points), 3), 0.65, dtype=np.float32)
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.texture.TextureVisuals) and visual.uv is not None:
        try:
            img = np.asarray(visual.material.image.convert("RGB"), dtype=np.float32) / 255.0
            uv = visual.uv
            faces = mesh.faces[face_idx]
            bc = trimesh.triangles.points_to_barycentric(mesh.vertices[faces], points)
            tri_uv = uv[faces]
            samp_uv = (tri_uv * bc[:, :, None]).sum(axis=1)
            u_px = np.clip((samp_uv[:, 0] * (img.shape[1] - 1)).astype(np.int32), 0, img.shape[1] - 1)
            v_px = np.clip(((1.0 - samp_uv[:, 1]) * (img.shape[0] - 1)).astype(np.int32), 0, img.shape[0] - 1)
            return img[v_px, u_px]
        except Exception:
            pass
    if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
        vc = np.asarray(visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        faces = mesh.faces[face_idx]
        bc = trimesh.triangles.points_to_barycentric(mesh.vertices[faces], points)
        return (vc[faces] * bc[:, :, None]).sum(axis=1)
    return colors
