"""2DGS 高斯合并与变换工具（用于场景融合）。"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn


def euler_xyz_matrix(euler_deg: list[float]) -> np.ndarray:
    rx, ry, rz = [math.radians(a) for a in euler_deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def matrix_to_quaternion_wxyz(R: np.ndarray) -> torch.Tensor:
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = torch.tensor([w, x, y, z], dtype=torch.float32, device="cuda")
    return q / torch.linalg.norm(q)


def quaternion_multiply_batch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """q1: (4,), q2: (N, 4) wxyz -> (N, 4)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def crop_object_gaussians(
    model,
    center: torch.Tensor,
    radius: float = 0.65,
) -> torch.Tensor:
    """保留物体中心附近的高斯。"""
    xyz = model.get_xyz
    dist = torch.norm(xyz - center, dim=1)
    return dist < radius


def crop_object_a_gaussians(model, center: torch.Tensor) -> torch.Tensor:
    """保留物体中心附近、且非纯蓝背景的高斯。"""
    mask = crop_object_gaussians(model, center, radius=0.65)

    # 去掉 COLMAP 拍摄蓝底上的背景高斯（B 明显高于 R/G）
    dc = model._features_dc[:, 0, :]
    c0 = 0.28209479177387814
    rgb = c0 * dc + 0.5
    blue_bg = (rgb[:, 2] > rgb[:, 0] + 0.08) & (rgb[:, 2] > rgb[:, 1] + 0.08)
    return mask & (~blue_bg)


def subset_gaussians(model, mask: torch.Tensor):
    """返回裁剪后的参数字典（不含 optimizer）。"""
    return {
        "xyz": model._xyz[mask].detach(),
        "f_dc": model._features_dc[mask].detach(),
        "f_rest": model._features_rest[mask].detach(),
        "opacity": model._opacity[mask].detach(),
        "scaling": model._scaling[mask].detach(),
        "rotation": model._rotation[mask].detach(),
    }


def _placement_rotation_matrix(placement: dict[str, Any]) -> np.ndarray:
    """placement 旋转；含 mesh_pre_rotation 时合并为一次旋转（位置与朝向一致）。"""
    R = euler_xyz_matrix(placement["rotation_euler"])
    pre = placement.get("mesh_pre_rotation")
    if pre is not None:
        pre = [float(x) for x in pre]
        if not all(abs(x) < 1e-9 for x in pre):
            R = R @ euler_xyz_matrix(pre)
    return R


def apply_placement(
    tensors: dict[str, torch.Tensor],
    placement: dict[str, Any],
    center: torch.Tensor,
    normalize_extent: float | None = None,
) -> dict[str, torch.Tensor]:
    """中心化、可选归一化，再按 placement 变换高斯。"""
    xyz = tensors["xyz"] - center
    log_norm = 0.0
    if normalize_extent is not None and normalize_extent > 1e-6:
        xyz = xyz / normalize_extent
        log_norm = -math.log(normalize_extent)

    R = _placement_rotation_matrix(placement)
    R_t = torch.tensor(R, dtype=torch.float32, device="cuda")
    s = float(placement.get("scale", 1.0))
    t = torch.tensor(placement["position"], dtype=torch.float32, device="cuda")

    xyz_new = (xyz @ R_t.T) * s + t
    q_global = matrix_to_quaternion_wxyz(R)

    rots = tensors["rotation"]
    q_new = quaternion_multiply_batch(q_global, rots)
    log_s = math.log(max(s, 1e-6)) + log_norm
    scaling_new = tensors["scaling"] + log_s

    out = dict(tensors)
    out["xyz"] = xyz_new
    out["rotation"] = q_new
    out["scaling"] = scaling_new
    return out


def translate_gaussians(
    tensors: dict[str, torch.Tensor],
    position: list[float] | np.ndarray,
) -> dict[str, torch.Tensor]:
    """平移高斯（用于 object_c 逐帧放置，局部张量原点即几何中心）。"""
    t = torch.tensor(position, dtype=torch.float32, device=tensors["xyz"].device)
    out = dict(tensors)
    out["xyz"] = tensors["xyz"] + t
    return out


def nudge_gaussians_toward_camera(
    tensors: dict[str, torch.Tensor],
    cam_center: torch.Tensor,
    delta: float,
) -> dict[str, torch.Tensor]:
    """沿相机方向平移高斯（delta>0 靠近相机），避免被椅面遮挡。"""
    if abs(delta) < 1e-6:
        return tensors
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
    xyz = out["xyz"]
    cc = cam_center.to(device=xyz.device, dtype=xyz.dtype).reshape(1, 3)
    direction = cc - xyz
    direction = direction / torch.norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    out["xyz"] = xyz + float(delta) * direction
    return out


def nudge_gaussians_depth_preserve_uv(
    tensors: dict[str, torch.Tensor],
    world_view_transform: torch.Tensor,
    delta: float,
) -> dict[str, torch.Tensor]:
    """相机空间减小深度（靠近相机），保持屏幕投影近似不变。"""
    if abs(delta) < 1e-6:
        return tensors
    w2c = world_view_transform
    xyz = tensors["xyz"]
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    hom = torch.cat([xyz, ones], dim=1)
    cam = hom @ w2c
    z = cam[:, 2:3].clamp_min(1e-4)
    s = (z - float(delta)) / z
    cam_new = torch.cat([cam[:, 0:1] * s, cam[:, 1:2] * s, z - float(delta), cam[:, 3:4]], dim=1)
    c2w = torch.linalg.inv(w2c)
    world = cam_new @ c2w
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
    out["xyz"] = world[:, :3]
    return out


def boost_opacity(tensors: dict[str, torch.Tensor], log_boost: float) -> dict[str, torch.Tensor]:
    """提高高斯不透明度（log 域加法），使 mesh 物体不易被背景遮挡。"""
    if abs(log_boost) < 1e-6:
        return tensors
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
    out["opacity"] = tensors["opacity"] + float(log_boost)
    return out


def gaussian_tensors_from_model(model) -> dict[str, torch.Tensor]:
    return {
        "xyz": model._xyz.detach(),
        "f_dc": model._features_dc.detach(),
        "f_rest": model._features_rest.detach(),
        "opacity": model._opacity.detach(),
        "scaling": model._scaling.detach(),
        "rotation": model._rotation.detach(),
    }


def merge_gaussian_tensors(list_of_tensors: list[dict[str, torch.Tensor]], sh_degree: int):
    from scene.gaussian_model import GaussianModel

    merged = GaussianModel(sh_degree)
    merged.active_sh_degree = sh_degree
    merged._xyz = nn.Parameter(torch.cat([t["xyz"] for t in list_of_tensors], dim=0))
    merged._features_dc = nn.Parameter(torch.cat([t["f_dc"] for t in list_of_tensors], dim=0))
    merged._features_rest = nn.Parameter(
        torch.cat([t["f_rest"] for t in list_of_tensors], dim=0)
    )
    merged._opacity = nn.Parameter(torch.cat([t["opacity"] for t in list_of_tensors], dim=0))
    merged._scaling = nn.Parameter(torch.cat([t["scaling"] for t in list_of_tensors], dim=0))
    merged._rotation = nn.Parameter(torch.cat([t["rotation"] for t in list_of_tensors], dim=0))
    return merged
