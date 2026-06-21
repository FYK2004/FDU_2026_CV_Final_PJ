#!/usr/bin/env python3
"""将 LeRobot v3 图像数据集转为 MP4 视频格式，加速后续训练。"""
from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from pathlib import Path

import yaml

# 作业实际用到：Step1 splitB，Step2 splitABC_merged，Step3 评测 splitD
DEFAULT_SPLITS = ["splitB", "splitABC_merged", "splitD"]


def patch_lerobot_video_convert() -> None:
    """CALVIN 使用 image/wrist_image 列名，需扩展 LeRobot 的图像键检测。"""
    import lerobot.datasets.dataset_tools as dt

    if getattr(dt, "_calvin_video_patched", False):
        return

    src = inspect.getsource(dt.convert_image_to_video_dataset)
    old = "img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]"
    new = (
        "img_keys = [key for key, ft in dataset.meta.features.items() if ft.get('dtype') == 'image']\n"
        "    if not img_keys:\n"
        "        img_keys = [key for key in hf_dataset.features if key.startswith(OBS_IMAGE)]"
    )
    if old not in src:
        print("警告: LeRobot convert_image_to_video 源码已变，跳过 patch", file=sys.stderr)
        return

    ns = dict(dt.__dict__)
    exec(compile(src.replace(old, new), "convert_image_to_video_dataset", "exec"), ns)
    dt.convert_image_to_video_dataset = ns["convert_image_to_video_dataset"]
    dt._calvin_video_patched = True


def convert_one(
    data_root: Path,
    split_name: str,
    *,
    num_workers: int,
    delete_source: bool,
    vcodec: str,
    crf: int,
) -> Path | None:
    from lerobot.datasets.dataset_tools import convert_image_to_video_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = data_root / split_name
    dst_name = f"{split_name}_video"
    dst = data_root / dst_name

    info = src / "meta" / "info.json"
    if not info.exists():
        print(f"  [skip] 缺少 {info}")
        return None

    import json

    meta = json.loads(info.read_text())
    if meta.get("codebase_version") != "v3.0":
        print(f"  [skip] {split_name} 不是 v3.0")
        return None

    has_image = any(ft.get("dtype") == "image" for ft in meta.get("features", {}).values())
    if not has_image:
        print(f"  [skip] {split_name} 无 image 特征（可能已是 video）")
        return None

    if (dst / "meta" / "info.json").exists():
        print(f"  [skip] {dst_name} 已存在")
        return dst

    print(f"  转换 {split_name} -> {dst_name} ...")
    dataset = LeRobotDataset(split_name, root=src)
    convert_image_to_video_dataset(
        dataset=dataset,
        output_dir=dst,
        repo_id=dst_name,
        vcodec=vcodec,
        crf=crf,
        num_workers=num_workers,
        fast_decode=1,
    )
    print(f"  完成: {dst}")

    if delete_source and src.exists():
        print(f"  删除原图像数据集: {src}")
        shutil.rmtree(src)
        dst.rename(src)
        renamed = data_root / split_name
        print(f"  视频数据集已替换为: {renamed}")
        return renamed

    return dst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--vcodec", default="h264", help="LeRobot 支持 h264 / libsvtav1 / h264_nvenc 等")
    p.add_argument("--crf", type=int, default=23)
    p.add_argument(
        "--replace-source",
        action="store_true",
        help="转换成功后删除原 parquet 图像版并改用原目录名（省磁盘）",
    )
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = root / cfg["dataset"]["root"]
    patch_lerobot_video_convert()

    for name in args.splits:
        convert_one(
            data_root,
            name,
            num_workers=args.num_workers,
            delete_source=args.replace_source,
            vcodec=args.vcodec,
            crf=args.crf,
        )

    print("视频转换完成。")


if __name__ == "__main__":
    main()
