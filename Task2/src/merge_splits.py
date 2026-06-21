#!/usr/bin/env python3
"""将 splitA、splitB、splitC 合并为 LeRobot v3 数据集 splitABC_merged。"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

SPLIT_NAMES = ["splitA", "splitB", "splitC"]
MERGED_NAME = "splitABC_merged"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--force", action="store_true", help="删除已有合并结果并重新合并")
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = root / cfg["dataset"]["root"]
    out_dir = data_root / MERGED_NAME

    for name in SPLIT_NAMES:
        info = data_root / name / "meta" / "info.json"
        if not info.exists():
            print(f"缺少 {info}，请先 download_splits.py + convert_splits.py", file=sys.stderr)
            sys.exit(1)

    if out_dir.exists():
        if args.force:
            print(f"删除已有合并目录: {out_dir}")
            shutil.rmtree(out_dir)
        elif (out_dir / "meta" / "info.json").exists():
            info = json.loads((out_dir / "meta" / "info.json").read_text())
            print(f"[skip] {MERGED_NAME} 已存在: {info['total_episodes']} episodes")
            return

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = [
        LeRobotDataset(name, root=data_root / name) for name in SPLIT_NAMES
    ]
    print(f"合并 {SPLIT_NAMES} -> {out_dir} ...")
    merged = merge_datasets(
        datasets,
        output_repo_id=MERGED_NAME,
        output_dir=out_dir,
    )
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    print(
        f"  完成: {merged.meta.total_episodes} episodes, "
        f"{merged.meta.total_frames} frames -> {out_dir}"
    )
    print(f"  scene: {info.get('scene', 'A+B+C')}")


if __name__ == "__main__":
    main()
