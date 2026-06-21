#!/usr/bin/env python3
"""恢复/同步 split meta：parquet 列名保持 image/actions，仅 stats 键与 ACT 对齐。"""
import argparse
import json
from pathlib import Path

import yaml

# parquet 列名（勿改 info.json features）
PARQUET_KEYS = ("image", "wrist_image", "state", "actions")
# stats / 策略使用的标准名
STATS_RENAME = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action",
}


def restore_info_features(split_dir: Path) -> None:
    """若 info.json 被误改，从 parquet 元数据恢复原始列名。"""
    info_path = split_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    if "image" in feats:
        return
    reverse = {v: k for k, v in STATS_RENAME.items()}
    info["features"] = {reverse.get(k, k): v for k, v in feats.items()}
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def fix_stats(split_dir: Path) -> None:
    stats_path = split_dir / "meta" / "stats.json"
    if not stats_path.exists():
        return
    stats = json.loads(stats_path.read_text())
    new_stats = {}
    for k, v in stats.items():
        new_stats[STATS_RENAME.get(k, k)] = v
    stats_path.write_text(json.dumps(new_stats, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--splits", nargs="+", default=["splitB"])
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_root = root / cfg["dataset"]["root"]

    for name in args.splits:
        split_dir = data_root / name
        restore_info_features(split_dir)
        fix_stats(split_dir)
        print(f"  已处理: {split_dir}")


if __name__ == "__main__":
    main()
