#!/usr/bin/env python3
"""将 v2.1 LeRobot 格式 split 转为 v3.0（lerobot>=0.4 要求）。"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SPLIT_MAP = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}
V30 = "v3.0"


def patch_episodes_stats(split_dir: Path) -> Path:
    """为 v2.1 episodes_stats 补全 count 字段（官方转换器需要）。"""
    stats_path = split_dir / "meta" / "episodes_stats.jsonl"
    ep_path = split_dir / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    if ep_path.exists():
        with ep_path.open() as f:
            for line in f:
                row = json.loads(line)
                lengths[int(row["episode_index"])] = int(row["length"])

    backup = stats_path.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(stats_path, backup)

    lines_out = []
    with stats_path.open() as f:
        for line in f:
            row = json.loads(line)
            ep = int(row["episode_index"])
            n = lengths.get(ep, 60)
            stats = row["stats"]
            for feat_stats in stats.values():
                if "count" not in feat_stats:
                    feat_stats["count"] = [n]
            lines_out.append(json.dumps(row))
    stats_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return stats_path


def cleanup_partial_v30(data_parent: Path, split_name: str) -> None:
    for suffix in ("_v30", "_old"):
        p = data_parent / f"{split_name}{suffix}"
        if p.is_dir():
            shutil.rmtree(p)
    nested_old = data_parent / f"{split_name}_old" / split_name
    if nested_old.is_dir():
        shutil.rmtree(data_parent / f"{split_name}_old")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--splits", nargs="+", default=["B"])
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_parent = root / cfg["dataset"]["root"]

    for env in args.splits:
        env = env.upper()
        split_name = SPLIT_MAP[env]
        split_dir = data_parent / split_name
        info_path = split_dir / "meta" / "info.json"
        if not info_path.exists():
            print(f"缺少 {info_path}，请先运行 download_splits.py", file=sys.stderr)
            sys.exit(1)

        info = json.loads(info_path.read_text())
        if info.get("codebase_version") == V30:
            print(f"[skip] {split_name} 已是 {V30}")
            continue

        cleanup_partial_v30(data_parent, split_name)
        patch_episodes_stats(split_dir)

        print(f"转换 {split_name} v2.1 -> v3.0 ...")
        cmd = [
            sys.executable,
            "-m",
            "lerobot.datasets.v30.convert_dataset_v21_to_v30",
            f"--repo-id={split_name}",
            f"--root={data_parent}",
            "--push-to-hub=false",
        ]
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"  完成: {split_dir}")


if __name__ == "__main__":
    main()
