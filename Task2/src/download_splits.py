#!/usr/bin/env python3
"""从 HuggingFace 下载 xiaoma26/calvin-lerobot 的 splitA/B/C/D 子目录。"""
import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


SPLIT_MAP = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument(
        "--splits",
        nargs="+",
        default=["B"],
        help="环境字母 A/B/C/D，默认仅下载 B（基础策略训练）",
    )
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hf_repo = cfg["dataset"].get("hf_repo", cfg["dataset"]["repo_id"])
    data_root = root / cfg["dataset"]["root"]
    data_root.mkdir(parents=True, exist_ok=True)

    for env in args.splits:
        env = env.upper()
        if env not in SPLIT_MAP:
            print(f"未知环境: {env}", file=sys.stderr)
            sys.exit(1)
        split_name = SPLIT_MAP[env]
        out = data_root / split_name
        if (out / "meta" / "info.json").exists():
            print(f"[skip] {split_name} 已存在: {out}")
            continue
        print(f"下载 {hf_repo}/{split_name} -> {out} ...")
        snapshot_download(
            hf_repo,
            repo_type="dataset",
            allow_patterns=[f"{split_name}/**"],
            local_dir=data_root.parent / "_hf_raw",
        )
        raw = data_root.parent / "_hf_raw" / split_name
        if not raw.is_dir():
            print(f"下载失败: {raw}", file=sys.stderr)
            sys.exit(1)
        raw.rename(out)
        print(f"  完成: {out}")


if __name__ == "__main__":
    main()
