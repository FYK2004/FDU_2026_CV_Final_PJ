#!/usr/bin/env python3
"""
从本地 splitA/B/C/D 生成 episode 列表。
splitA=环境A, splitB=环境B, splitC=环境C, splitD=环境D（零样本评测）
"""
import argparse
import json
from pathlib import Path

import yaml

SPLIT_MAP = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}


def load_episode_indices(split_dir: Path) -> list[int]:
    info_path = split_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"缺少 {info_path}，请先下载并转换数据集")
    info = json.loads(info_path.read_text())
    n = int(info["total_episodes"])
    return list(range(n))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--output", type=Path, default=Path("data/splits"))
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = root / cfg["dataset"]["root"]
    args.output.mkdir(parents=True, exist_ok=True)

    episodes_by_env: dict[str, list[int]] = {}
    for env, split_name in SPLIT_MAP.items():
        split_dir = data_root / split_name
        if not (split_dir / "meta" / "info.json").exists():
            print(f"  [skip] 环境 {env} ({split_name}) 未下载")
            continue
        eps = load_episode_indices(split_dir)
        episodes_by_env[env] = eps
        print(f"  环境 {env} ({split_name}): {len(eps)} episodes")

    if "B" not in episodes_by_env:
        raise SystemExit("至少需要 splitB 数据，请运行 download_splits.py --splits B")

    splits: dict[str, list[int]] = {"env_b_only": episodes_by_env["B"]}

    merged_dir = data_root / "splitABC_merged"
    if (merged_dir / "meta" / "info.json").exists():
        n = int(json.loads((merged_dir / "meta" / "info.json").read_text())["total_episodes"])
        splits["env_abc"] = list(range(n))
        print(f"  使用合并数据集 splitABC_merged: {n} episodes")
    elif all(k in episodes_by_env for k in ("A", "B", "C")):
        splits["env_abc"] = (
            episodes_by_env["A"] + episodes_by_env["B"] + episodes_by_env["C"]
        )
    if "D" in episodes_by_env:
        splits["env_d_eval"] = episodes_by_env["D"]

    for name, eps in splits.items():
        out = args.output / f"{name}.json"
        out.write_text(json.dumps(eps), encoding="utf-8")
        print(f"  {name}: {len(eps)} episodes -> {out}")

    # 记录各 split 的 repo 名（供 train_act 使用）
    meta = {
        "split_repos": SPLIT_MAP,
        "data_root": str(cfg["dataset"]["root"]),
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("完成。")


if __name__ == "__main__":
    main()
