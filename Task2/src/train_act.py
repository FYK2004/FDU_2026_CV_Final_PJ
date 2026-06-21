#!/usr/bin/env python3
"""环境 B 基础 ACT 训练（xiaoma26/calvin-lerobot splitB）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

RENAME_MAP = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
    "state": "observation.state",
    "actions": "action",
}

SPLIT_FOR_TRAIN = {
    "env_b_only": "splitB",
    "env_abc": "splitABC_merged",
}

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


def apply_calvin_patches() -> None:
    """CALVIN parquet 列名与 ACT 标准接口不一致，需在 import 后 patch。"""
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies import factory as policy_factory
    from lerobot.policies.act import processor_act
    from lerobot.processor.converters import batch_to_transition, create_transition
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.datasets import factory as dataset_factory
    from lerobot.utils.constants import ACTION, DONE, OBS_PREFIX, REWARD, TRUNCATED

    def calvin_batch_to_transition(batch: dict):
        batch = dict(batch)
        if "actions" in batch and ACTION not in batch:
            batch[ACTION] = batch.pop("actions")
        if "actions_is_pad" in batch and "action_is_pad" not in batch:
            batch["action_is_pad"] = batch.pop("actions_is_pad")

        observation = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
        for old, new in RENAME_MAP.items():
            if old in batch and new.startswith(OBS_PREFIX):
                observation[new] = batch.pop(old)

        pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
        comp = {**pad_keys}
        for key in ("task", "subtask", "index", "task_index", "episode_index"):
            if key in batch:
                comp[key] = batch[key]

        return create_transition(
            observation=observation or None,
            action=batch.get(ACTION),
            reward=batch.get(REWARD, 0.0),
            done=batch.get(DONE, False),
            truncated=batch.get(TRUNCATED, False),
            info=batch.get("info", {}),
            complementary_data=comp or None,
        )

    _orig_resolve = dataset_factory.resolve_delta_timestamps

    def patched_resolve_delta_timestamps(cfg, ds_meta):
        """数据集动作列名为 actions，需纳入 chunk 时间窗。"""
        from lerobot.utils.constants import ACTION as ACT_KEY

        delta_timestamps = {}
        for key in ds_meta.features:
            if key in (ACT_KEY, "actions") and cfg.action_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
            elif key == "reward" and cfg.reward_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
            elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
                delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        return delta_timestamps if delta_timestamps else None

    dataset_factory.resolve_delta_timestamps = patched_resolve_delta_timestamps

    _orig_make_policy = policy_factory.make_policy

    def patched_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
        if ds_meta is None:
            return _orig_make_policy(cfg, ds_meta, env_cfg, rename_map)

        features = dataset_to_policy_features(ds_meta.features)
        mapped = {RENAME_MAP.get(k, k): v for k, v in features.items()}
        cfg.output_features = {k: ft for k, ft in mapped.items() if ft.type is FeatureType.ACTION}
        if not cfg.input_features:
            cfg.input_features = {k: ft for k, ft in mapped.items() if k not in cfg.output_features}

        stats = rename_stats(ds_meta.stats or {}, RENAME_MAP) if ds_meta.stats else {}
        for key, ft in cfg.input_features.items():
            if ft.type is FeatureType.VISUAL and key not in stats:
                stats[key] = {
                    k: torch.tensor(v, dtype=torch.float32) for k, v in IMAGENET_STATS.items()
                }

        policy_cls = policy_factory.get_policy_class(cfg.type)
        policy = policy_cls(config=cfg, dataset_stats=stats)
        policy.to(cfg.device)
        return policy

    policy_factory.make_policy = patched_make_policy

    _orig_act_pp = processor_act.make_act_pre_post_processors

    def patched_act_pp(config, dataset_stats=None):
        stats = dict(dataset_stats or {})
        for key in config.input_features:
            ft = config.input_features[key]
            if ft.type is FeatureType.VISUAL and key not in stats:
                stats[key] = {
                    k: torch.tensor(v, dtype=torch.float32) for k, v in IMAGENET_STATS.items()
                }
        pre, post = _orig_act_pp(config, stats or None)
        pre.to_transition = calvin_batch_to_transition
        for step in pre.steps:
            if step.__class__.__name__ == "RenameObservationsProcessorStep":
                step.rename_map = dict(RENAME_MAP)
        return pre, post

    processor_act.make_act_pre_post_processors = patched_act_pp


def build_argv(cfg: dict, split: str, out_dir: Path, dataset_root: Path) -> list[str]:
    tr = cfg["training"]
    wb = cfg["wandb"]
    repo_id = SPLIT_FOR_TRAIN[split]
    rename_arg = json.dumps(RENAME_MAP, separators=(",", ":"))

    argv = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.use_imagenet_stats=false",
        f"--dataset.video_backend={cfg['dataset'].get('video_backend', 'pyav')}",
        f"--rename_map={rename_arg}",
        "--policy.type=act",
        "--policy.push_to_hub=false",
        f"--output_dir={out_dir}",
        f"--job_name={split}",
        f"--policy.device={tr['device']}",
        f"--batch_size={tr['batch_size']}",
        f"--num_workers={tr['num_workers']}",
        f"--steps={tr['steps']}",
        f"--eval_freq={tr['eval_freq']}",
        f"--save_freq={tr['save_freq']}",
        f"--log_freq={tr['log_freq']}",
        f"--policy.chunk_size={tr['chunk_size']}",
        f"--policy.n_action_steps={tr['n_action_steps']}",
        f"--optimizer.lr={tr['lr']}",
        f"--policy.use_amp={str(tr.get('use_amp', False)).lower()}",
        f"--wandb.enable={str(wb['enable']).lower()}",
        f"--wandb.project={wb['project']}",
    ]
    if wb.get("mode"):
        argv.append(f"--wandb.mode={wb['mode']}")
    if wb.get("entity"):
        argv.append(f"--wandb.entity={wb['entity']}")
    if wb.get("notes"):
        argv.append(f"--wandb.notes={wb['notes']}")
    return argv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--split", required=True, choices=["env_b_only", "env_abc"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--steps", type=int, default=None, help="覆盖 config 中的 steps")
    p.add_argument("--force", action="store_true", help="删除已有 checkpoint 并重新训练")
    args = p.parse_args()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_path = root / "data" / "splits" / f"{args.split}.json"
    if not split_path.exists():
        print(f"缺少 {split_path}，请先: python src/prepare_splits.py", file=sys.stderr)
        sys.exit(1)

    out = args.output_dir or cfg["outputs"]["env_b" if args.split == "env_b_only" else "env_abc"]
    out_dir = (root / out).resolve()
    if args.force and out_dir.exists():
        import shutil
        print(f"删除已有输出: {out_dir}")
        shutil.rmtree(out_dir)
    elif out_dir.exists() and any(out_dir.iterdir()):
        has_ckpt = (out_dir / "checkpoints").exists()
        if not has_ckpt:
            print(f"清理无效输出目录: {out_dir}")
            import shutil
            shutil.rmtree(out_dir)
        else:
            print(f"输出目录已有 checkpoint: {out_dir}，跳过训练（加 --force 重训）", file=sys.stderr)
            sys.exit(0)

    data_parent = root / cfg["dataset"]["root"]
    repo_id = SPLIT_FOR_TRAIN[args.split]
    dataset_root = (data_parent / repo_id).resolve()
    if not (dataset_root / "meta" / "info.json").exists():
        hint = "merge_splits.py（env_abc）" if args.split == "env_abc" else "download_splits.py（splitB）"
        print(f"缺少 {dataset_root}，请先运行 {hint}", file=sys.stderr)
        sys.exit(1)

    if args.steps is not None:
        cfg["training"]["steps"] = args.steps

    apply_calvin_patches()
    sys.argv = build_argv(cfg, args.split, out_dir, dataset_root)
    print("$", " ".join(sys.argv))

    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
