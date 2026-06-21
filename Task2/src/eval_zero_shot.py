#!/usr/bin/env python3
"""在环境 D 上对两个 ACT 模型做 Zero-shot 离线评测（Action L1 + Chunk 分析）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from train_act import RENAME_MAP, apply_calvin_patches


def attach_calvin_preprocessor(preprocessor):
    """checkpoint 预处理器默认不识别 CALVIN 列名，需与训练时一致。"""
    from lerobot.processor.converters import create_transition
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

    preprocessor.to_transition = calvin_batch_to_transition
    for step in preprocessor.steps:
        if step.__class__.__name__ == "RenameObservationsProcessorStep":
            step.rename_map = dict(RENAME_MAP)
    return preprocessor


def build_delta_timestamps(chunk_size: int, fps: int) -> dict[str, list[float]]:
    ts = [i / fps for i in range(chunk_size)]
    return {"actions": ts}


def evaluate_checkpoint(
    ckpt: Path,
    dataset_root: Path,
    episodes: list[int],
    device: str,
    batch_size: int,
    num_workers: int,
    video_backend: str,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION

    policy = ACTPolicy.from_pretrained(ckpt)
    policy.to(device)
    policy.eval()

    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
    )
    preprocessor = attach_calvin_preprocessor(preprocessor)

    fps = 10
    dataset = LeRobotDataset(
        repo_id="splitD",
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=build_delta_timestamps(policy.config.chunk_size, fps),
        video_backend=video_backend,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.startswith("cuda"),
    )

    total_l1 = 0.0
    total_first_step_l1 = 0.0
    total_frames = 0
    chunk_pos_l1 = torch.zeros(policy.config.chunk_size)
    chunk_pos_count = torch.zeros(policy.config.chunk_size)
    episode_l1: dict[int, list[float]] = {}

    with torch.no_grad():
        for batch in loader:
            batch = preprocessor(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if policy.config.image_features:
                from lerobot.utils.constants import OBS_IMAGES

                batch = dict(batch)
                batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

            actions_hat, _ = policy.model(batch)
            gt = batch[ACTION]
            pad = ~batch["action_is_pad"].unsqueeze(-1)
            per_elem = (actions_hat - gt).abs() * pad

            valid_elems = pad.expand_as(per_elem)
            batch_l1 = per_elem.sum() / valid_elems.sum().clamp(min=1)
            total_l1 += batch_l1.item() * gt.shape[0]
            total_first_step_l1 += per_elem[:, 0].mean().item() * gt.shape[0]
            total_frames += gt.shape[0]

            for t in range(gt.shape[1]):
                valid = pad[:, t].squeeze(-1)
                if valid.any():
                    chunk_pos_l1[t] += per_elem[:, t][valid].mean().item() * valid.sum().item()
                    chunk_pos_count[t] += valid.sum().item()

            ep_idx = batch["episode_index"].view(-1).tolist()
            frame_l1 = per_elem.mean(dim=(1, 2)).tolist()
            for ep, l1 in zip(ep_idx, frame_l1):
                episode_l1.setdefault(int(ep), []).append(float(l1))

    ep_means = {ep: sum(v) / len(v) for ep, v in episode_l1.items()}
    threshold = 0.25
    success_proxy = sum(1 for v in ep_means.values() if v < threshold) / max(len(ep_means), 1)

    chunk_curve = (chunk_pos_l1 / chunk_pos_count.clamp(min=1)).tolist()

    return {
        "episodes_evaluated": len(ep_means),
        "frames_evaluated": total_frames,
        "action_l1_mean": total_l1 / max(total_frames, 1),
        "action_l1_first_step": total_first_step_l1 / max(total_frames, 1),
        "success_rate_proxy": success_proxy,
        "success_threshold_l1": threshold,
        "chunk_l1_by_horizon": chunk_curve,
        "episode_action_l1_mean": float(sum(ep_means.values()) / max(len(ep_means), 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/task2.yaml"))
    p.add_argument("--checkpoint-b", type=Path, default=None)
    p.add_argument("--checkpoint-abc", type=Path, default=None)
    p.add_argument("--max-episodes", type=int, default=100, help="评测 episode 数（默认 100）")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    apply_calvin_patches()

    root = args.config.resolve().parent.parent
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_path = root / "data" / "splits" / "env_d_eval.json"
    if not split_path.exists():
        print(f"缺少 {split_path}，请先: python src/prepare_splits.py", file=sys.stderr)
        sys.exit(1)

    ckpt_b = args.checkpoint_b or (
        root / cfg["outputs"]["env_b"] / "checkpoints" / "last" / "pretrained_model"
    )
    ckpt_abc = args.checkpoint_abc or (
        root / cfg["outputs"]["env_abc"] / "checkpoints" / "last" / "pretrained_model"
    )
    eval_out = root / cfg["outputs"]["eval_d"]
    eval_out.mkdir(parents=True, exist_ok=True)

    all_eps = json.loads(split_path.read_text())
    episodes = all_eps[: min(args.max_episodes, len(all_eps))]
    dataset_root = (root / cfg["dataset"]["root"] / "splitD").resolve()
    device = cfg["training"]["device"]
    video_backend = cfg["dataset"].get("video_backend", "pyav")

    results = {
        "eval_env": "D",
        "episodes_used": episodes,
        "num_episodes": len(episodes),
        "note": "无 CALVIN 仿真环境，使用 splitD 离线 Action L1；success_rate_proxy 为 episode 均值 L1<threshold 的比例。",
    }

    for name, ckpt in [("act_env_b_only", ckpt_b), ("act_env_abc", ckpt_abc)]:
        if not ckpt.exists():
            print(f"[skip] 未找到权重: {ckpt}")
            results[name] = {"status": "missing_checkpoint"}
            continue
        print(f"评测 {name} @ {ckpt} ({len(episodes)} episodes)...")
        metrics = evaluate_checkpoint(
            ckpt=ckpt,
            dataset_root=dataset_root,
            episodes=episodes,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            video_backend=video_backend,
        )
        metrics["status"] = "ok"
        metrics["checkpoint"] = str(ckpt)
        results[name] = metrics
        print(
            f"  L1={metrics['action_l1_mean']:.4f}, "
            f"first_step={metrics['action_l1_first_step']:.4f}, "
            f"success_proxy={metrics['success_rate_proxy']*100:.1f}%"
        )

    summary = eval_out / "summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"评测摘要: {summary}")


if __name__ == "__main__":
    main()
