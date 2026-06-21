# HW3 计算机视觉 · 期末项目

GitHub: https://github.com/FYK2004/FDU_2026_CV_Final_PJ

题目一：2DGS + AIGC 多源三维重建与场景融合  
题目二：LeRobot ACT + CALVIN 跨环境泛化

```
final_pj/
├── Task1/          # 题目一
├── Task2/          # 题目二
└── report/         # 实验报告（LaTeX）
```

## Requirements

| 任务 | 环境 | 安装 |
|------|------|------|
| 题目一 | `task1-tools` / `task1-2dgs` / `task1-threestudio` | `cd Task1 && bash scripts/setup_env.sh` |
| 题目二 | `task2-lerobot` | `cd Task2 && bash scripts/setup.sh` |

- GPU：NVIDIA（建议 ≥24GB）
- 题目一额外依赖：COLMAP、Blender、`Task1/environment/*.yml`、`Task1/requirements.txt`
- 题目二依赖：`Task2/requirements.txt`
- 配置文件：`Task1/config/task1.yaml`、`Task2/config/task2.yaml`

详细步骤见 [`Task1/README.md`](Task1/README.md)、[`Task2/README.md`](Task2/README.md)。

## Train

**题目一**

```bash
cd Task1
bash scripts/run_pipeline.sh
```

**题目二**

```bash
cd Task2
bash scripts/prepare_data.sh
bash scripts/train_env_b.sh
bash scripts/train_env_abc.sh
```

## Test

**题目一**：融合漫游视频 `Task1/outputs/fusion/wander.mp4`

```bash
cd Task1
bash scripts/run_fusion_2dgs.sh
```

**题目二**：环境 D 零样本评测

```bash
cd Task2
bash scripts/eval_zero_shot_d.sh
```

结果：`Task2/outputs/eval/zero_shot_env_d/summary.json`
