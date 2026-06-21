# 题目二：LeRobot ACT + CALVIN 跨环境泛化

## Requirements

```bash
cd Task2
bash scripts/setup.sh
conda activate task2-lerobot
```

- 依赖：`requirements.txt`（PyTorch cu121、LeRobot、WandB 等）
- 配置：`config/task2.yaml`（ACT 架构、超参、WandB）
- 数据：HuggingFace `xiaoma26/calvin-lerobot`（由 `prepare_data.sh` 自动下载）

## Train

```bash
bash scripts/prepare_data.sh    # 下载 split、合并 ABC、转 video、划分 episode
bash scripts/train_env_b.sh     # 阶段一：仅环境 B
bash scripts/train_env_abc.sh   # 阶段二：A+B+C 联合（同架构同超参）
```

权重输出：

- `outputs/train/act_env_b_only/checkpoints/050000/`
- `outputs/train/act_env_abc/checkpoints/050000/`

## Test

环境 D 零样本评测：

```bash
bash scripts/eval_zero_shot_d.sh
```

结果：`outputs/eval/zero_shot_env_d/summary.json`（Action L1、Success Rate 代理、chunk horizon 曲线）
