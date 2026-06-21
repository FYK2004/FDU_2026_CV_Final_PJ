# 题目二：LeRobot ACT + CALVIN 跨环境泛化

## 环境配置

### 硬件要求

| 项目 | 说明 |
|------|------|
| GPU | NVIDIA，建议 ≥16GB（batch_size=16，ACT + ResNet-18） |
| 磁盘 | 数据下载与 video 转换约需 **10–15 GB** |
| Python | 3.10（由 `setup.sh` 自动创建） |

### 安装步骤

```bash
cd Task2
bash scripts/setup.sh
conda activate task2-lerobot
```

`setup.sh` 会创建 `task2-lerobot` 环境并安装：

- PyTorch（cu121）
- LeRobot、WandB、HuggingFace Hub
- 视频解码：`av`（pyav）

依赖清单见 `requirements.txt`，超参与 WandB 配置见 `config/task2.yaml`。

### 验证安装

```bash
conda activate task2-lerobot
python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import av; print('pyav OK')"
```

---

## 数据集下载

### 数据来源

HuggingFace 数据集：[xiaoma26/calvin-lerobot](https://huggingface.co/datasets/xiaoma26/calvin-lerobot)（LeRobot v3 格式，含 splitA/B/C/D）。

### 一键下载与预处理

```bash
conda activate task2-lerobot
bash scripts/prepare_data.sh
```

该脚本依次执行：

| 步骤 | 说明 |
|------|------|
| 下载 splitB、splitD | 环境 B 训练 + 环境 D 评测 |
| 下载 splitA、splitC 并合并 | 生成 `splitABC_merged`（ABC 联合训练） |
| 字段修复 | 统一 CALVIN 列名至 LeRobot ACT 接口 |
| 转 video | splitB / splitABC_merged / splitD → H.264 video dtype |
| episode 划分 | 生成 `data/splits/env_b_only.json` 等 |

### 手动分步（可选）

```bash
# 仅下载某一 split
python src/download_splits.py --splits B
python src/convert_splits.py --splits B
python src/fix_dataset_features.py --splits splitB

# 合并 ABC（联合训练前）
python src/download_splits.py --splits A C
python src/convert_splits.py --splits A C
python src/merge_splits.py

# 转 video + 划分
python src/convert_to_video.py --splits splitB --replace-source
python src/prepare_splits.py
```

### HuggingFace 访问

若下载失败，请先登录：

```bash
huggingface-cli login
# 或设置 Token
export HF_TOKEN=your_token
```

国内网络可尝试：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 下载后目录结构

```
Task2/data/
├── calvin-lerobot/
│   ├── splitB/              # 环境 B 训练
│   ├── splitABC_merged/     # A+B+C 联合训练
│   └── splitD/              # 环境 D 评测
└── splits/
    ├── env_b_only.json
    ├── env_abc.json
    └── env_d_eval.json
```

---

## Train

```bash
bash scripts/train_env_b.sh     # 阶段一：仅环境 B → outputs/train/act_env_b_only/
bash scripts/train_env_abc.sh   # 阶段二：A+B+C 联合 → outputs/train/act_env_abc/
```

两阶段使用相同 ACT 架构与超参（`config/task2.yaml`：chunk_size=100，lr=1e-5，50k steps，L1+KL λ=10）。

## Test

```bash
bash scripts/eval_zero_shot_d.sh
```

结果：`outputs/eval/zero_shot_env_d/summary.json`（Action L1、Success Rate 代理、chunk horizon 曲线）
