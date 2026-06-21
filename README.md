# HW3 计算机视觉 · 期末项目

GitHub: https://github.com/FYK2004/FDU_2026_CV_Final_PJ

题目一：2DGS + AIGC 多源三维重建与场景融合  
题目二：LeRobot ACT + CALVIN 跨环境泛化

```
final_pj/
├── Task1/          # 题目一
└── Task2/          # 题目二
```

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，建议 ≥24GB 显存 |
| 系统 | Linux，已安装 [Conda](https://docs.conda.io/) |
| 题目一额外依赖 | COLMAP、Blender（B/C 多视角渲染） |
| 题目一配置文件 | `Task1/environment/*.yml`、`Task1/requirements.txt` |
| 题目二配置文件 | `Task2/requirements.txt`、`Task2/config/task2.yaml` |

## 环境配置

**题目一**

```bash
cd Task1
chmod +x scripts/*.sh
bash scripts/setup_env.sh    # 创建 conda 环境 + 克隆第三方仓库 + 编译 2DGS
```

将创建 `task1-tools`、`task1-2dgs`、`task1-threestudio` 等环境（约 20–40 分钟，需 GPU 与网络）。

**题目二**

```bash
cd Task2
bash scripts/setup.sh
conda activate task2-lerobot
```

详细说明见 [`Task1/README.md`](Task1/README.md)、[`Task2/README.md`](Task2/README.md)。

## 数据集下载

**题目一**（需手动准备，见 Task1 README）：

- 物体 A：自备多视角照片 → `Task1/data/object_a/images/`
- 物体 C：仓库已含 `teddy_rgba.png`，链至 `rgba.png` 即可
- 背景：下载 [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)，解压 `bicycle/` 至 `Task1/data/background/bicycle/images/`
- AIGC 模型：Stable Diffusion v1.5、Zero123（首次运行 threestudio 时从 HuggingFace 自动下载）

**题目二**（脚本自动下载）：

```bash
cd Task2
conda activate task2-lerobot
bash scripts/prepare_data.sh
```

从 HuggingFace [`xiaoma26/calvin-lerobot`](https://huggingface.co/datasets/xiaoma26/calvin-lerobot) 下载 splitB/D、合并 ABC、转 video 并划分 episode。需能访问 HuggingFace（必要时 `huggingface-cli login`）。

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
