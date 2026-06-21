# 题目一：2DGS + AIGC 多源三维重建与场景融合

## 环境配置

### 硬件与系统依赖


| 项目      | 说明                                               |
| ------- | ------------------------------------------------ |
| GPU     | NVIDIA，建议 ≥24GB（物体 C refine 可在 10GB 上以 128px 运行） |
| COLMAP  | 由 `task1-tools` conda 环境提供                       |
| Blender | 系统安装，用于 B/C mesh 多视角渲染（`run_object_bc_2dgs.sh`）  |


### 一键安装

```bash
cd Task1
chmod +x scripts/*.sh
bash scripts/setup_env.sh
```

该脚本会：

1. 克隆第三方仓库至 `external/`（2DGS、threestudio、Magic123）
2. 创建 conda 环境并安装依赖
3. 编译 2DGS CUDA 扩展（`bash scripts/build_2dgs_cuda.sh`）


| Conda 环境            | 用途                               |
| ------------------- | -------------------------------- |
| `task1-tools`       | COLMAP、Python 预处理脚本              |
| `task1-2dgs`        | 物体 A / 背景 / B&C 2DGS 训练与融合       |
| `task1-threestudio` | 物体 B（DreamFusion）、物体 C（Magic123） |


各阶段脚本会自动激活对应环境；2DGS 也可通过环境变量指定：`export TASK1_2DGS_ENV=task1-2dgs`。

### 验证安装

```bash
conda activate task1-tools && colmap -h | head -1
conda activate task1-2dgs && python -c "import torch; print(torch.cuda.is_available())"
blender --version
```

### AIGC 预训练模型

物体 B/C 依赖 HuggingFace 模型，首次运行时会自动下载（需网络）：


| 模型                               | 用途             | 配置                                        |
| -------------------------------- | -------------- | ----------------------------------------- |
| `runwayml/stable-diffusion-v1-5` | SDS / Magic123 | `config/task1.yaml` → `object_c.sd_model` |
| Zero123                          | Magic123 单图先验  | threestudio 配置中指定                         |


也可提前下载至本地并修改 `config/task1.yaml` 中的模型路径。若网络受限，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 可选
```

---

## 数据集下载与准备

### 物体 A（真实多视角）

将 **30 张以上** 环绕拍摄的照片放入：

```
data/object_a/images/*.jpg   # 或 *.png
```

也可放置 `data/object_a/video.mp4`，脚本会自动抽帧。

### 物体 B（文本 AIGC）

无需额外数据。编辑 `config/task1.yaml` → `object_b.prompt`（默认 `"a hamburger"`）。

### 物体 C（单图 AIGC）

仓库已包含 `data/object_c/teddy_rgba.png`。运行前链至 `rgba.png`：

```bash
cd Task1/data/object_c
ln -sf teddy_rgba.png rgba.png
```

### 背景（Mip-NeRF 360 bicycle）

```bash
cd Task1
mkdir -p data/background
wget -O /tmp/360_v2.zip http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip /tmp/360_v2.zip bicycle -d data/background/
# 确保图像在 data/background/bicycle/images/
```

场景名在 `config/task1.yaml` → `background.scene: bicycle`。

### 数据目录结构（训练前检查）

```
Task1/data/
├── object_a/images/          # 自备多视角照片
├── object_c/rgba.png         # 链至 teddy_rgba.png
└── background/bicycle/images/  # Mip-NeRF 360 解压
```

---

## Train

```bash
bash scripts/run_pipeline.sh
```

分步：

```bash
bash scripts/run_object_a.sh       # A：COLMAP + 2DGS
bash scripts/run_object_b.sh       # B：SDS → mesh
bash scripts/run_object_c.sh       # C：Magic123 → mesh
bash scripts/run_background.sh     # 背景 2DGS
bash scripts/run_object_bc_2dgs.sh # B/C mesh → 2DGS
bash scripts/run_fusion_2dgs.sh    # 融合
```

## Test

主输出：`outputs/fusion/wander.mp4`

修改融合位姿（`config/task1.yaml` → `fusion.placements`）后仅重渲染：

```bash
bash scripts/run_fusion_2dgs.sh \
  --a-iteration 10000 --b-iteration 10000 \
  --c-iteration 10000 --bg-iteration 30000
```

