# 题目一：2DGS + AIGC 多源三维重建与场景融合

## Requirements

```bash
cd Task1
chmod +x scripts/*.sh
bash scripts/setup_env.sh      # conda 环境
bash scripts/build_2dgs_cuda.sh # 2DGS CUDA 扩展
```

| 环境 | 用途 |
|------|------|
| `task1-tools` | COLMAP、预处理 |
| `task1-2dgs` | 2DGS 训练与融合（可通过 `TASK1_2DGS_ENV` 指定其他环境） |
| `task1-threestudio` | 物体 B/C（DreamFusion / Magic123） |

硬件：NVIDIA GPU；需安装 COLMAP、Blender。  
配置：`config/task1.yaml`（Prompt、训练步数、融合位姿）。

### 数据准备

| 资产 | 路径 |
|------|------|
| 物体 A（多视角实拍） | `data/object_a/images/` |
| 物体 B（文本 Prompt） | 编辑 `config/task1.yaml` → `object_b.prompt` |
| 物体 C（单图 RGBA） | `data/object_c/rgba.png` |
| 背景 bicycle | 下载 [360_v2.zip](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)，解压至 `data/background/bicycle/images/` |

Stable Diffusion / Zero123 等模型需按 `config/task1.yaml` 中路径提前下载。

## Train

```bash
bash scripts/run_pipeline.sh
```

等价分步：

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

修改融合位姿后仅重渲染：

```bash
bash scripts/run_fusion_2dgs.sh \
  --a-iteration 10000 --b-iteration 10000 \
  --c-iteration 10000 --bg-iteration 30000
```

其他 checkpoint：`outputs/object_{a,b,c}/train/point_cloud/`、`outputs/background/bicycle/train/point_cloud/`、`outputs/object_{b,c}/model.obj`
