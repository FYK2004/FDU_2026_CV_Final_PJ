# 基于 2D Gaussian Splatting 与 LeRobot ACT 的多源三维重建融合与跨环境策略学习

**HW3 计算机视觉 · 实验报告（CVPR 结构对照稿，编译见 `report/main.tex`）**

| | |
|---|---|
| 姓名 / 学号 | 【姓名】【学号】 |
| 单位 | 复旦大学 |
| 邮箱 | 【email@fudan.edu.cn】 |
| GitHub | 【Public Repo URL】 |
| 模型权重 | 【网盘链接，提取码：****】 |

---

## Abstract

本文完成 HW3 两项实验。**题目一**：从真实多视角（A）、文本 Prompt（B）、单张 RGBA（C）重建三维资产；在 Mip-NeRF 360 `bicycle` 上训练 2DGS 背景；将背景与 A/B/C **全部表示为 2D 高斯**合并光栅化，输出 `wander.mp4`。AIGC mesh 经多视角渲染与 2DGS 训练后与实拍资产在同一高斯域融合。**题目二**：LeRobot ACT 在 CALVIN 环境 B 与 A+B+C 上训练，于环境 D 零样本评测；ABC 的 Success Rate 代理由 74% 升至 97%，Action Chunking 长程 L1 更平稳。

---

## 1. 题目一：多源 3D 重建与场景融合

### 1.1 实验设置

| 组件 | 配置 |
|------|------|
| 背景 | Mip-NeRF 360 `bicycle`，2DGS 30,000 iters |
| 物体 A | 72 张实拍，COLMAP + 2DGS 30k iters（融合用 iter 10000） |
| 物体 B | Prompt `"a hamburger"`，threestudio 12k steps → **obj+UV**（`model.obj` + `texture_kd.png`） |
| 物体 C | `teddy_rgba.png`，Magic123 coarse 5k + **refine 5k** → **obj+UV**（同 B 格式，~3.5 万顶点） |
| B/C 2DGS | 72 视角 Blender 渲染 + 各 10k iters（`run_object_bc_2dgs.sh`） |
| 配置 | `Task1/config/task1.yaml` |

### 1.2 融合方法与统一表示（作业核心问题）

**最终方案：全 2DGS 原生融合**

```
背景 2DGS + A 2DGS + B/C 训练 2DGS
  → 裁剪、中心化、apply_placement
  → torch.cat 合并 → 单次光栅化 → wander.mp4
```

**Mesh / 隐式场与 2DGS 如何统一：**

1. A 路径直接输出 2D Gaussian。
2. B/C 的 AIGC 输出为 mesh（SDS / Magic123），不直接在融合阶段使用 mesh 光栅化。
3. 对 B/C mesh 做环绕多视角渲染，再训练 2DGS，得到与 A 同类型的 `point_cloud.ply`。
4. 融合时在张量层面对全部高斯做刚体变换后拼接，**一次 2DGS 渲染**，保证光照与表示一致。

**位姿**：在 `config/task1.yaml` 的 `fusion.placements` 中手动配置平移、旋转与缩放；点云导出朝向经 `fix_gaussian_ply_upright.py`（绕 X 轴 180°）校正。

**最终位姿**（`task1.yaml`）：

| 物体 | Position | Rotation (°) | Scale | 语义 |
|------|----------|--------------|-------|------|
| A 狗 | [0.97, 1.20, 1.18] | [0, 0, 90] | 0.55 | 地面 |
| B 汉堡 | [0.62, 1.00, 0.39] | [90, -90, 90] | 0.24 | 椅面 |
| C 熊 | [0.31, 0.20, -0.10] | [90, -90, 90] | 0.30 | 车座 |

**相机**：120 帧，radius=3.0，height=1.2，FOV=50°，1280×850，24 fps。

**命令**：

```bash
cd Task1
bash scripts/run_object_bc_2dgs.sh
bash scripts/run_fusion_2dgs.sh --b-iteration 10000 --c-iteration 10000
```

### 1.3 三种重建路径对比（作业要求）

| 维度 | A (实拍) | B (SDS) | C (Magic123) |
|------|----------|---------|--------------|
| 输入 | 72 张多视角 | 文本 Prompt | 单张 RGBA |
| 初始资产 | 2DGS 点云 | obj+UV mesh | obj+UV mesh |
| 几何准确度 | 最高 | 中（先验生成） | 中（单视图歧义） |
| 纹理细节 | 真实采集 | Prompt 驱动 | Refine 较细 |
| 融合表示 | 2D Gaussian | 2D Gaussian† | 2D Gaussian† |
| 相对耗时 | **低** | **中** | **高** |

† 由 mesh 多视角渲染后 2DGS 训练得到。

**重建耗时对比**（NVIDIA A100-80GB，单卡；统计至获得可用于融合的初始三维表示，不含场景融合渲染）：

| 阶段 | A (实拍) | B (SDS) | C (Magic123) |
|------|----------|---------|--------------|
| 主要环节 | COLMAP SfM → 2DGS 10k | DreamFusion SDS 12k → mesh | coarse 5k + refine 5k → mesh 导出 |
| COLMAP / SfM | ~19 min | — | — |
| 优化训练 | 2DGS ~14 min | SDS ~58 min‡ | coarse ~24 min + refine ~25 min‡ |
| 后处理 | — | mesh 导出 ~5 min | mesh-exporter + UV 纹理 ~10 min |
| **合计墙钟** | **~33 min** | **~63 min** | **~75 min** |
| 相对耗时 | **低** | **中** | **高** |

‡ B 按 12k 步、C refine 按与 coarse 相近的单步速率估算（coarse 5k 实测约 24 min，见 `threestudio/outputs/magic123-coarse-sd/...`）。

**分析**：A 无需扩散模型前向/反向，瓶颈在 COLMAP 特征匹配与高分辨率 2DGS 光栅化，三者中**最快**。B 为单阶段文本 SDS，每步需 Stable Diffusion 梯度回传，耗时约为 A 的 **1.9×**。C 需 **coarse→refine 两阶段**独立优化，并额外加载 Zero123、做 mesh 导出与 UV 烘焙；refine 在 10GB vGPU 上以 128px 分辨率训练，墙钟最长，约为 A 的 **2.3×**。B/C 若需与 A 统一为 2DGS 再融合，另各增加多视角渲染 + 2DGS 10k（约 20 min），对两者相同。相对 A，C 的 refine 显著改善正面纹理，但**单视图对后脑几何仍不可观**，后脑区域易出现局部凹陷或发虚（见 §3）。

### 1.4 结果

- 视频：`Task1/outputs/fusion/wander.mp4`（120 帧，1280×850，24 fps）
- 融合用 2DGS 点云规模：背景 ~410 万 / A ~25 万 / B ~2.0 万 / C ~2.3 万 高斯
- B/C 初始资产：`model.obj` + `model.mtl` + `texture_kd.png`（格式相同；C 约 3.5 万顶点）
- 【Figure 2】A/B/C 重建与点云
- 【Figure 3】融合关键帧：`report/figures/fusion/frame_{000,030,060,090}.png`（由 `report/scripts/extract_fusion_frames.sh` 从 `wander.mp4` 抽取）

---

## 2. 题目二：LeRobot ACT 跨环境泛化

### 2.1 数据集

- 来源：`xiaoma26/calvin-lerobot`（LeRobot v3），`splitA/B/C/D`
- 预处理：`convert_to_video.py`，解码 `pyav`
- 配置：`Task2/config/task2.yaml`

### 2.2 ACT 训练超参数（作业要求，两阶段完全一致）

| 项目 | 配置 |
|------|------|
| Network Architecture | ACT (ResNet-18 + Transformer) |
| Vision backbone | ResNet18, ImageNet 预训练 |
| $d_{model}$ / heads / FFN | 512 / 8 / 3200 |
| Encoder / Decoder layers | 4 / 1 |
| Chunk size / $n_{action\_steps}$ | 100 / 100 |
| Action dimension | 7 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Training steps | 50,000 |
| Loss Function | L1 + KL (λ=10) |
| AMP | 启用 |

| 阶段 | 训练数据 | 输出 |
|------|----------|------|
| B-only | splitB | `outputs/train/act_env_b_only` |
| ABC | splitABC_merged | `outputs/train/act_env_abc` |

WandB：`hw3_task2_act`（entity: fangyk-fudan-university-school-of-management）

【Figure 4】WandB 训练 Loss / L1 曲线

### 2.3 环境 D Zero-shot 评测

- 数据：splitD，100 episodes，6036 frames
- 方法：离线 Action L1（无 CALVIN 仿真）
- Success Rate 代理：episode 均值 L1 < 0.25 的比例

| 指标 | ACT (B-only) | ACT (ABC) |
|------|--------------|-----------|
| Action L1（首步） | 0.698 | **0.585** |
| Action L1（episode 均值） | 0.216 | **0.181** |
| Success Rate 代理 | 74% | **97%** |
| 训练终局 L1 loss | 0.120 | 0.149 |
| 训练时长 (s) | 9496 | 8996 |

### 2.4 分析：Action Chunking 与 Visual Distribution Shift（作业要求）

**Visual Distribution Shift**：D 未参与训练。B-only 过拟合 B 的视觉外观；ABC 见过 A/B/C 多样场景，D 上泛化更好（+23% Success Rate 代理）。

**Action Chunking**（`Task2/outputs/eval/zero_shot_env_d/summary.json`）：

- B-only：horizon > 40 时 L1 由 ~0.70 升至 ~0.75，长程漂移明显
- ABC：horizon 0–64 内 L1 由 0.58 缓升至 ~0.65，更平稳

【Figure 5】`chunk_l1_by_horizon` 对比图

---

## 3. 局限

- **题目一**：B/C 依赖 AIGC 几何质量；C 单视图导致**后脑不可观、易凹陷或发虚**；位姿在 `task1.yaml` 手调；B/C 点云朝向需 `fix_gaussian_ply_upright.py` 后处理
- **题目二**：离线 L1 为代理指标，非仿真 Success Rate

---

## 4. 结论

题目一通过「AIGC mesh → 多视角 2DGS → 高斯域合并」实现多源资产统一渲染；题目二验证 ABC 联合 ACT 在环境 D 零样本指标优于 B-only。

---

## 5. 提交清单（作业 4.1–4.3）

- [ ] PDF：`xelatex main.tex` 编译 `report/main.pdf`
- [ ] 封面：姓名、学号、邮箱
- [ ] WandB Loss 曲线截图插入 Figure 4
- [ ] 融合关键帧截图插入 Figure 3
- [ ] GitHub Public Repo + 根目录 `README.md`（Train/Test）
- [ ] 网盘：Task1 点云/mesh + Task2 checkpoint `050000/`

---

## References

1. Huang et al., 2D Gaussian Splatting, SIGGRAPH 2024.
2. Poole et al., DreamFusion, ICLR 2023.
3. Qian et al., Magic123, ICCV 2023.
4. Barron et al., Mip-NeRF 360, CVPR 2022.
5. Zhao et al., ACT, RSS 2023.
6. Mees et al., CALVIN, RAL 2022.

---

*提交请编译 `report/main.tex` 为 PDF；本 Markdown 为结构对照稿。*
