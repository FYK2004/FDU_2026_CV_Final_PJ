# HW3 实验报告（CVPR 模板）

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.tex` | **主报告**（题目一 + 题目二），CVPR 双栏格式 |
| `refs.bib` | 参考文献 |
| `report.md` | Markdown 对照稿（与 LaTeX 同步，便于编辑） |

## 编译步骤

1. 从 [CVPR Author Kit](https://github.com/cvpr-org/author-kit) 下载 `cvpr.sty`，放入本目录 `report/`。

2. 安装 TeX 与中文支持（推荐 TeX Live + xelatex）。

3. 编译：

```bash
cd report
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

输出：`main.pdf`

## 待补全项（提交前）

- [ ] 封面姓名、学号、邮箱（`main.tex` 第 24–27 行）
- [ ] GitHub Public 链接（`main.tex` 提交信息节）
- [ ] 模型权重网盘链接与提取码
- [x] 题目一融合位姿（与 `Task1/config/task1.yaml` 一致）
- [x] 题目一训练耗时与三种路径对比（§1.3 / 表 task1_time）
- [ ] WandB Loss 曲线截图（Figure 4）
- [ ] 融合视频关键帧截图（Figure 3）
- [ ] chunk L1 对比图（Figure 5，数据见 `Task2/outputs/eval/zero_shot_env_d/summary.json`）

## 实验数据路径

| 内容 | 路径 |
|------|------|
| 题目一融合视频 | `Task1/outputs/fusion/wander.mp4` |
| 题目一配置 | `Task1/config/task1.yaml` |
| 题目二评测 | `Task2/outputs/eval/zero_shot_env_d/summary.json` |
| WandB | `hw3_task2_act` |

## 仓库 README（作业 4.2）

根目录 [`../README.md`](../README.md) 含环境说明与 Train/Test 命令；分任务见 `Task1/README.md`、`Task2/README.md`。
