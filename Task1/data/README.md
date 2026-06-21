# 数据目录

## object_a/ — 真实多视角

- `images/`：放置 `*.jpg` / `*.png`（建议 30 张以上）
- 或 `video.mp4`：环绕视频（脚本自动抽帧）

## object_c/ — 单图物体

- `teddy_rgba.png`：仓库自带 RGBA 输入
- `rgba.png`：运行前执行 `ln -sf teddy_rgba.png rgba.png`

## background/ — Mip-NeRF 360

下载 [360_v2.zip](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) 后：

```bash
unzip 360_v2.zip bicycle -d background/
# → background/bicycle/images/
```

场景名在 `config/task1.yaml` → `background.scene` 中配置（默认 `bicycle`）。
