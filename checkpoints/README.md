# 权重部署

权重文件体积超过 GitHub 单文件限制（100 MB），不随仓库分发，需单独获取（随附 `MD5SUMS`，
部署后执行 `md5sum -c MD5SUMS` 校验）。

本仓库涉及三份权重，用途不同：

| 权重 | 用途 | 部署路径 |
|---|---|---|
| v2 `1999/` | **v6 训练的初始化权重**（50 集基线，训练 20000 步） | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/` |
| v6 `1999/` | **推理与评估用**（README 第 1 节成绩单所用） | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999/` |
| pi0.5 base | v2 训练的初始化权重（官方发布，非本项目产物） | 按官方 openpi 文档下载 |

## 每个 checkpoint 目录的结构

```
1999/
├── params/           54 个文件，约 5.1 GB —— 推理必需
├── train_state/      续训必需（含优化器状态与步数）
└── assets/           空；归一化统计用本仓库 checkpoints/norm_stats.json
```

## 部署命令

```bash
OPENPI_ROOT=/home/user/openpi

# v6 训练起点
mkdir -p $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified
cp -r 1999_v2 $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999

# v6 推理权重
mkdir -p $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified
cp -r 1999_v6 $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999

md5sum -c MD5SUMS
```

三个要点：

1. **路径必须带两层同名目录**（`配置名/实验名/`），少一层 `serve_policy.py` 找不到 checkpoint。
2. 只做推理时 `train_state/` 可以不给，但 `params/` 必须完整（54 个文件）。
3. 归一化统计与 checkpoint 绑定：v2 权重配 50 集版本的 norm_stats，v6 权重配本仓库
   `checkpoints/norm_stats.json`（60 集）。混用会直接导致动作尺度错误。

## norm_stats.json

`checkpoints/norm_stats.json` 是 60 集 v6 数据集的归一化分位数，与
`dataset/lerobot_g1_pick_v6/norm_stats.json` 为同一份。关键值：

| 字段 | q01 | q99 |
|---|---|---|
| actions[7]（夹爪左指） | −0.02072 | +0.00518 |
| actions[8]（夹爪右指） | −0.02072 | +0.00518 |
| state[7..8] | −0.02044 / −0.02052 | +0.00518 / +0.00473 |
