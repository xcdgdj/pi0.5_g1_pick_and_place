# 权重部署

checkpoint 体积超过 GitHub 单文件限制（每个 5.4 GB），放在本仓库两个 Release 里，
每个切成 3 卷（Release 单附件上限 2 GiB）。下载、校验、解压命令见根目录 `README.md` 的 4.4 节。

| 权重 | 用途 | Release | 部署路径 |
|---|---|---|---|
| v2 `1999/` | **v6 训练的初始化权重**（50 集基线，训练 20000 步） | `weights-v2-1999` | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/` |
| v6 `1999/` | **推理与评估用**（README 第 1 节成绩单所用） | `weights-v6-1999` | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999/` |
| pi0.5 base | v2 训练的初始化权重（官方发布，非本项目产物） | 官方 | 按官方 openpi 文档下载 |

合并后整包的 MD5：v6 = `eef5c8701cc3088cd432ce6b2ab63941`，v2 = `2a993088493448f4ff52c1b0d6a7b378`；
每卷的 MD5 见各 Release 里的 `MD5SUMS.parts`。

## 每个 checkpoint 目录的结构

```
1999/
├── params/           6 个条目（orbax 目录：array_metadatas、d、manifest.ocdbt、
│                     _METADATA、ocdbt.process_0、_sharding），约 5.1 GB —— 推理必需
├── train_state/      约 300 MB，续训必需（含优化器状态与步数）
├── assets/           空；归一化统计用本仓库 checkpoints/norm_stats.json
└── _CHECKPOINT_METADATA
```

## 三个要点

1. **路径必须带两层同名目录**（`配置名/实验名/`），少一层 `serve_policy.py` 找不到 checkpoint。
2. 只做推理时 `train_state/` 可以不给，但 `params/` 必须整个目录拷，不能只挑文件
   （orbax 格式里 `d/`、`manifest.ocdbt`、`_METADATA` 互相引用）。
3. 归一化统计与 checkpoint 绑定：v2 权重配 50 集版本的 norm_stats，v6 权重配本仓库
   `checkpoints/norm_stats.json`（60 集）。混用会直接导致动作尺度错误。

## norm_stats.json

`checkpoints/norm_stats.json` 是 60 集 v6 数据集的归一化分位数，与
`dataset/lerobot_g1_pick_v6/norm_stats.json` 为同一份（MD5 一致）。关键值：

| 字段 | q01 | q99 |
|---|---|---|
| actions[7]（夹爪左指） | −0.02072 | +0.00518 |
| actions[8]（夹爪右指） | −0.02072 | +0.00518 |
| state[7..8] | −0.02044 / −0.02052 | +0.00518 / +0.00473 |
