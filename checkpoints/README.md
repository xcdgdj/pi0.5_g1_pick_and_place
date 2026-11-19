# 权重

## 包含内容

| 文件 | 说明 |
|---|---|
| `15000/` | 第 15000 步 checkpoint，5.4GB，54 个文件 |
| `19999/` | 第 19999 步 checkpoint，5.4GB，54 个文件 |
| `norm_stats.json` | 训练时使用的归一化统计（从 `lerobot_g1_pick/` 复制） |
| `MD5SUMS` | 全部文件的校验清单，用于核对 U 盘拷贝完整性 |

每个 checkpoint 目录结构：

```
15000/
├── _CHECKPOINT_METADATA
├── assets/          空目录（本站训练未写入）
├── params/          Orbax 参数树，5.1GB，推理必需
└── train_state/     优化器状态，300MB，仅续训需要
```

注意：`assets/` 是空的，所以归一化统计不在 checkpoint 里，必须使用本目录附带的 `norm_stats.json`。

## 部署位置

解压后放到 openpi 的标准路径：

```
$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora/15000
$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora/19999
```

## 完整性校验

```bash
cd <权重根目录>
md5sum -c MD5SUMS
# 应全部输出 OK，共 54 行
```

## 启动推理服务

```bash
cd $OPENPI_ROOT
OPENPI_POLICY_SEED=3 CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_g1_pick_cube_lora \
  --policy.dir $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora/15000
```

`OPENPI_POLICY_SEED` 决定流匹配采样的噪声起点，同一 seed 下动作可复现；实测 seed 3 在开环 rollout 中完整成功。
