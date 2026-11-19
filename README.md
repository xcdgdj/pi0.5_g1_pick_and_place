# pi0.5 G1 抓取放置（pick-and-place）复现包

G1 人形机器人 + DEX1 夹爪，在 OrcaStudio「试管操作」场景里抓取红色方块并放入盒子。
包含：机器人模型、采集数据、采集/转换/推理脚本、openpi 侧改动、训练配置。

---

## 目录结构

```
.
├── models/                 MuJoCo 机器人模型（g1_pick.xml + assets 网格）
├── dataset/g1_pick_usda/pi/  遥操作采集的原始数据，50 集
│   └── <episode>/record/proprio_stats.hdf5
│       <episode>/video/{Camera,wrist_r}.mp4
├── openpi_changes/         相对官方 openpi 仓库的改动文件（按原相对路径存放）
│   ├── scripts/serve_policy.py       推理服务（含 OPENPI_POLICY_SEED 补丁）
│   ├── scripts/train.py
│   └── src/openpi/{models,training,policies}/...
├── scripts/
│   ├── collection/         遥操作采集链路（Pico + OSC + DEX1）
│   ├── convert/            HDF5 → LeRobot v2.1、体检、开环验证
│   └── infer/              闭环推理、HDF5 回放、FK 校验
├── configs/                训练启动脚本
└── checkpoints/            norm_stats.json + 权重说明（权重本体单独分发，见该目录 README）
```

## 环境

| 组件 | 版本/说明 |
|---|---|
| OrcaStudio / OrcaLab | 需订阅「试管操作」(testtube_operation) 资产包 |
| OrcaGym | conda 环境 `orcalab_26.9.1` |
| openpi | 官方 Physical-Intelligence/openpi，打上 `openpi_changes/` 里的文件 |
| 训练硬件 | 2× RTX 4090，JAX SPMD 数据并行 |

## 复现顺序

### 1. 导入场景

1. 打开 OrcaStudio，资产中心订阅 `testtube_operation`（作者 Orca）
2. 启动场景，选择 `testtube_operation`
3. 菜单「打开布局」→ 加载本仓库的 `g1_pick_tube.json`（见根目录）
4. 机器人由 `models/g1_pick.xml` 加载，`--agent_name g1_pick_usda`

### 2. 采集（可选，数据已附带）

```bash
python scripts/collection/g1_dex1_collection_tele.py \
  --level pi --agent_name g1_pick_usda
```

- 左 Grip：开始/结束一集；右 Grip：丢弃本集
- 数据落到 `dataset/g1_pick_usda/pi/<episode>/`

### 3. 转 LeRobot 数据集

```bash
python scripts/convert/convert_g1_to_lerobot.py \
  --dataset-root dataset/g1_pick_usda/pi \
  --output ./lerobot_g1_pick
```

产出 `data/chunk-000/*.parquet`、`videos/`、`meta/info.json`（fps=30）、`norm_stats.json`。

### 4. 训练

```bash
cp -r openpi_changes/* $OPENPI_ROOT/
cd $OPENPI_ROOT && bash configs/train_g1_pick_cube.sh
```

配置名 `pi05_g1_pick_cube_lora`，20000 步，LoRA 仅 1.53% 参数可训练。

### 5. 推理

```bash
# 训练机
OPENPI_POLICY_SEED=3 python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_g1_pick_cube_lora \
  --policy.dir checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora/15000

# 仿真机
python scripts/infer/infer_g1_pick.py --policy server \
  --host <训练机IP> --port 8000 --execute-steps 10
```

## 关键约定（改动前必读）

| 项 | 约定 |
|---|---|
| 动作维度 | 9 维：右臂 7 关节 + DEX1 两指 |
| 动作语义 | 前 7 维是**相对当前状态的增量**，夹爪 2 维是**绝对值** |
| 单位 | 臂 = rad，夹爪 = m（开 -0.020，闭 +0.003） |
| 控制方式 | 6 个臂关节是 `<motor>`（力矩），必须走 PD + 重力补偿，不能直接写角度 |
| 控制周期 | 数据集 fps=30，一个控制步 = 33 × 1ms 子步 |
| 归一化 | q01/q99 逐维映射，分母含 1e-6，与 checkpoint 自带 norm_stats 绑定 |

## 已知问题

- 开环 rollout 成功率约 1/5（seed 3 完整成功，seed 4 抓取成功但不松爪，seed 1/2/5 抓空）
- 闭环推理存在协变量偏移：状态漂移后策略按演示节奏继续走，不做纠偏
- 验证集（另采 10 集）wr_pitch 分布比训练集窄，不含腕部深压姿态

## 许可

仅供合作研究使用；OrcaStudio 场景资产版权归 Orca 所有。
