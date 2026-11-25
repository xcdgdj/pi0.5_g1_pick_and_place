# pi0.5 G1 抓取放置复现包（v6）

G1 人形机器人（右臂 6 关节 + DEX1 两指夹爪）在 OrcaStudio 自建场景「pi」中
**抓起红色方块并放进盒子**。基于 pi0.5（PaliGemma 2B + 300M 动作专家，LoRA 微调）。

本仓库包含：OrcaStudio 场景（level `pi` + 搭建用 4 个 xml）、60 集训练数据、
采集→转换→训练→推理→评估全链路脚本、openpi 侧全部改动、以及 v6 权重的实测闭环成功率。

---

## 1. 结果摘要（v6，闭环 rollout）

**评估协议**：v6 checkpoint 1999、每次从干净场景 reset、600 步上限、30 Hz、
每 10 步重推理、保护规则全部关闭、成功判据 `in_box_hold ≥ 30`（方块连续 30 步在盒内并贴底静止）。

### 基线（方块在默认位置）

| 试验 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 结果 | ✅ | ✅ | ✅ | ❌滑脱 | ✅ | ✅ | ✅ | ❌盒外 | ✅ | ✅ |
| 抓起步 | 337 | 188 | 197 | 196 | 258 | 209 | 166 | 233 | 178 | 260 |
| 落盒步 | 456 | 388 | 344 | — | 466 | 423 | 370 | — | 408 | 424 |

| 试验 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 结果 | ✅ | ✅ | ❌滑脱 | ✅ | ✅滑入 | ❌滑脱 | ✅ | ✅ | ✅ | ✅ |
| 抓起步 | 179 | 216 | 439 | 257 | 187 | 267 | 214 | 209 | 237 | 219 |
| 落盒步 | 448 | 505 | — | 461 | 329 | 555 | — | 395 | 468 | 391 |

- **成功率 16/20 = 80%**，抓起率 20/20 = 100%
- 完成例总步数 359~585（中位 447）
- 失败 4 次：3 次搬运滑脱（方块在闭合指令下被指头挤出，落桌）、1 次松爪位置在盒口外（12.49 cm）

### 方块位置偏移泛化（每次全新 reset，同一协议）

| 场景 | 次数 | 成功 | 成功率 | 抓起率 | 失败构成 |
|---|---|---|---|---|---|
| 基线（默认位置） | 20 | 16 | **80%** | 100% | 3 滑脱落桌 + 1 松爪盒外 |
| x +2 cm | 5 | 4 | 80% | 100% | 1 次 600 步预算内未完成（截断时仍夹持、距盒心 3.27 cm） |
| x −2 cm | 5 | 5 | **100%** | 100% | — |
| y +2 cm | 5 | 3 | 60% | 80% | 1 次到位不合爪 + 1 次保持 29/30 步 |
| y +4 cm | 5 | 1 | **20%** | 20% | 4 次接近到 4.1~5.5 cm 后悬停、全程全开不合爪 |
| **合计** | **40** | **29** | **72.5%** | 87.5% | — |

参考：v4（56 集、无最后 4 条放置示范）同协议 10 次成功率 50%、自主松爪率 40%。

### 主要失败模式

| 模式 | 频次 | 特征 |
|---|---|---|
| 搬运滑脱 | 4/40 | 抬起后 13~39 步内，夹爪保持闭合指令（+0.005）时实际 q0 从 +0.003 突跳到 +0.022，TCP→方块 3~4 cm 跳到 12~15 cm，方块回到初始区；方块是被闭合中的指头**挤出**，不是松爪丢失 |
| 松爪位置偏外 | 1/40 | 夹持与松爪动作正常，松爪时距盒心 12.49 cm（盒口半径 8 cm） |
| 到位不合爪 | 4/40 | 仅在 y+4 cm 偏移出现：TCP→方块 4.1~5.5 cm 悬停，夹爪指令全程 −0.021（全开） |
| 步数预算不足 | 2/40 | 合爪/松爪偏晚，600 步截断时仍在搬运或下放途中 |

---

## 2. 目录结构

```
.
├── models/                     MuJoCo 模型（搭建 level pi 的 xml 源头）
│   ├── g1_pick.xml                 机器人本体（OrcaStudio 版，DEX1 手指网格不带碰撞）
│   ├── assets/                     网格与材质
│   └── g1_table_scene/             table_only.xml / box_container.xml / red_cube_5cm.xml（纯 MuJoCo 也可单独跑）
├── dataset/
│   ├── lerobot_g1_pick_v6/    ★ 训练数据集（60 集 / 17105 帧，LeRobot v2.1）
│   │   ├── data/chunk-000/         episode_000000~000059.parquet
│   │   ├── videos/chunk-000/       base_0_rgb / right_wrist_0_rgb 两路 mp4
│   │   ├── meta/                   info.json / episodes.jsonl / episodes_stats.jsonl / tasks.jsonl
│   │   └── norm_stats.json         归一化分位数（训练与推理共用）
│   ├── g1_pick_usda/pi/       原始采集（55 个 episode 目录，HDF5 + 两路 mp4）
│   └── episode_uuid_mapping.json    v6 episode_index ↔ 原始目录 ↔ 起始行 的映射表
├── openpi_changes/            相对官方 openpi 仓库的改动（按原相对路径存放）
│   ├── scripts/{serve_policy.py, train.py}
│   └── src/openpi/
│       ├── models/{gemma.py, pi0_config.py}
│       ├── policies/{g1_pick_cube_policy.py, g1_omnipicker_policy.py}
│       └── training/{config.py, data_loader.py}
├── scripts/
│   ├── collection/            遥操作采集链路（Pico + OSC + DEX1），含 DAgger 纠正采集
│   ├── convert/               HDF5 → LeRobot v2.1、合并、体检、开环验证
│   └── infer/                 闭环推理、HDF5 回放、LeRobot 示范回放、消融
├── orcastudio/                OrcaStudio 场景「pi」（自建，整目录交付）
│   ├── levels/pi/pi.prefab         level 文件，把 `pi` 整个文件夹拖入项目 `levels/` 即可打开
│   └── assets/importer/            场景全部资产（G1 网格 42 + 桌面/盒子/方块 50，共 92 个文件）
├── configs/train_g1_pick_cube.sh       训练启动脚本（v6）
└── checkpoints/
    ├── norm_stats.json              v6 归一化统计（与数据集内同一份）
    └── README.md                    权重部署与校验说明（权重本体单独分发）
```

---

## 3. 环境

| 组件 | 版本/说明 |
|---|---|
| OrcaStudio / OrcaLab | 任意版本即可，**无需订阅任何资产包**（场景由本仓库自带文件搭建） |
| OrcaGym | conda 环境 `orcalab_26.9.1` |
| openpi | 官方 Physical-Intelligence/openpi，打上 `openpi_changes/` 里的文件 |
| 训练硬件 | 2× RTX 4090 24GB（无 NVLink） |
| 推理 | 单卡即可（策略服务约 8.5 GB 显存） |

---

## 4. 复现顺序

> **脚本路径约定**：`scripts/collection/` 与 `scripts/infer/` 是按 OrcaManipulation 仓库
> （`OrcaManipulation_g1_pick_tube`）内的相对路径镜像的，运行时需拷回该仓库对应位置
> （`src/examples/dataCollection/`），并把 `scripts/convert/`、`configs/` 放在任意工作目录即可。
> 采集与推理都在 conda 环境 `orcalab_26.9.1` 下执行，通过 gRPC 连接 `localhost:50051` 的
> OrcaGym 服务（OrcaStudio 里先启动场景）。

### 4.1 导入场景（自建 level「pi」）

场景**不是订阅资产包**，而是用下面 4 个 MuJoCo xml 在 OrcaStudio 里搭建后保存为 level `pi`：

| 实体名（level 内） | 来源 xml | 内容 |
|---|---|---|
| `g1_pick_usda` | `models/g1_pick.xml` | G1 本体（右臂 6 关节力矩 motor + DEX1 两指） |
| `table_only_usda` | `models/g1_table_scene/table_only.xml` | 桌面（`table_top` + 4 条腿） |
| `box_container_usda` | `models/g1_table_scene/box_container.xml` | 目标盒子 |
| `red_cube_5cm_usda` | `models/g1_table_scene/red_cube_5cm.xml` | 5 cm 红色方块 |

打开方式（两步）：

**第一步：放场景文件**

把 `orcastudio/levels/pi` 这个文件夹整个拖进 OrcaStudio 项目的 `levels/` 目录。
重启 OrcaStudio 后，level 列表里会出现 `pi`，点开就是完整场景。

**第二步：放模型文件（只在新电脑 / 新项目时需要）**

`pi.prefab` 里没有模型，只存了路径和资产 ID。模型全在 `orcastudio/assets/importer/` 下，
共 92 个文件，**整个目录原样复制**：

```bash
# PROJ = 放着 levels/ 和 assets/ 的那一层目录
PROJ="<OrcaStudio 项目根>"
cp -r orcastudio/assets/importer/* "$PROJ/assets/importer/"
```

| 内容 | 文件 | 数量 |
|---|---|---|
| G1 机器人 | `g1_pick/meshes/*.stl`、`*.obj` | 42 |
| 桌面 | `table_only.usda` 及其导入产物 `table_only_*` | 17 |
| 盒子 | `box_container.usda` + `box_container_*` + `default_box_container_*.usda.azmodel` | 21 |
| 红色方块 | `red_cube_5cm.usda` 及其导入产物 `red_cube_5cm_*` | 12 |

为什么第一眼只看到 G1 和盒子带路径：prefab 里 G1 的网格、盒子的网格写的是文件路径
（`assets/importer/...`），而桌面和方块的网格组件只存了资产 ID（GUID，没有路径文字），
所以**不能只挑带路径的文件复制**，必须整目录放。那 3 个 `.usda` 是桌面/盒子/方块的源文件，
OrcaStudio 启动时自动导入；GUID 由文件路径决定，路径不变就能和 prefab 对上。

复制完按下面 4 条确认（依次输出 `42`、`17`、`21`、`12`）：

```bash
ls "$PROJ/assets/importer/g1_pick/meshes" | wc -l
ls "$PROJ/assets/importer"/table_only* | wc -l
ls "$PROJ/assets/importer"/box_container* "$PROJ/assets/importer"/default_box_container* | wc -l
ls "$PROJ/assets/importer"/red_cube_5cm* | wc -l
```

少文件不会报错，只是对应零件变成白色占位块（机器人缺零件、盒子缺一面），
物理和脚本仍然正常。已在本机跑过的不需要重复做，先跑上面的命令确认即可。

**场景内的名字**：实体 `g1_pick_usda`、`box_container_usda`、`table_only_usda`、
`red_cube_5cm_usda`；相机 `Camera`（第三人称全局）、`camera_right`（右腕）。
采集/推理统一用 `--agent_name g1_pick_usda --level pi`。

> `models/g1_table_scene/` 是不依赖 OrcaStudio 的纯 MuJoCo 桌面场景素材，可按需单独使用。

### 4.2 数据（已附带，可跳过）

训练直接用 `dataset/lerobot_g1_pick_v6/`，无需重新转换。

若要重新采集或转换：

```bash
# 遥操作采集（Pico）：左 Grip 开始/结束一集，右 Grip 丢弃本集
python scripts/collection/g1_dex1_collection_tele.py --level pi --agent_name g1_pick_usda

# DAgger 纠正采集：模型闭环 → 偏离/停滞时 Pico 接管 → 整段落盘
python scripts/collection/dagger_release_collect.py \
  --init-state <失败状态快照.npz> --lerobot-root <数据集根> --level pi

# 转换（DAgger 集必须带 --trim-takeover 和夹爪反解）
python scripts/convert/convert_g1_to_lerobot.py \
  --dataset-root dataset/g1_pick_usda/pi \
  --output ./lerobot_g1_pick_v6 \
  --overwrite --trim-takeover --gripper-action-from-force

# 多版本合并（重编号 episode / 合并 videos+meta / 删除过期 norm_stats）
python scripts/convert/append_lerobot_episodes.py \
  --base <旧数据集> --add <新增数据集> --out <输出数据集>
```

转换后必须重算归一化统计：

```bash
cd $OPENPI_ROOT
uv run scripts/compute_norm_stats.py --config-name pi05_g1_pick_cube_lora_v6
```

### 4.3 训练

```bash
# 1) 把改动打到 openpi
cp -r openpi_changes/* $OPENPI_ROOT/

# 2) 确认 config.py 里 pi05_g1_pick_cube_lora_v6 的两处绝对路径
#    - repo_id      → 本仓库 dataset/lerobot_g1_pick_v6
#    - weight_loader→ v2/1999 params 目录（见 4.4）

# 3) 从 v2 的 1999 步继续微调 2000 步
cd $OPENPI_ROOT && bash configs/train_g1_pick_cube.sh full
```

v6 训练配置：

| 项 | 值 |
|---|---|
| 初始化 | v2 的 `1999/params`（50 集基线，20000 步训练结果） |
| 数据 | 60 集 / 17105 帧，ep50–59 为 DAgger 纠正集 |
| 采样权重 | ep50–59 各 ×3（加权后 DAgger 帧占 35.3%，未加权 15.4%） |
| 模型 | pi05 + LoRA（PaliGemma 2B LoRA + 300M 动作专家 LoRA），冻结视觉主干 |
| action_horizon / dim | 50 / 32 |
| batch / grad_accum | 4 / 4（等效 16） |
| 学习率 | CosineDecay，warmup 100，peak 2e-5 → 2e-6（2000 步） |
| 优化器 | AdamW，weight_decay 1e-4，clip 1.0 |
| 步数 | 2000，每 250 步存 checkpoint |

### 4.4 权重部署

权重超过 GitHub 单文件限制（5.4 GB > 100 MB），单独分发（随附 `MD5SUMS`）。

| 权重 | 用途 | 路径 |
|---|---|---|
| v2 `1999/` | **v6 训练的初始化权重** | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/` |
| v6 `1999/` | **推理/评估用**（本文成绩单所用） | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999/` |

```bash
md5sum -c MD5SUMS     # 全部 OK
```

checkpoint 目录里 `params/` 是推理必需（约 5.1 GB），`train_state/` 仅续训需要。
`assets/` 为空，归一化统计用本仓库 `checkpoints/norm_stats.json`。

### 4.5 推理

```bash
# 策略服务（训练机 / 本机 GPU）
OPENPI_POLICY_SEED=3 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_g1_pick_cube_lora_v6 \
  --policy.dir $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999

# 闭环推理（仿真机 / 同一台机器）
python scripts/infer/infer_g1_pick.py --policy server \
  --host 127.0.0.1 --port 8000 --execution-steps 10 --max-steps 600 \
  --success-hold 30
```

### 4.6 复现成功率评估

评估就是按第 1 节的协议跑 `infer_g1_pick.py`，每次一个全新进程（保证干净 reset）：

```bash
for i in $(seq 1 10); do
  python scripts/infer/infer_g1_pick.py --policy server --host 127.0.0.1 --port 8000 \
    --execution-steps 10 --max-steps 600 --success-hold 30 --trace /tmp/run_$i.json
done
```

- 成功判据由脚本内的 `success_radius / success_floor_tol / success_speed / success_hold` 实现，
  `--success-hold 30` 即“连续 30 步满足在盒内且静止”
- 偏移泛化测试：在 OrcaStudio 里把方块沿世界 x/y 轴平移，或改布局后 reset，其余不变
- 记录项：抓起步、首次实际张开步、落盒步、`in_box_hold`、桌面接触、失败原因

---

## 5. 数据集说明（v6）

| 项 | 值 |
|---|---|
| 集数 / 帧数 | 60 / 17105（30 fps，224×224，两路相机） |
| ep0–49 | 基础遥操作示范（含早期失败样本，v1 起逐步筛选） |
| ep50–59 | DAgger 纠正集：远距提前合爪、近距接触未抓稳、停滞接管、搬运后放置 |
| 动作维度 | 9 维 = 右臂 7 关节（rad）+ DEX1 两指（m，负=张开） |
| 夹爪范围 | action q01 = −0.02072，q99 = +0.00518（全开 −0.0207 / 全闭 +0.0052） |
| 语言指令 | `Pick up the red cube and place it in the box.` |

`dataset/episode_uuid_mapping.json` 给出每个 episode 对应的原始采集目录与 HDF5 起始行：

```json
{"episode_index": 52, "uuid": "fbb84e10-2b45-448e-8a04-49d329578484",
 "hdf5_start_row": 601, "takeover_row": 600, "hdf5_total_rows": 3131,
 "lerobot_frames": 380, "est_frames_from_rows": 379.6, "frame_err": 0.4,
 "match_err_rad": 0.0178, "match": "exact"}
```

校验方式（55 集全部通过）：

- 普通集（ep1–46 中的 45 集）：首帧手臂关节与 HDF5 对应行逐元素最大差 ≤ 7e-5 rad
- DAgger 集（ep50–59）：`(hdf5_total_rows − takeover_row) / 200 × 30` 与 LeRobot 帧数吻合，
  误差 ≤ 0.4 帧；首帧差 1.6e-2~1.9e-2 rad（30 Hz 抽帧带来的插值差）
- ep0 / ep11 / ep47–49 这 5 集标 `ambiguous`：均为“从初始姿态快速失败”的短集（120~369 帧），
  原始采集目录已被后续采集覆盖，仓库不含其原始数据。这 5 集已固化在 `lerobot_g1_pick_v6`
  中，训练复现不受影响；只有“从原始数据重做转换”时缺这 5 条

---

## 6. 关键约定（改动前必读）

| 项 | 约定 |
|---|---|
| 动作维度 | 9 维：右臂 7 关节 + DEX1 两指 |
| 动作语义 | 前 7 维是**相对 chunk 首帧的增量**（`delta_action=True`），夹爪 2 维是**绝对值** |
| 单位 | 臂 = rad，夹爪 = m（开 −0.020，闭 +0.0245；数据集尺度 −0.02072 ~ +0.00518） |
| 控制方式 | 6 个臂关节是 `<motor>`（力矩），必须走 PD + 重力补偿，不能直接写角度 |
| 夹爪控制 | 位置阻抗 `tau = kp(q_d − q) − kd·dq`，kp=1500、kd=15；指令由电机力反解 |
| 控制周期 | 数据 fps=30，一个控制步 = 33 × 1ms 子步 |
| 归一化 | q01/q99 逐维映射，与 `norm_stats.json` 绑定；换数据集必须重算 |
| 推理节奏 | 每 10 步重取观测并重新推理（50 步动作块只执行前 10 步） |

---

## 7. 已知问题

- **搬运滑脱**：4/40 次。方块在夹爪闭合指令下被指头挤出（q0 +0.003 → +0.022），
  发生在抬起后 13~39 步内，与方块在指间的初始位置有关
- **横向偏移敏感**：y+4 cm 时抓取率降到 20%，模型接近到位后不输出合爪
- **松爪位置精度**：成功例松爪时距盒心 2.0~10.0 cm，盒口半径 8 cm，边界成功依赖判据余量
- **步数预算**：合爪/松爪偏晚时 600 步内无法完成保持判据（约 5%）
- 闭环仍存在协变量偏移，状态漂移后策略倾向按演示节奏继续，不做主动纠偏

---

## 8. 许可

仅供合作研究使用；G1 机器人模型版权归 Unitree 所有。
