# pi0.5 G1 抓取放置复现包（v6）

G1 人形机器人（右臂 6 关节 + DEX1 两指夹爪）在 OrcaStudio 自建场景「pi」中
**抓起红色方块并放进盒子**。基于 pi0.5（PaliGemma 2B + 300M 动作专家，LoRA 微调）。

本仓库包含：OrcaStudio 场景（level `pi` + 搭建用 4 个 xml）、60 集训练数据、
采集→转换→训练→推理→评估全链路脚本、openpi 侧全部改动、以及 v6 权重的实测闭环成功率。

---

## 0. 开始前：设置 4 个路径变量

下面所有命令都直接使用这 4 个变量。**每开一个新终端都要先执行一次**（或把它们写进 `~/.bashrc`）：

```bash
export REPO=$HOME/pi05-release
export OPENPI_ROOT=$HOME/openpi
export ORCA_REPO=$HOME/OrcaManipulation
export PROJ=$HOME/orca_studio/OrcaSim/OrcaSim
```

| 变量 | 指向 | 怎么确定 |
|---|---|---|
| `REPO` | 本仓库解压/克隆后的根目录 | 里面有 `README.md`、`dataset/`、`models/` |
| `OPENPI_ROOT` | 官方 openpi 仓库根目录 | 里面有 `pyproject.toml`、`scripts/train.py` |
| `ORCA_REPO` | OrcaManipulation 仓库根目录 | 里面有 `src/examples/dataCollection/` |
| `PROJ` | OrcaStudio 项目根目录 | 里面有 `Levels/` 和 `Assets/`（Linux 区分大小写，以你项目里实际名字为准） |

```bash
# 检查 4 个变量是否都对（4 行都要存在）
ls -d $REPO $OPENPI_ROOT $ORCA_REPO $PROJ
```

如果某行报 `No such file or directory`，就把上面 `export` 里对应的路径改成你机器上的真实路径，再往下做。

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
│   │   ├── videos/chunk-000/       observation.images.base_0_rgb / observation.images.right_wrist_0_rgb 两路 mp4
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
│   ├── levels/pi/pi.prefab         level 文件
│   └── assets/importer/            模型源文件 45 个（g1_pick/meshes 42 + 3 个场景 usda）
├── configs/train_g1_pick_cube.sh       训练启动脚本（v6）
└── checkpoints/
    ├── norm_stats.json              v6 归一化统计（与数据集内同一份）
    └── README.md                    权重部署与校验说明（权重本体单独分发）
```

---

## 3. 环境安装（从头照抄）

### 3.1 需要什么

| 组件 | 版本 / 说明 |
|---|---|
| 操作系统 | Linux（命令按 bash 写） |
| OrcaStudio / OrcaLab | 任意近期版本，**无需订阅任何资产包**（场景由本仓库自带文件搭建） |
| Python（采集/推理） | conda 环境 `orcalab_26.9.1`，Python 3.12 + orca-gym 26.9.1 |
| openpi（训练/策略服务） | 官方 Physical-Intelligence/openpi，commit `981483d`，用 `uv` 管理 |
| 训练硬件 | 2× RTX 4090 24GB（无 NVLink）；单卡也能跑，只是慢 |
| 推理硬件 | 单卡即可（策略服务约 8.5 GB 显存） |

### 3.2 下载三个仓库

```bash
cd $HOME
git clone https://github.com/xcdgdj/pi0.5_g1_pick_and_place.git pi05-release
git clone -b g1_pick_tube https://github.com/openverse-orca/OrcaManipulation.git
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && git checkout 981483d
```

正常结果（最后一条应输出 `981483d`）：

```bash
git -C $HOME/openpi rev-parse --short HEAD
```

> 如果你把仓库放在了别的路径，回到第 0 节把 4 个变量改成新路径。

### 3.3 安装 openpi

先装 `uv`（已装过会提示已存在，可跳过）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

再装依赖（第一次会下载较多包）：

```bash
cd $OPENPI_ROOT
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

正常结果：`uv sync` 最后输出 `Installed ... packages`，没有红色报错。

把本仓库的 openpi 改动打进去：

```bash
cp -r $REPO/openpi_changes/* $OPENPI_ROOT/
cp $REPO/configs/train_g1_pick_cube.sh $OPENPI_ROOT/
```

正常结果（能打印 `openpi ok`）：

```bash
cd $OPENPI_ROOT
uv run python -c "from openpi.training import config; print('openpi ok')"
```

### 3.4 安装采集/推理用的 conda 环境

```bash
conda create -n orcalab_26.9.1 python=3.12 -y
conda activate orcalab_26.9.1
pip install "orca-gym==26.9.1" h5py pyyaml opencv-python pyarrow pandas
pip install -e $OPENPI_ROOT/packages/openpi-client
```

正常结果（应输出 `26.9.1`）：

```bash
python -c "import orca_gym, cv2, pyarrow, pandas, openpi_client; print(orca_gym.__version__)"
```

> 以后每次跑**采集或推理**，都要先 `conda activate orcalab_26.9.1`。
> **训练和策略服务**不用 conda，它们走 openpi 自己的 `uv` 环境。

### 3.5 把本项目的脚本放进 OrcaManipulation

```bash
cp $REPO/scripts/collection/g1_dex1_collection_tele.py $ORCA_REPO/src/examples/dataCollection/
cp $REPO/scripts/collection/dagger_release_collect.py     $ORCA_REPO/src/examples/dataCollection/
cp $REPO/scripts/infer/*.py                               $ORCA_REPO/src/examples/dataCollection/
cp $REPO/scripts/collection/g1_dex1_data_storage.py       $ORCA_REPO/src/dataStorage/
```

正常结果（第一条打印出脚本路径；第二条输出 `SAME`）：

```bash
ls $ORCA_REPO/src/examples/dataCollection/infer_g1_pick.py
diff $ORCA_REPO/src/dataStorage/g1_dex1_data_storage.py $REPO/scripts/collection/g1_dex1_data_storage.py && echo SAME
```

为什么要拷这些文件：

- `infer_g1_pick.py`、`dagger_release_collect.py`、`replay_*.py` 在上游仓库里没有，必须从本仓库拷。
- `g1_dex1_data_storage.py` 相对上游多了一处：每集 HDF5 写入固定语言指令
  `Pick up the red cube and place it in the box.`。不覆盖的话，采集出的 Episode 没有
  `task_description`，转换后 LeRobot 的 task 字段为空，训练的 prompt 就错了。

### 3.6 改 openpi 配置里的两条绝对路径（必做）

`openpi_changes` 里的 `config.py` 用的是作者机器上的绝对路径。**只跑推理也要做这一步**，
因为策略服务会从 `repo_id` 指向的数据集目录读取归一化统计 `norm_stats.json`。

```bash
cd $OPENPI_ROOT
sed -i "s|repo_id=\"/home/user/pi/lerobot_g1_pick_v6\"|repo_id=\"$REPO/dataset/lerobot_g1_pick_v6\"|" src/openpi/training/config.py
sed -i "s|/home/user/openpi/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/params|$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/params|g" src/openpi/training/config.py
```

正常结果（第一条打印出新数据集路径；第二条输出 `4`）：

```bash
grep -n "repo_id=\"$REPO/dataset/lerobot_g1_pick_v6\"" src/openpi/training/config.py
grep -c "$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/params" src/openpi/training/config.py
```

---

## 4. 复现顺序

### 4.0 只想复现成功率（最短路径）

按顺序做这 5 步，不用采集、不用训练：

1. 装环境并改配置路径：第 3.2 ~ 3.6 节
2. 放场景并打开 level `pi`：4.1 节
3. 下载 v6 权重并解压：4.4 节第一段
4. 启动策略服务：4.5 节「终端 A」
5. 跑闭环推理：4.5 节「终端 B」

### 4.1 导入场景（自建 level「pi」）

场景**不是订阅资产包**，而是用下面 4 个 MuJoCo xml 在 OrcaStudio 里搭建后保存为 level `pi`：

| 实体名（level 内） | 来源 xml | 内容 |
|---|---|---|
| `g1_pick_usda` | `models/g1_pick.xml` | G1 本体（右臂 6 关节力矩 motor + DEX1 两指） |
| `table_only_usda` | `models/g1_table_scene/table_only.xml` | 桌面（`table_top` + 4 条腿） |
| `box_container_usda` | `models/g1_table_scene/box_container.xml` | 目标盒子 |
| `red_cube_5cm_usda` | `models/g1_table_scene/red_cube_5cm.xml` | 5 cm 红色方块 |

**第一步：把 level 文件放进 OrcaStudio 项目**

```bash
mkdir -p "$PROJ/Levels/pi"
cp -r $REPO/orcastudio/levels/pi/. "$PROJ/Levels/pi/"
```

如果你项目里这个目录叫 `levels`（全小写），把命令里的 `Levels` 改成实际名字。

**第二步：把模型源文件放进项目**

```bash
mkdir -p "$PROJ/Assets/Importer"
cp -r $REPO/orcastudio/assets/importer/. "$PROJ/Assets/Importer/"
```

**第三步：检查**（应输出 `pi.prefab` 和 `42`）

```bash
ls "$PROJ/Levels/pi/pi.prefab"
ls "$PROJ/Assets/Importer/g1_pick/meshes" | wc -l
```

**第四步：打开场景**

重启 OrcaStudio，在 level 列表里双击 `pi`。
场景显式加载后，点击工具栏的「运行 / Play」启动仿真（策略脚本连的 gRPC 服务就在这里启动）。

正常结果（有 `LISTEN` 行即成功）：

```bash
ss -lnt | grep 50051
```

如果打开后桌面/盒子/方块显示成白色方块：在 OrcaStudio 资产浏览器里找到
`table_only.usda`、`box_container.usda`、`red_cube_5cm.usda`，右键「重新导入」，
等导入完成后重新打开 level。`Cache/` 下的 `.azmodel` 等文件是自动生成的，仓库里不放，也不用管。

**场景内的名字**：实体 `g1_pick_usda`、`box_container_usda`、`table_only_usda`、
`red_cube_5cm_usda`；相机 `Camera`（第三人称全局）、`camera_right`（右腕）。
采集/推理统一用 `--agent_name g1_pick_usda --level pi`，两路相机用 `Camera,wrist_r`。

> `models/g1_table_scene/` 是不依赖 OrcaStudio 的纯 MuJoCo 桌面场景素材，可按需单独使用。

### 4.2 数据（已附带，可跳过）

训练直接用 `dataset/lerobot_g1_pick_v6/`，**不需要**重新采集或转换。

只有你要重新采集 / 转换时才继续（在 `orcalab_26.9.1` 环境里执行）：

```bash
conda activate orcalab_26.9.1
cd $ORCA_REPO

# 遥操作采集（Pico）：左 Grip 开始/结束一集，右 Grip 丢弃本集
python src/examples/dataCollection/g1_dex1_collection_tele.py \
  --level pi --agent_name g1_pick_usda --cameras Camera,wrist_r
```

数据会写到 `$ORCA_REPO/src/examples/dataCollection/dataset/g1_pick_usda/pi/<目录>/`
（`record/*.hdf5` + `video/*.mp4`）。

```bash
# DAgger 纠正采集：模型闭环 → 偏离/停滞时 Pico 接管 → 整段落盘
# 需要先按 4.5 节启动策略服务；默认从零位（手臂 6 关节全 0）+ 夹爪全开开始
python src/examples/dataCollection/dagger_release_collect.py \
  --level pi --lerobot-root $REPO/dataset/lerobot_g1_pick_v6
```

> `--init-state` 可以额外指定一个失败状态快照（`.npz`，含 `q9` / `qpos` / `qvel`），
> 从失败现场继续采集。本仓库不附带这类快照，所以上面的命令不传它。

转换（HDF5 + 两路 mp4 → LeRobot v2.1）：

```bash
conda activate orcalab_26.9.1
NEW=$HOME/lerobot_g1_pick_v6_rebuild
cd $REPO
python scripts/convert/convert_g1_to_lerobot.py \
  --dataset-root $ORCA_REPO/src/examples/dataCollection/dataset/g1_pick_usda/pi \
  --output $NEW \
  --overwrite --trim-takeover --gripper-action-from-force
```

> 重新转换只能得到 55 集：ep0 / ep11 / ep47–49 这 5 条的原始采集目录已被后续采集覆盖，
> 仓库不含它们的原始数据（详见第 5 节）。训练请直接用自带的 60 集数据集。

要把新采集的数据并进已有数据集：

```bash
python scripts/convert/append_lerobot_episodes.py \
  --base $REPO/dataset/lerobot_g1_pick_v6 --add $NEW --out $HOME/lerobot_merged
```

换了数据集就要重算归一化统计：先把 4.3 第 2 步的 `repo_id` 改成新数据集路径，再执行

```bash
cd $OPENPI_ROOT
uv run scripts/compute_norm_stats.py --config-name pi05_g1_pick_cube_lora_v6
```

### 4.3 训练

**前置**：训练从 v2/1999 起，先把 4.4 节里 v2 权重的「下载 + 解压」做完。

> **起点约定**：下面 4.3 写的是 v6 当初**怎么来的**（从 v2/1999 起训 2000 步），仅用于复现 v6。
> 后续任何继续训练一律以 **v6/1999** 为起点，不再从 v2 起训：
>
> - 新数据再微调：新建 config（如 `pi05_g1_pick_cube_lora_v7`），`weight_loader` 指向
>   `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999/params`，学习率降到 peak 1e-5
> - 同数据继续练更多步：同一 config 加 `--resume`，`checkpoint_dir` 指向 v6 输出目录，
>   会从 `train_state/` 恢复 optimizer / 步数 / EMA（先把 v6 的 `1999/` 备份一份）

**第 1 步：确认 openpi 改动已就位**

第 3.3 节已经拷过；如果你跳过了，补一次：

```bash
cp -r $REPO/openpi_changes/* $OPENPI_ROOT/
cp $REPO/configs/train_g1_pick_cube.sh $OPENPI_ROOT/
```

**第 2 步：确认路径已改（第 3.6 节做过就跳过，只做检查）**

```bash
cd $OPENPI_ROOT
grep -n "repo_id=\"$REPO/dataset/lerobot_g1_pick_v6\"" src/openpi/training/config.py
grep -c "$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/params" src/openpi/training/config.py
```

第一条打印出新路径、第二条输出 `4` 就说明改好了；没有的话回到第 3.6 节执行那两条 `sed`。

**第 3 步：先跑 100 步，确认不 OOM、能存盘**

```bash
export OPENPI_DIR=$OPENPI_ROOT
export DATASET_DIR=$REPO/dataset/lerobot_g1_pick_v6
export INIT_PARAMS=$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/1999/params

cd $OPENPI_ROOT && bash train_g1_pick_cube.sh debug
```

正常结果：日志持续打印 `step` 和 loss，不出现 `CUDA out of memory`；
结束后生成 `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_debug_unified/99/`：

```bash
ls $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_debug_unified/99
```

**第 4 步：正式训练 2000 步**

```bash
cd $OPENPI_ROOT && bash train_g1_pick_cube.sh full
```

正常结果：checkpoint 每 250 步存一次，最终目录为
`$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999/`：

```bash
ls $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified
```

> 训练脚本启动时会检查 `$DATASET_DIR/norm_stats.json`；本仓库数据集已自带，不会重算。

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

### 4.4 权重（GitHub Release 分卷下载）

**为什么放 Release**：每个 checkpoint 5.4 GB（`params/` 5.1 GB + `train_state/` 300 MB），
超过仓库单文件 100 MB 限制。两个 checkpoint 各一个 Release，每个切成 3 卷
（Release 单附件上限 2 GiB）：

| Release | 用途 | 卷 | 合并后 MD5 |
|---|---|---|---|
| [`weights-v6-1999`](https://github.com/xcdgdj/pi0.5_g1_pick_and_place/releases/tag/weights-v6-1999) | **推理/评估**（本文成绩单） | 2.0 + 2.0 + 1.5 GB | `eef5c8701cc3088cd432ce6b2ab63941` |
| [`weights-v2-1999`](https://github.com/xcdgdj/pi0.5_g1_pick_and_place/releases/tag/weights-v2-1999) | **重训 v6 的初始化权重** | 2.0 + 2.0 + 1.5 GB | `2a993088493448f4ff52c1b0d6a7b378` |

#### v6（推理/评估用，必下）

```bash
mkdir -p $HOME/weights && cd $HOME/weights
BASE=https://github.com/xcdgdj/pi0.5_g1_pick_and_place/releases/download/weights-v6-1999
for f in pi05_g1_pick_cube_lora_v6_1999.tar.00.part \
         pi05_g1_pick_cube_lora_v6_1999.tar.01.part \
         pi05_g1_pick_cube_lora_v6_1999.tar.02.part MD5SUMS.parts; do
  curl -L -O $BASE/$f
done
```

正常结果（`md5sum -c` 输出 3 行 `OK`；合并 MD5 等于 `eef5c8701cc3088cd432ce6b2ab63941`）：

```bash
md5sum -c MD5SUMS.parts
cat pi05_g1_pick_cube_lora_v6_1999.tar.0*.part | md5sum
```

解压（`mkdir -p` 不能省，`tar -C` 要求目标目录已存在；压缩包根目录就是 `1999/`）：

```bash
mkdir -p $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified
cat pi05_g1_pick_cube_lora_v6_1999.tar.0*.part | tar -xf - \
  -C $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified
ls $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified
```

正常结果：输出 `1999`。

装了 `gh` 的话可以一条命令下完（省掉上面的 `curl` 循环，后面校验/解压照旧）：

```bash
gh release download weights-v6-1999 -R xcdgdj/pi0.5_g1_pick_and_place -D $HOME/weights
gh release download weights-v2-1999 -R xcdgdj/pi0.5_g1_pick_and_place -D $HOME/weights_v2
```

#### v2（只有重新训练 v6 才需要）

```bash
mkdir -p $HOME/weights_v2 && cd $HOME/weights_v2
BASE=https://github.com/xcdgdj/pi0.5_g1_pick_and_place/releases/download/weights-v2-1999
for f in pi05_g1_pick_cube_lora_v2_1999.tar.00.part \
         pi05_g1_pick_cube_lora_v2_1999.tar.01.part \
         pi05_g1_pick_cube_lora_v2_1999.tar.02.part MD5SUMS.parts; do
  curl -L -O $BASE/$f
done

md5sum -c MD5SUMS.parts                                   # 3 行都要 OK
cat pi05_g1_pick_cube_lora_v2_1999.tar.0*.part | md5sum   # 2a993088493448f4ff52c1b0d6a7b378

mkdir -p $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified
cat pi05_g1_pick_cube_lora_v2_1999.tar.0*.part | tar -xf - \
  -C $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified
```

正常结果：`ls $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified` 输出 `1999`。

#### 放到哪

| 你要做的事 | 需要 | 解压到 |
|---|---|---|
| 只跑推理、复现成功率 | v6 Release | `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/` |
| 还要重新训练 v6 | v2 + v6 两个 Release | v6 同上；v2 解压到 `$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v2/g1_pick_cube_lora_v2_unified/` |

路径那两层目录名（`pi05_..._v6/` 和 `g1_pick_cube_lora_v6_unified/`）必须原样保留，
openpi 按这个名字找权重，改了就报找不到。

#### 拷完检查（应输出 `6` 和约 `5.1G`）

```bash
CKPT=$OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999
ls $CKPT/params | wc -l && du -sh $CKPT/params
```

`1999/` 目录内容：

```
1999/
├── params/                5.1 GB，6 个条目（orbax 格式，要整个目录拷，不能只挑文件）
│                          ← 推理必需
├── train_state/           300 MB，优化器状态，只有"接着训练"才用；只推理可以删
├── assets/                空，归一化统计用仓库里的 checkpoints/norm_stats.json
└── _CHECKPOINT_METADATA
```

为什么有两个版本：

- v2/1999：v6 训练的起点（50 集基线训 20000 步的结果），只在重训时用
- v6/1999：最终权重，本文第 1 节的 80% 成功率就是用它跑出来的

### 4.5 推理

跑之前确认 3 件事：

1. OrcaStudio 已打开 level `pi` 并点了「运行」，`ss -lnt | grep 50051` 有输出
2. v6 权重已按 4.4 节解压到 `.../g1_pick_cube_lora_v6_unified/1999`
3. 推理脚本已在 `$ORCA_REPO/src/examples/dataCollection/`（第 3.5 节拷过）

**终端 A：启动策略服务**（openpi 目录，`uv` 环境；第一次加载权重较慢，等日志出现服务启动再动）

```bash
cd $OPENPI_ROOT
OPENPI_POLICY_SEED=3 CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_g1_pick_cube_lora_v6 \
  --policy.dir $OPENPI_ROOT/checkpoints/pi05_g1_pick_cube_lora_v6/g1_pick_cube_lora_v6_unified/1999
```

正常结果：日志出现 `策略采样种子已设为 3`、`Creating server (host: ..., ip: ...)`，进程保持运行不退出。

**终端 B：跑闭环推理**（OrcaManipulation 目录，conda 环境）

```bash
conda activate orcalab_26.9.1
cd $ORCA_REPO
python src/examples/dataCollection/infer_g1_pick.py \
  --host 127.0.0.1 --port 8000 --execution-steps 10 --max-steps 600 \
  --success-hold 30 --lerobot-root $REPO/dataset/lerobot_g1_pick_v6
```

正常结果：依次出现

```
[初始姿态] 数据集 [...] / 写入仿真 [...] / 实际 [...]
[策略] 连接 ws://127.0.0.1:8000 ...
[策略] 已连接 metadata=...
[相机] simulate_index 可用 ...
[循环] 步    0  TCP→方块=...cm ...
...
[结果] 放置成功（步 XXX，盒内稳定 30 步）...
```

`--lerobot-root` 必须指向本仓库的数据集：脚本要从里面读 `norm_stats.json`、
`meta/info.json` 和 `data/chunk-000/episode_000000.parquet`（用于字段顺序校验）。
初始姿态固定为手臂零位 + 夹爪全开，不再依赖 episode 0 首帧。
脚本内默认值是旧的 `/home/user/pi/lerobot_g1_pick`，不传会直接报找不到文件。

如果报 `ModuleNotFoundError: openpi_client`，说明 3.4 节的
`pip install -e $OPENPI_ROOT/packages/openpi-client` 没执行，补执行一次再重跑。

### 4.6 复现成功率评估

评估就是按第 1 节的协议跑 `infer_g1_pick.py`，每次一个全新进程：

```bash
conda activate orcalab_26.9.1
cd $ORCA_REPO
for i in $(seq 1 10); do
  python src/examples/dataCollection/infer_g1_pick.py \
    --host 127.0.0.1 --port 8000 --execution-steps 10 --max-steps 600 \
    --success-hold 30 --lerobot-root $REPO/dataset/lerobot_g1_pick_v6 \
    --trace /tmp/run_$i.npz
  # 本轮结束后在 OrcaStudio 里 reset 场景，让方块/盒子回到 level 初始位置
done
```

**reset 的确切含义**：新进程启动时会 `env.reset()`，并把机器人右臂摆到零位、夹爪张开
（见 [infer_g1_pick.py#L483-L502](scripts/infer/infer_g1_pick.py#L483-L502)）；
方块和盒子的位置来自 level 初始状态，所以要保证每轮开始前方块在默认位置——脚本不会替你把方块搬回去。

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

`dataset/episode_uuid_mapping.json` 给出每个 episode 对应的原始采集目录与 HDF5 起始行。
注意 `uuid` 字段对 ep0–46 的早期集其实是**原始目录名**（纯数字，如 `1`、`11`、`12`），
只有后期集和 DAgger 集才是真 UUID（如 `fbb84e10-…`）；两种都能在
`dataset/g1_pick_usda/pi/` 下找到对应目录（55/55 命中）。示例：

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

## 8. 文件完整性自检

在 `$REPO` 下执行，输出应逐项对上（括号里是期望值）：

```bash
cd $REPO
echo "数据集 parquet : $(ls dataset/lerobot_g1_pick_v6/data/chunk-000/*.parquet | wc -l)  (期望 60)"
echo "数据集 mp4     : $(find dataset/lerobot_g1_pick_v6/videos -name '*.mp4' | wc -l)  (期望 120)"
echo "原始采集目录   : $(ls dataset/g1_pick_usda/pi | wc -l)  (期望 55)"
echo "场景网格       : $(ls orcastudio/assets/importer/g1_pick/meshes | wc -l)  (期望 42)"
echo "openpi 改动    : $(find openpi_changes -type f | wc -l)  (期望 8)"
echo "推理脚本       : $(ls scripts/infer/*.py | wc -l)  (期望 5)"
md5sum dataset/lerobot_g1_pick_v6/norm_stats.json checkpoints/norm_stats.json
```

最后两行 MD5 必须完全相同（两行都是 `c829c3bdd75ec1d70b774f87b61c5e42`）。

---

## 9. 许可

仅供合作研究使用；G1 机器人模型版权归 Unitree 所有。
