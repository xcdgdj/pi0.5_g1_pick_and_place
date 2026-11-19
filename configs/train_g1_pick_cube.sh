#!/usr/bin/env bash
# pi05_base + LoRA 微调：宇树 G1 单臂抓红方块（50 集，9 维 state/action，2 路相机）
# 目标机器：2×RTX 4090 24GB（无 NVLink）
#
# 用法:
#   ./train_g1_pick_cube.sh debug    # 100 步，验证不 OOM / 能存盘
#   ./train_g1_pick_cube.sh full     # 20000 步正式训练
set -euo pipefail

# ── 路径（按服务器实际情况改）──────────────────────────────────────────
OPENPI_DIR="${OPENPI_DIR:-/home/user/openpi}"
# 数据集绝对路径；也要同步修改 config.py 里 pi05_g1_pick_cube_lora 的 repo_id
DATASET_DIR="${DATASET_DIR:-/home/user/pi/lerobot_g1_pick}"
CONFIG_NAME="pi05_g1_pick_cube_lora"

# ── 分布式通信：4090 无 NVLink，强制关闭 P2P / InfiniBand 防死锁 ────────
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# JAX 每张卡最多占用 90% 显存（防系统卡死）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=true

MODE="${1:-debug}"
case "$MODE" in
  debug)
    STEPS=100
    EXP_NAME="g1_pick_cube_debug"
    EXTRA_ARGS="--overwrite"
    ;;
  full)
    STEPS=20000
    EXP_NAME="g1_pick_cube_lora"
    EXTRA_ARGS=""
    ;;
  *)
    echo "用法: $0 [debug|full]" >&2
    exit 1
    ;;
esac

cd "$OPENPI_DIR"

# 归一化统计只算一次（缺失时才跑，全量 15k 帧约 1 分钟）
# 统计文件直接落在数据集根目录 norm_stats.json，拷数据时一并带走
if [ ! -f "$DATASET_DIR/norm_stats.json" ]; then
  echo ">>> 计算 norm stats ..."
  uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

# OpenPI 是 JAX 框架：单进程会自动用满所有可见 GPU 做数据并行，
# 不需要 torchrun --nproc_per_node。batch_size=4 会按 2 张卡拆成每卡 2。
uv run scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --num-train-steps "$STEPS" \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --num-workers 8 \
  $EXTRA_ARGS
