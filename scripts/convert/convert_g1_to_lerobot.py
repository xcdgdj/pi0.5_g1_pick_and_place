#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 单臂遥操采集数据 (HDF5 + 两路 MP4) → LeRobot v2.1 数据集

适配现在的纯单臂数据格式：
    - HDF5 里的关节/夹爪已经是右臂维度（不再含左臂），脚本**不做任何左右切片**，
      state/action 维度直接按实际读取的列数动态确定；
    - 只有两路相机：Camera.mp4 → base_0_rgb，wrist_r.mp4 → right_wrist_0_rgb；
    - 不生成任何黑图 / 左腕视角。

用法示例:
    python convert_g1_to_lerobot.py \
        --dataset-root /home/user/OrcaManipulation_g1_pick_tube/src/examples/dataCollection/dataset/g1_pick_usda/pi \
        --output /home/user/pi/lerobot_g1_pick

输入（每个 episode 一个目录）:
    <episode_uuid>/record/proprio_stats.hdf5
    <episode_uuid>/video/Camera.mp4
    <episode_uuid>/video/wrist_r.mp4

输出（LeRobot v2.1）:
    <output>/
    ├── data/chunk-000/episode_000000.parquet
    ├── videos/chunk-000/observation.images.base_0_rgb/episode_000000.mp4
    ├── videos/chunk-000/observation.images.right_wrist_0_rgb/episode_000000.mp4
    └── meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl}

依赖:
    pip install h5py numpy pandas pyarrow opencv-python scipy
    系统需要 ffmpeg（用于把视频重编码成稳定的 30fps H.264）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ────────────────────────────── 常量配置 ──────────────────────────────

# 输出视频统一 30fps（LeRobot / π0.5 训练常用）
FPS_OUT = 30

# 滑动平均窗口（200Hz 下 10 帧 = 50ms）
SMOOTH_WINDOW = 10

# 图像尺寸：源视频为 640x480，输出统一烘成 224x224
# （等比缩放 + 黑边填充，不裁剪、不变形，与 OpenPI 训练端的 resize_with_pad 一致）
IMG_H, IMG_W = 224, 224

# 图像 feature key → 源视频文件名
VIDEO_SOURCES = {
    "observation.images.base_0_rgb": "Camera.mp4",
    "observation.images.right_wrist_0_rgb": "wrist_r.mp4",
}

DEFAULT_TASK = "Pick up the red cube and place it in the box."

# 双臂数据里右臂/左臂的切片位置：
#   joint   (N, 14) = [左臂 7 | 右臂 7]
#   gripper (N,  4) = [左夹爪 2 | 右夹爪 2]
# 场景里左臂仍然存在（只是删了相机），所以采集到的仍是 14+4 维，
# 转换时用 --arm right 抽出右臂即可，无需重录数据。
ARM_SLICES = {
    "right": (slice(7, 14), slice(2, 4)),
    "left": (slice(0, 7), slice(0, 2)),
    "none": (slice(None), slice(None)),
}

# 已知维度的可读名字（仅用于 info.json 的 names 字段，方便调试；
# 维度对不上时自动退回通用命名，不影响训练）
KNOWN_NAMES = {
    9: [  # 右臂 7 + 右夹爪 2
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint",
        "right_wrist_yaw_joint", "right_dex1_joint1", "right_dex1_joint2",
    ],
    18: [  # 双臂 14 + 双夹爪 4（兼容旧数据）
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
        "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        "left_dex1_joint1", "left_dex1_joint2", "right_dex1_joint1", "right_dex1_joint2",
    ],
}


def build_state_names(dim: int) -> list[str]:
    """按实际维度生成 names：已知维度用可读名，否则用 state_0..state_{D-1}。"""
    return KNOWN_NAMES.get(dim, [f"state_{i}" for i in range(dim)])


# ────────────────────────────── 工具函数 ──────────────────────────────

def run_ffmpeg(args: list[str]) -> None:
    """调用系统 ffmpeg，失败时抛出带 stderr 的异常。"""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(cmd)}\n{proc.stderr}")


def probe_video(path: Path) -> tuple[int, float]:
    """读取视频的 (帧数, fps)。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if frames <= 0 or fps <= 0:
        raise RuntimeError(f"视频元数据异常: {path} frames={frames} fps={fps}")
    return frames, fps


def resample_video(src: Path, dst: Path, n_frames: int) -> None:
    """把源视频重采样成固定 30fps、固定帧数、224x224 的 H.264。

    - `fps=30` 按时间轴均匀重采样（丢帧/补帧），避免时间漂移；
    - `scale+pad` 等比缩放到 224x224 并上下补黑边（640x480 → 224x168 → 224x224），
      与 OpenPI 训练端 resize_with_pad 的行为一致，推理端无需再缩放；
    - `-frames:v` 截断到统一长度，使两路视频与 parquet 行数严格一致；
    - yuv420p 是 LeRobot / decord 解码的必要像素格式。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", str(src),
        "-vf", (
            f"fps={FPS_OUT},"
            f"scale={IMG_W}:{IMG_H}:force_original_aspect_ratio=decrease,"
            f"pad={IMG_W}:{IMG_H}:(ow-iw)/2:(oh-ih)/2"
        ),
        "-frames:v", str(n_frames),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst),
    ])


def smoothing(x: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """沿时间轴做滑动平均（窗口 window），抑制遥操作手抖。

    用 scipy.ndimage.uniform_filter1d 而不是 np.convolve(mode='same')：
    - 前者边界用边缘值延拓(mode='nearest')，首尾不会被 0 拉低下沉；
    - 后者需要手动补边，容易在 episode 开头/结尾引入跳变。
    """
    return uniform_filter1d(x.astype(np.float64), size=window, axis=0, mode="nearest")


def interp_to_timestamps(t_src: np.ndarray, x_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """把 200Hz 的 (N, D) 数据按时间戳线性插值到目标时间轴 (M, D)。

    t_src: HDF5 的相对时间（秒），单调递增
    t_dst: 视频帧对应的相对时间 i / 30
    逐维 np.interp（分段线性），区间外自动裁剪到首/尾值。
    """
    out = np.empty((len(t_dst), x_src.shape[1]), dtype=np.float64)
    for d in range(x_src.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, x_src[:, d])
    return out


def image_stats(video_path: Path) -> dict:
    """视频每通道像素统计（LeRobot 的 image stats 用）。

    每 10 帧采样一次以控制耗时。LeRobot 的 aggregate_stats 要求每个特征
    同时具备 mean/std/min/max（形状 (3,1,1)）和 count（形状 (1,)），
    缺一个都会在 LeRobotDataset 初始化时报 KeyError。
    """
    cap = cv2.VideoCapture(str(video_path))
    sums = np.zeros(3, dtype=np.float64)
    sqs = np.zeros(3, dtype=np.float64)
    vmin = np.ones(3, dtype=np.float64)
    vmax = np.zeros(3, dtype=np.float64)
    sampled = 0
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frames % 10 == 0:
            rgb = frame[:, :, ::-1].astype(np.float64) / 255.0  # BGR→RGB, 0~1
            pixels = rgb.reshape(-1, 3)
            sums += pixels.mean(axis=0)
            sqs += (pixels ** 2).mean(axis=0)
            vmin = np.minimum(vmin, pixels.min(axis=0))
            vmax = np.maximum(vmax, pixels.max(axis=0))
            sampled += 1
        frames += 1
    cap.release()
    if sampled == 0:
        return {}
    mean = sums / sampled
    var = np.maximum(sqs / sampled - mean ** 2, 1e-12)
    return {
        "mean": mean.reshape(3, 1, 1).tolist(),
        "std": np.sqrt(var).reshape(3, 1, 1).tolist(),
        "min": vmin.reshape(3, 1, 1).tolist(),
        "max": vmax.reshape(3, 1, 1).tolist(),
        "count": [int(frames)],
    }


def hdf5_dataset_keys(f: h5py.File) -> dict:
    """兼容两种 HDF5 字段命名：采集脚本原始命名 / 通用命名。

    关节数据优先取 joint_qpos，其次 action/joint/position；
    夹爪数据优先取 gripper，其次 action/effector/position。
    """
    keys = {}
    if "joint_qpos" in f:
        keys["joint"] = "joint_qpos"
    elif "action/joint/position" in f:
        keys["joint"] = "action/joint/position"
    if "gripper" in f:
        keys["gripper"] = "gripper"
    elif "action/effector/position" in f:
        keys["gripper"] = "action/effector/position"
    if keys.get("joint") is None or keys.get("gripper") is None:
        raise RuntimeError(
            f"HDF5 里找不到关节/夹爪数据，实际 keys: {list(f.keys())}"
        )
    return keys


def read_episode(ep_dir: Path, arm: str = "right") -> dict:
    """读取单个 episode：HDF5 轨迹（按 arm 抽单臂）+ 视频路径 + 语言指令。"""
    record_dir = ep_dir / "record"
    h5_candidates = sorted(record_dir.glob("*.hdf5")) if record_dir.is_dir() else []
    if not h5_candidates:
        raise RuntimeError(f"未找到 HDF5: {ep_dir}")
    h5_path = h5_candidates[0]

    with h5py.File(h5_path, "r") as f:
        k = hdf5_dataset_keys(f)
        # 不做左右切片，直接使用 HDF5 里的实际维度
        joint = np.asarray(f[k["joint"]][:], dtype=np.float64)      # (N, J)
        grip = np.asarray(f[k["gripper"]][:], dtype=np.float64)     # (N, G)
        t = np.asarray(f["time_step"][:], dtype=np.float64)         # (N,)
        task = None
        if "task_description" in f.attrs:
            task = f.attrs["task_description"]
        elif "task_description" in f:
            task = f["task_description"][()]
        if isinstance(task, bytes):
            task = task.decode("utf-8")
        if not task:
            task = DEFAULT_TASK

    if joint.ndim != 2 or grip.ndim != 2:
        raise RuntimeError(f"{h5_path} 维度异常: joint={joint.shape}, gripper={grip.shape}")

    # 按 --arm 抽单臂：只有数据确实是双臂维度时才切，单臂数据原样使用
    j_slice, g_slice = ARM_SLICES[arm]
    if arm != "none" and joint.shape[1] >= 14 and grip.shape[1] >= 4:
        joint = joint[:, j_slice]
        grip = grip[:, g_slice]

    return {
        "h5_path": h5_path,
        "time": t,
        "x": np.concatenate([joint, grip], axis=1),   # (N, D)
        "dim": joint.shape[1] + grip.shape[1],
        "task": str(task),
        "video_dir": ep_dir / "video",
    }


def discover_episodes(root: Path) -> list[Path]:
    """找出 root 下所有含 record/*.hdf5 的 episode 目录，按名称排序。"""
    eps = sorted(
        {p.parent.parent for p in root.glob("*/record/*.hdf5")},
        key=lambda d: d.name,
    )
    if not eps:
        raise RuntimeError(f"{root} 下没有找到 */record/*.hdf5")
    return eps


# ────────────────────────────── 主转换流程 ──────────────────────────────

def convert(dataset_root: Path, output: Path, overwrite: bool = False, arm: str = "right") -> None:
    episodes = discover_episodes(dataset_root)

    if output.exists() and any((output / d).exists() for d in ("data", "videos", "meta")):
        if not overwrite:
            raise RuntimeError(
                f"输出目录已有数据集: {output}（加 --overwrite 覆盖，或换一个 --output）"
            )
        for d in ("data", "videos", "meta"):
            shutil.rmtree(output / d, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    print(f"共发现 {len(episodes)} 个 episode，输出到 {output}")

    meta_episodes: list[dict] = []
    meta_stats: list[dict] = []
    global_index = 0
    total_frames = 0
    dim: int | None = None   # 由第一集决定，后续集必须一致

    for ep_idx, ep_dir in enumerate(episodes):
        chunk = ep_idx // 1000
        ep = read_episode(ep_dir, arm=arm)
        video_dir: Path = ep["video_dir"]
        if dim is None:
            dim = ep["dim"]
            print(f"state/action 维度（从数据动态读取）: {dim}")
        elif ep["dim"] != dim:
            raise RuntimeError(
                f"维度不一致: 第 0 集 {dim} 维, {ep_dir.name} 为 {ep['dim']} 维"
            )

        # ── 1. 时间轴对齐：以视频帧数为基准 ─────────────────────────────
        base_src = video_dir / VIDEO_SOURCES["observation.images.base_0_rgb"]
        right_src = video_dir / VIDEO_SOURCES["observation.images.right_wrist_0_rgb"]
        for v in (base_src, right_src):
            if not v.exists():
                raise RuntimeError(f"缺少视频: {v}")

        base_frames, base_fps = probe_video(base_src)
        right_frames, right_fps = probe_video(right_src)
        # 以 30fps 重编码后，各路的帧数 = 源时长 × 30
        m_base = int(round(base_frames / base_fps * FPS_OUT))
        m_right = int(round(right_frames / right_fps * FPS_OUT))
        # 取较短者，保证两路视频 + parquet 三者行数一致
        M = min(m_base, m_right)
        if M <= 0:
            raise RuntimeError(f"视频长度异常: base={m_base}, right={m_right}")

        # 视频时间轴（秒）：第 i 帧对应 i / 30
        t_dst = np.arange(M, dtype=np.float64) / FPS_OUT

        # ── 2. 滤波 + 插值：200Hz → 30Hz ────────────────────────────────
        t_src = ep["time"] - ep["time"][0]          # 从 0 开始的相对时间
        x_s = smoothing(ep["x"])                    # 滑动平均，压手抖
        state = interp_to_timestamps(t_src, x_s, t_dst)     # (M, D)
        # 按需求：action 与 state 使用同一组位置指令。
        # 若要做"动作 = 下一步状态"，改为 np.roll(state, -1, axis=0) 即可。
        action = state.copy()

        # ── 3. 视频落盘（两路）─────────────────────────────────────────
        for key, filename in VIDEO_SOURCES.items():
            dst = output / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            resample_video(video_dir / filename, dst, M)

        # ── 4. Parquet ─────────────────────────────────────────────────
        df = pd.DataFrame({
            "observation.state": [state[i].astype(np.float32) for i in range(M)],
            "action": [action[i].astype(np.float32) for i in range(M)],
            "timestamp": t_dst.astype(np.float32),
            "frame_index": np.arange(M, dtype=np.int64),
            "episode_index": np.full(M, ep_idx, dtype=np.int64),
            "index": np.arange(global_index, global_index + M, dtype=np.int64),
            "task_index": np.zeros(M, dtype=np.int64),
        })
        parquet_path = (
            output / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        # ── 5. meta 统计（含单臂数据的分位数）──────────────────────────
        def stats_of(a: np.ndarray) -> dict:
            return {
                "mean": a.mean(axis=0).tolist(),
                "std": a.std(axis=0).tolist(),
                "min": a.min(axis=0).tolist(),
                "max": a.max(axis=0).tolist(),
                "q01": np.quantile(a, 0.01, axis=0).tolist(),
                "q99": np.quantile(a, 0.99, axis=0).tolist(),
                "count": [int(M)],
            }

        stats = {
            "observation.state": stats_of(state),
            "action": stats_of(action),
        }
        # 图像统计（可选，便于训练侧做可视化/归一化）
        for key in VIDEO_SOURCES:
            vs = image_stats(
                output / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
            )
            if vs:
                stats[key] = vs

        meta_episodes.append({
            "episode_index": ep_idx,
            "tasks": [ep["task"]],
            "length": int(M),
        })
        meta_stats.append({"episode_index": ep_idx, "stats": stats})

        global_index += M
        total_frames += M
        print(
            f"  [{ep_idx:03d}] {ep_dir.name}  {M} 帧  "
            f"(源 {base_frames}f@{base_fps:.2f} / {right_frames}f@{right_fps:.2f})  "
            f"D={dim}  task={ep['task']}"
        )

    # ── 6. meta/info.json ──────────────────────────────────────────────
    if dim is None:
        raise RuntimeError("没有可转换的 episode")
    state_names = build_state_names(dim)

    def image_feature() -> dict:
        return {
            "dtype": "video",
            "shape": [IMG_H, IMG_W, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": IMG_H,
                "video.width": IMG_W,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS_OUT,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    task = meta_episodes[0]["tasks"][0] if meta_episodes else DEFAULT_TASK
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_dex1_right_arm",
        "total_episodes": len(meta_episodes),
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": len(meta_episodes) * len(VIDEO_SOURCES),
        "total_chunks": (len(meta_episodes) + 999) // 1000,
        "chunks_size": 1000,
        "fps": FPS_OUT,
        "splits": {"train": f"0:{len(meta_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [dim],
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [dim],
                "names": state_names,
            },
            "observation.images.base_0_rgb": image_feature(),
            "observation.images.right_wrist_0_rgb": image_feature(),
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(output / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)

    # ── 7. meta/episodes.jsonl / episodes_stats.jsonl / tasks.jsonl ────
    with open(output / "meta" / "episodes.jsonl", "w", encoding="utf-8") as f:
        for row in meta_episodes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output / "meta" / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for row in meta_stats:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n")

    print(f"\n完成: {len(meta_episodes)} 集 / {total_frames} 帧 / D={dim} → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 单臂 HDF5 + 双路视频 → LeRobot v2.1")
    parser.add_argument(
        "--dataset-root", required=True,
        help="采集输出根目录（其下每个子目录含 record/*.hdf5 与 video/*.mp4）",
    )
    parser.add_argument("--output", required=True, help="LeRobot 数据集输出目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 data/videos/meta")
    parser.add_argument(
        "--arm", choices=["right", "left", "none"], default="right",
        help="双臂数据里抽哪只手臂（默认 right；已是单臂数据时自动不切）",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"输入目录不存在: {root}")

    convert(root, out, overwrite=args.overwrite, arm=args.arm)


if __name__ == "__main__":
    main()
