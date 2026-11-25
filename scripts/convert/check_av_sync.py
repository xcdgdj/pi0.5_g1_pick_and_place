"""极简验证：LeRobot episode 的 parquet 行数与 mp4 帧数是否对齐，并抽夹爪突变前后三帧存图。

不需要仿真环境，只读数据集。

用法:
    /home/user/openpi/.venv/bin/python /home/user/pi/g1_lerobot/check_av_sync.py
    /home/user/openpi/.venv/bin/python /home/user/pi/g1_lerobot/check_av_sync.py --episode 49 --gripper-dim 8
"""

import argparse
import os

import cv2
import numpy as np
import pyarrow.parquet as pq

CAMS = ["base_0_rgb", "right_wrist_0_rgb"]


def count_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frame(path: str, idx: int):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"读取失败: {path} 第 {idx} 帧")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/pi/lerobot_g1_pick")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--gripper-dim", type=int, default=7, help="action 里的夹爪维度（7 或 8）")
    ap.add_argument("--outdir", default="/home/user/pi/g1_lerobot/sync_frames")
    args = ap.parse_args()

    ep = f"{args.episode:06d}"
    parquet = os.path.join(args.root, "data", "chunk-000", f"episode_{ep}.parquet")
    actions = np.stack(pq.read_table(parquet).column("action").to_pylist())
    n_rows = actions.shape[0]
    print(f"[parquet] {parquet}\n          行数 = {n_rows}  action 维度 = {actions.shape[1]}")

    videos = {
        cam: os.path.join(
            args.root, "videos", "chunk-000", f"observation.images.{cam}", f"episode_{ep}.mp4"
        )
        for cam in CAMS
    }
    n_frames = {}
    for cam, path in videos.items():
        n_frames[cam] = count_frames(path)
        gap = n_frames[cam] - n_rows
        verdict = "OK" if gap == 0 else f"差 {gap:+d} 帧"
        print(f"[video]   {cam:18s} 帧数 = {n_frames[cam]:5d}  {verdict}")

    # 夹爪突变帧：相邻帧差的绝对值最大处；t 取突变后的第一帧
    grip = actions[:, args.gripper_dim]
    diffs = np.abs(np.diff(grip))
    t = int(np.argmax(diffs)) + 1
    print("\n[夹爪] 维度 %d，范围 %.4f ~ %.4f" % (args.gripper_dim, grip.min(), grip.max()))
    print(f"       突变帧 t = {t}，|Δ| = {diffs[t - 1]:.4f}（{grip[t - 4]:.4f} → {grip[t + 3]:.4f}）")
    print("       t 附近逐帧值：" + ", ".join(f"{i}:{grip[i]:+.3f}" for i in range(max(0, t - 3), min(n_rows, t + 4))))

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\n[抽帧] 保存到 {args.outdir}")
    for cam, path in videos.items():
        for k in (-2, 0, 2):
            idx = t + k
            if not 0 <= idx < n_frames[cam]:
                print(f"  跳过 {cam} 第 {idx} 帧（越界）")
                continue
            img = read_frame(path, idx)
            name = f"ep{ep}_t{idx}_{cam}.png"
            cv2.imwrite(os.path.join(args.outdir, name), img)
            print(f"  {name}  shape={img.shape}")


if __name__ == "__main__":
    main()
