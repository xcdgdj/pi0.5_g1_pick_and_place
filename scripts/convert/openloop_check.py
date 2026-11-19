"""开环（教师强制）验证：用数据集真实观测逐帧喂服务端，比较模型动作与演示动作。

不接仿真、不看闭环，只看模型本身：
  1. chunk[0]（当前帧动作）与演示的偏差
  2. 整块 50 步的平均偏差
  3. 模型"预计在第几帧合爪"与演示第几帧合爪的对比

用法: python openloop_check.py [--stride 5] [--episode 0]
"""

import argparse
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/user/openpi/packages/openpi-client/src")
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

ROOT = "/home/user/pi/lerobot_g1_pick"
PROMPT = "Pick up the red cube and place it in the box."
ARM = 7
HORIZON = 50
CLOSED_TH = -0.0078


def load_episode(ep: int, root: str = ROOT):
    tag = f"episode_{ep:06d}"
    t = pq.read_table(f"{root}/data/chunk-000/{tag}.parquet")
    if "frame_index" in t.column_names:
        t = t.sort_by("frame_index")
    states = np.stack([np.asarray(x.as_py(), np.float32) for x in t["observation.state"]])
    actions = np.stack([np.asarray(x.as_py(), np.float32) for x in t["action"]])
    frames = {}
    for cam in ("base_0_rgb", "right_wrist_0_rgb"):
        cap = cv2.VideoCapture(f"{root}/videos/chunk-000/observation.images.{cam}/{tag}.mp4")
        buf = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            buf.append(np.transpose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (2, 0, 1)))
        cap.release()
        frames[cam] = np.stack(buf)
    return states, actions, frames


def first_closed(seq: np.ndarray) -> int | None:
    idx = np.nonzero(seq[:, ARM] > CLOSED_TH)[0]
    return int(idx[0]) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--host", default="192.168.110.51")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=ROOT, help="LeRobot 数据集根目录")
    args = ap.parse_args()

    states, actions, frames = load_episode(args.episode, args.root)
    T = min(len(states), min(len(v) for v in frames.values()))
    demo_close = first_closed(actions[:T])
    print(f"[数据] 帧数={T}  演示合爪帧={demo_close}", flush=True)

    pol = WebsocketClientPolicy(host=args.host, port=args.port)
    rows = []
    for i in range(0, T - HORIZON, args.stride):
        obs = {
            "state": states[i],
            "images": {k: frames[k][i] for k in frames},
            "prompt": PROMPT,
        }
        pred = np.asarray(pol.infer(obs)["actions"], np.float32)
        demo = actions[i : i + HORIZON]
        m = min(len(pred), len(demo))
        e0 = np.abs(pred[0, :ARM] - demo[0, :ARM]).max()
        em = np.abs(pred[:m, :ARM] - demo[:m, :ARM]).mean()
        g0 = float(pred[0, ARM])
        rows.append((i, e0, em, g0, first_closed(pred)))
        print(
            f"[帧 {i:3d}] chunk[0]误差={e0:.4f} 整块误差={em:.4f} "
            f"夹爪={g0:+.4f} 模型合爪于块内第 {rows[-1][4]} 步"
            f"（对应帧 {i + rows[-1][4] if rows[-1][4] is not None else '-'}）",
            flush=True,
        )

    arr = np.array([[r[1], r[2]] for r in rows])
    print(f"\n[汇总] chunk[0] 误差 mean={arr[:,0].mean():.4f} max={arr[:,0].max():.4f}")
    print(f"[汇总] 整块误差    mean={arr[:,1].mean():.4f} max={arr[:,1].max():.4f}")
    pred_close = [i + r[4] for r in rows if r[4] is not None and r[3] < CLOSED_TH]
    if pred_close:
        print(f"[汇总] 模型预测合爪帧: mean={np.mean(pred_close):.0f} "
              f"min={min(pred_close)} max={max(pred_close)}（演示 {demo_close}）")
    else:
        print("[汇总] 全程没有预测出合爪")


if __name__ == "__main__":
    main()
