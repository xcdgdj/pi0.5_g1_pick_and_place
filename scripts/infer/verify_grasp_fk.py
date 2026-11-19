"""验证抓取数据是否真的抓到了方块：HDF5 正运动学 + LeRobot 视频抽帧。

Part A  对每个 HDF5 episode 找到夹爪闭合帧，用 MuJoCo 正运动学算 TCP 到方块中心的距离。
        < 1cm  → 动作数据没问题，问题在视频/抽帧对齐
        > 2cm  → 采集时就抓空了，属于失败样本
Part B  把这些帧（±2）从 LeRobot 的 right_wrist_0_rgb 视频里抽出来存图，
        并打印 HDF5 帧号 ↔ LeRobot 帧号的映射。

用法（需要 OrcaStudio 已加载 pi 场景；只做 FK，不推进物理）:
    python verify_grasp_fk.py
    python verify_grasp_fk.py --episodes 1 20 35 49
    python verify_grasp_fk.py --no-random
"""

import argparse
import glob
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import h5py
import mujoco
import numpy as np
from yaml import load, Loader

from scene.scene_manager import SceneManager
from conf import g1_dex1_osc_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from examples.dataCollection.replay_hdf5 import _replay_obs_callback

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
FPS_OUT = 30.0
LEROBOT_ROOT = "/home/user/pi/lerobot_g1_pick"
GPS = g1_dex1_osc_conf


def discover_episodes(root: str) -> list[str]:
    """与 convert_g1_to_lerobot.discover_episodes 相同的排序，保证 HDF5 ↔ LeRobot 编号一致。"""
    names = {
        os.path.basename(os.path.dirname(os.path.dirname(p)))
        for p in glob.glob(os.path.join(root, "*/record/*.hdf5"))
    }
    return sorted(names)


def h5_path_of(root: str, name: str) -> str:
    return os.path.join(root, name, "record", "proprio_stats.hdf5")


def find_close_frame(f: h5py.File, gripper_dim: int = 1) -> tuple[int, str]:
    """右夹爪发力（或位置突变）的第一帧。"""
    if "action/effector/motor" in f:
        force = np.asarray(f["action/effector/motor"][:], dtype=np.float64)[:, gripper_dim]
        hit = np.where(np.abs(force) > 5.0)[0]
        if hit.size:
            return int(hit[0]), f"|力|>5N（{force[hit[0]]:.1f}N）"
    pos = np.asarray(f["action/effector/position"][:], dtype=np.float64)[:, gripper_dim]
    return int(np.argmax(np.abs(np.diff(pos)))) + 1, "位置突变"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="pi")
    ap.add_argument("--agent_name", default="g1_pick_usda")
    ap.add_argument("--episodes", nargs="*", default=None, help="HDF5 episode 目录名，默认 1 + 随机 3 条")
    ap.add_argument("--no-random", action="store_true", help="只测 --episodes 指定的集")
    ap.add_argument("--task_config", default="example.yaml")
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    ap.add_argument("--outdir", default="/home/user/pi/g1_lerobot/sync_frames")
    ap.add_argument("--no-images", action="store_true", help="只做 FK，不抽帧存图")
    args = ap.parse_args()

    dataset_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", args.agent_name, args.level
    )
    all_eps = discover_episodes(dataset_root)
    if not all_eps:
        raise SystemExit(f"没找到 episode: {dataset_root}")

    targets = list(args.episodes) if args.episodes else ["1"]
    if not args.no_random:
        pool = [e for e in all_eps if e not in targets]
        targets += [str(e) for e in np.random.default_rng(0).choice(pool, 3, replace=False)]

    # ── 建环境（只用于 FK，不 step）───────────────────────────────────
    default_joint_values = {j: 0.0 for j in GPS.l_arm["joint_names"] + GPS.r_arm["joint_names"]}
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), args.task_config),
        "r",
        encoding="utf-8",
    ) as fh:
        config = load(fh, Loader=Loader)
    config.setdefault("data_collection", {})["agent_joint_prefix"] = f"{args.agent_name}_"
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="FKCheck",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=_replay_obs_callback,
        env_index=0,
        frame_skip=1,
        time_step=0.001,
        orcagym_addr=args.orcagym_addr,
        scene_manager=scene_manager,
    )
    env = manager.env

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    mj, md = gym._mjModel, gym._mjData
    arms = [env.joint(n) for n in GPS.l_arm["joint_names"] + GPS.r_arm["joint_names"]]
    grips = [env.joint(n) for n in GPS.gripper_l["joint_names"] + GPS.gripper_r["joint_names"]]
    site = env.site(GPS.r_arm["ee_site_name"])
    cube_bid = next(
        (
            i
            for i in range(mj.nbody)
            if "cube" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
        ),
        -1,
    )
    if cube_bid < 0:
        raise SystemExit("模型里找不到方块 body")
    cube_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, cube_bid)
    # 方块在采集开始时处于模型初始位姿（抓取前不会被推动）
    cube_p0 = np.asarray(mj.body_pos[cube_bid], dtype=np.float64).copy()
    print(f"site={site}  cube_body={cube_name}  初始中心={np.round(cube_p0, 4)}  "
          f"当前位置={np.round(np.asarray(md.xpos[cube_bid]), 4)}")

    def tcp_cube_distance(t: int, joints, grip_q, init_qpos) -> float:
        q = {k: np.asarray(v, dtype=np.float64) for k, v in init_qpos.items()}
        for name, val in zip(arms, joints[t]):
            q[name] = np.array([val], dtype=np.float64)
        for name, val in zip(grips, grip_q[t]):
            q[name] = np.array([val], dtype=np.float64)
        env.set_joint_qpos(q)
        env.mj_forward()
        tcp = np.asarray(env.query_site_pos_and_quat([site])[site]["xpos"]).ravel()[:3]
        return float(np.linalg.norm(tcp - cube_p0))

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 78)
    for ep in targets:
        with h5py.File(h5_path_of(dataset_root, ep), "r") as f:
            joints = np.asarray(f["action/joint/position"][:], dtype=np.float64)
            grip_q = np.asarray(f["action/effector/position"][:], dtype=np.float64)
            init_qpos = json.loads(f["initial_joint_qpos"][()].decode())
            t_h5 = np.asarray(f["time_step"][:], dtype=np.float64)
            t_close, how = find_close_frame(f)

        n = joints.shape[0]
        lo, hi = max(0, t_close - 5), min(n, t_close + 6)
        dists = [tcp_cube_distance(i, joints, grip_q, init_qpos) for i in range(lo, hi)]
        lr_ep = all_eps.index(ep)
        lr_frame = int(round((t_h5[t_close] - t_h5[0]) * FPS_OUT))
        print(f"[ep {ep:>3s}] HDF5 {n} 帧, 闭合帧 {t_close}（{how}）, "
              f"t={t_h5[t_close] - t_h5[0]:.3f}s → LeRobot ep{lr_ep:03d} 第 {lr_frame} 帧")
        print(f"        TCP→方块中心: 闭合帧 {dists[t_close - lo] * 100:.2f}cm, "
              f"窗口最小 {min(dists) * 100:.2f}cm / 最大 {max(dists) * 100:.2f}cm")
        print("        " + " ".join(f"{i}:{d * 100:.1f}" for i, d in zip(range(lo, hi), dists)))

        video = os.path.join(
            LEROBOT_ROOT, "videos", "chunk-000", "observation.images.right_wrist_0_rgb",
            f"episode_{lr_ep:06d}.mp4",
        )
        if args.no_images:
            continue
        cap = cv2.VideoCapture(video)
        saved = []
        for k in (-2, 0, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, lr_frame + k)
            ok, img = cap.read()
            if not ok:
                continue
            name = f"fk_ep{ep}_lr{lr_ep:03d}_f{lr_frame + k}_right_wrist.png"
            cv2.imwrite(os.path.join(args.outdir, name), img)
            saved.append(name)
        cap.release()
        print(f"        抽帧: {', '.join(saved)}")
    print("=" * 78)
    print(f"图片目录: {args.outdir}")


if __name__ == "__main__":
    main()
