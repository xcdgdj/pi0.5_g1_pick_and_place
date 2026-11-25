"""部分关节替换消融实验：从某一控制步起，把选定关节换成示范动作，其余保留模型输出。

目的：定位 TCP 后退（当前基线在步 40 触底后开始后退）由哪一组关节的预测偏差主导。

用法:
    python ablate_override.py --episode 0 --start 35 --end 90 --group sh_roll+sh_yaw
    python ablate_override.py --episode 0 --start 35 --end 90 --group none      # 基线
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from infer_g1_pick import ARM_DIM, OrcaGymG1, RemotePolicy, RolloutConfig

GROUPS = {
    "none": [],
    "all7": list(range(7)),
    "sh_pitch": [0],
    "sh_roll": [1],
    "sh_yaw": [2],
    "wr_roll": [4],
    "sh_roll+sh_yaw": [1, 2],
    "sh_roll+sh_yaw+wr_roll": [1, 2, 4],
    "sh_yaw+sh_pitch": [0, 2],
    "sh_yaw+wr_roll": [2, 4],
    "sh_pitch+wr_roll": [0, 4],
    "sh_yaw+sh_pitch+wr_roll": [0, 2, 4],
    "shoulder": [0, 1, 2],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=int, default=35)
    p.add_argument("--end", type=int, default=90, help="替换窗口结束步")
    p.add_argument("--max-steps", type=int, default=None, help="总步数，默认等于 --end")
    p.add_argument("--group", default="none", choices=sorted(GROUPS))
    p.add_argument("--exec-steps", type=int, default=10)
    p.add_argument("--host", default="192.168.110.51")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--lerobot-root", default="/home/user/pi/lerobot_g1_pick")
    p.add_argument("--force-open", action="store_true",
                   help="方块到盒上方（z>0.85m 且距盒心<12cm）时强制输出打开夹爪，验证物理放置")
    p.add_argument("--dump-obs", default=None, help="在方块首次到盒上方时保存该观测到 .npz")
    p.add_argument("--dump-obs2", default=None, help="方块降到释放位姿（0.80<z<0.88 且距盒心<6cm）时保存观测")
    p.add_argument("--dump-samples", type=int, default=5, help="保存观测时重复采样的次数")
    p.add_argument("--dump-stop", action="store_true", help="保存观测后立即结束")
    args = p.parse_args()

    cfg = RolloutConfig(lerobot_root=args.lerobot_root, episode=args.episode, dump_dir=None)
    robot = OrcaGymG1(cfg)
    robot.set_camera_anchor()
    policy = RemotePolicy(args.host, args.port)
    demo = robot.states9
    idx = GROUPS[args.group]
    names = [n.replace("g1_pick_usda_right_", "").replace("_joint", "") for n in robot.right_joint_names]

    robot.settle()
    print(f"[消融] 替换={args.group} {[names[i] for i in idx] or '无（基线）'}，"
          f"窗口 步 {args.start}~{args.end}", flush=True)

    tcp_hist = []
    grip_cmds = []
    placed_seen = False
    forced_step = None
    dumped = None
    dumped2 = None
    q_open_ds = float(robot.ns["q01"][ARM_DIM])

    def capture(path):
        """保存当前观测（状态/两路图像/TCP/方块/盒）并对同一观测重复采样。"""
        tcp = np.asarray(robot.env.query_site_pos_and_quat([robot.ee_sid])[robot.ee_sid]["xpos"]).ravel()[:3]
        payload = {
            "state": obs["state"], "step": step, "tcp": tcp,
            "cube": np.asarray(robot._md.xpos[robot.cube_bid], dtype=np.float64).copy(),
            "box": np.asarray(robot._md.xpos[robot.box_bid], dtype=np.float64).copy(),
        }
        for k, im in obs["images"].items():
            payload[f"img_{k}"] = im
        samples = [policy.select_action(obs)[0] for _ in range(args.dump_samples)]
        np.savez(path, samples=np.stack(samples), **payload)
        print(f"[抓取观测] 步 {step}: 方块z={robot.cube_height():.3f}m "
              f"距盒心={robot.cube_box_dist_cm():.1f}cm TCP→方块={robot.tcp_cube_cm():.2f}cm，"
              f"已保存 {path}", flush=True)
        for i, s in enumerate(samples):
            g = s[:10, ARM_DIM:].mean(axis=1)
            print(f"   采样{i + 1} 夹爪逐帧=" + " ".join(f"{v:+.3f}" for v in g)
                  + f" 最小={g.min():+.4f} 打开={'Y' if g.min() <= q_open_ds + 0.015 else 'N'}",
                  flush=True)

    step = 0
    max_steps = args.max_steps or args.end
    while step < max_steps:
        obs = robot.get_observation(step)
        chunk, _ = policy.select_action(obs)
        n = min(cfg.execution_steps, max_steps - step, len(chunk))
        if args.start <= step < args.end and idx:
            src = demo[step : step + n]
            m = min(len(src), n)
            chunk[:m, idx] = src[:m, idx]
        if args.dump_obs and dumped is None \
                and robot.cube_height() > 0.85 and robot.cube_box_dist_cm() < 12.0:
            dumped = step
            capture(args.dump_obs)
            if args.dump_stop:
                break
        if args.dump_obs2 and dumped2 is None \
                and 0.80 < robot.cube_height() < 0.88 and robot.cube_box_dist_cm() < 6.0:
            dumped2 = step
            capture(args.dump_obs2)
            if args.dump_stop:
                break
        if args.force_open and forced_step is None \
                and robot.cube_height() > 0.85 and robot.cube_box_dist_cm() < 12.0:
            forced_step = step
            print(f"[强制松爪] 步 {step}: 方块z={robot.cube_height():.3f}m "
                  f"距盒心={robot.cube_box_dist_cm():.1f}cm，夹爪改输出打开值 {q_open_ds:+.4f}",
                  flush=True)
        if forced_step is not None:
            chunk[:, ARM_DIM:] = q_open_ds
        for t in range(n):
            robot.send_action(chunk[t], step + t)
            grip_cmds.append(float(np.mean(chunk[t, ARM_DIM:])))
            tcp_hist.append((step + t, robot.tcp_cube_cm(), robot.cube_height(),
                             robot.cube_box_dist_cm()))
            placed_seen = placed_seen or robot.in_box()
            if (step + t) % 10 == 0:
                s = step + t
                cmd = float(np.mean(chunk[t, ARM_DIM:]))
                print(f"  步 {s:3d}  TCP→方块={robot.tcp_cube_cm():6.2f}cm  方块z={robot.cube_height():.3f}m"
                      f"  距盒心={robot.cube_box_dist_cm():5.1f}cm  夹爪={cmd:+.4f}", flush=True)
                if s >= 240:
                    tgt = robot.dataset_grip_to_sim(cmd)
                    q = robot.env.query_joint_qpos(robot.grip_joint_names)[robot.grip_joint_names[0]]
                    print(f"      [夹爪] 模型输出={cmd:+.4f}  映射目标={tgt:+.4f}  "
                          f"实际={float(np.asarray(q).ravel()[0]):+.4f}", flush=True)
        step += n

    arr = np.array(tcp_hist, dtype=np.float64)
    in_win = arr[arr[:, 0] >= args.start]
    if len(in_win) == 0:
        in_win = arr
    k = int(in_win[:, 1].argmin())
    kz = int(arr[:, 2].argmax())
    # 盒上方：任一步方块高于 0.85m 且距盒心 < 12cm
    hi = arr[arr[:, 2] > 0.85]
    above_box = bool(len(hi) and float(hi[:, 3].min()) < 12.0)
    lifted = bool(arr[:, 2].max() > 0.80)                  # 方块被抬离桌面 3cm 以上
    placed = bool(placed_seen)
    # 松爪指令：合爪之后再出现打开端（q01 侧 30% 以内）指令；初始化时的张开不算
    q01 = float(robot.ns["q01"][ARM_DIM])
    q99 = float(robot.ns["q99"][ARM_DIM])
    c0 = next((i for i, g in enumerate(grip_cmds) if g > q01 + 0.7 * (q99 - q01)), None)
    post = grip_cmds[c0:] if c0 is not None else grip_cmds
    gmin = min(post) if post else float("nan")
    opened = bool(gmin <= q01 + 0.3 * (q99 - q01))
    print(f"[消融] {args.group:24} 最近 TCP={in_win[k, 1]:.2f}cm @步{int(in_win[k, 0])}"
          f" | z最高={arr[kz, 2]:.3f}@{int(arr[kz, 0])} 结束z={arr[-1, 2]:.3f}"
          f" | 距盒心={robot.cube_box_dist_cm():.1f}cm"
          f" | 抓起={'Y' if lifted else 'N'} 盒上方={'Y' if above_box else 'N'}"
          f" 落盒={'Y' if placed else 'N'}"
          f" | 夹爪最小={gmin:+.4f} 松爪指令={'Y' if opened else 'N'}"
          f" | 强制松爪步={forced_step}", flush=True)
    robot.disconnect()


if __name__ == "__main__":
    main()
