"""在仿真里回放一集 LeRobot 示范轨迹（不接策略模型）。

目的：排除场景 / 夹爪 / 控制接口问题。若机器人能按示范轨迹接近、夹起方块，
则说明失败来自策略输出，而不是仿真环境或底层控制。

复用 infer_g1_pick.py 的 OrcaGymG1（同一套 PD + 重力补偿 + 夹爪映射），
把数据集每帧的 9 维关节目标依次下发，并实时打印 TCP→方块、方块高度等指标。

用法:
    python replay_lerobot_demo.py --episode 0
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from infer_g1_pick import ARM_DIM, OrcaGymG1, RemotePolicy, RolloutConfig


def _compare_at(robot, policy, states, i, args, grip_mid):
    """把当前观测单独喂给策略，与示范动作逐维比较。

    对齐关系：回放到帧 i 时机器人已处在示范第 i 帧姿态，因此策略输出的
    动作块应与 states[i : i+H] 对应（数据集中 action 与 state 完全相同）。
    """
    obs = robot.get_observation(i)
    if args.dump_obs:
        tcp = np.asarray(robot.env.query_site_pos_and_quat([robot.ee_sid])[robot.ee_sid]["xpos"]).ravel()[:3]
        payload = {
            "state": obs["state"], "step": i, "tcp": tcp,
            "cube": np.asarray(robot._md.xpos[robot.cube_bid], dtype=np.float64).copy(),
            "box": np.asarray(robot._md.xpos[robot.box_bid], dtype=np.float64).copy(),
            "demo": states[i],
        }
        for k, im in obs["images"].items():
            payload[f"img_{k}"] = im
        np.savez(args.dump_obs, **payload)
        print(f"[dump] 已保存示范帧 {i} 观测到 {args.dump_obs}", flush=True)
    h = args.compare_horizon
    demo = states[i : i + h]
    names = [n.replace("g1_pick_usda_right_", "").replace("_joint", "")
             for n in robot.right_joint_names] + ["grip"]
    align = float(np.abs(obs["state"][:ARM_DIM] - states[i][:ARM_DIM]).max())
    print("=" * 66, flush=True)
    print(f"[对比] 帧 {i}：当前状态 vs 示范对齐误差 {align:.4f} rad，"
          f"TCP→方块={robot.tcp_cube_cm():.2f}cm，可对比示范帧数={len(demo)}", flush=True)
    dc = next((t for t in range(len(demo)) if float(np.mean(demo[t, ARM_DIM:])) > grip_mid), None)
    print(f"[对比] 示范：合爪偏移 t={dc}，相对当前姿态的关节最大位移="
          f"{float(np.abs(demo - demo[0])[:, :ARM_DIM].max()):.4f} rad", flush=True)
    samples = []
    q01 = float(robot.ns["q01"][ARM_DIM])
    q99 = float(robot.ns["q99"][ARM_DIM])
    open_thr = q01 + 0.3 * (q99 - q01)
    dg = demo[:, ARM_DIM:].mean(axis=1) if len(demo) else np.array([np.nan])
    print(f"[对比] 示范夹爪：t0={dg[0]:+.4f} 末={dg[-1]:+.4f} min={np.nanmin(dg):+.4f}"
          f"（打开阈值 {open_thr:+.4f}）", flush=True)
    for s in range(args.compare_samples):
        chunk, lat = policy.select_action(obs)
        k = min(h, len(chunk), len(demo))
        d = np.abs(chunk[:k] - demo[:k])
        g = chunk[:k, ARM_DIM:].mean(axis=1)
        pc = next((t for t in range(k) if float(np.mean(chunk[t, ARM_DIM:])) > grip_mid), None)
        print(f"[对比] 采样{s + 1}：关节|Δ|均值={float(d[:, :ARM_DIM].mean()):.4f} "
              f"最大={float(d[:, :ARM_DIM].max()):.4f} rad，"
              f"夹爪|Δ|最大={float(d[:, ARM_DIM:].max()):.4f}，"
              f"预测合爪 t={pc}，推理 {lat * 1000:.0f}ms", flush=True)
        print(f"         夹爪预测：min={g.min():+.4f}@t{int(g.argmin())} 末t{k - 1}={g[-1]:+.4f}"
              f" 预测打开={'Y' if g.min() <= open_thr else 'N'}，"
              f"逐帧=" + " ".join(f"{v:+.3f}" for v in g), flush=True)
        print("         逐维最大|Δ|：" + " ".join(
            f"{nm}={float(d[:, j].max()):.3f}" for j, nm in enumerate(names)), flush=True)
        pts = sorted({0, k // 4, k // 2, 3 * k // 4, k - 1})
        print("         关节误差@t：" + " ".join(
            f"t{t}={float(d[t, :ARM_DIM].max()):.3f}" for t in pts if 0 <= t < k), flush=True)
        samples.append(chunk[:k])
    if args.compare_out and samples:
        np.savez(args.compare_out, frame=i, demo=demo, samples=np.stack(samples),
                 state=obs["state"])
        print(f"[对比] 已保存 {args.compare_out}（{len(samples)} 个采样）", flush=True)
    print("=" * 66, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--lerobot-root", default="/home/user/pi/lerobot_g1_pick")
    p.add_argument("--norm-root", default=None)
    p.add_argument("--orcagym-addr", default="localhost:50051")
    p.add_argument("--task-config", default="example.yaml")
    p.add_argument("--max-frames", type=int, default=10**6)
    # 单帧对比模式：回放到该帧后，把当前观测喂给策略，和示范动作比较
    p.add_argument("--compare-frame", type=int, default=-1)
    p.add_argument("--compare-horizon", type=int, default=10)
    p.add_argument("--compare-samples", type=int, default=3)
    p.add_argument("--host", default="192.168.110.51")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--trace", default=None, help="逐帧记录 q7/TCP/夹爪到 .npz（示范基准）")
    p.add_argument("--compare-out", default=None, help="保存该帧所有采样到 .npz")
    p.add_argument("--dump-obs", default=None, help="保存该帧观测（state/图像/TCP/方块/盒子）到 .npz")
    args = p.parse_args()

    cfg = RolloutConfig(
        lerobot_root=args.lerobot_root,
        norm_root=args.norm_root,
        episode=args.episode,
        orcagym_addr=args.orcagym_addr,
        task_config=args.task_config,
        dump_dir=None,
    )
    robot = OrcaGymG1(cfg)
    states = robot.states9
    n = min(len(states), args.max_frames)

    grip_mid = 0.5 * (float(robot.ns["q01"][ARM_DIM]) + float(robot.ns["q99"][ARM_DIM]))
    print("=" * 66, flush=True)
    print(f"  回放示范   {robot.parquet}", flush=True)
    print(f"  帧数       {n}（30 Hz，使能 '--max-frames' 可截断）", flush=True)
    print("=" * 66, flush=True)

    robot.settle()
    print(f"[起点] TCP→方块={robot.tcp_cube_cm():.2f}cm  方块z={robot.cube_height():.3f}m"
          f"  方块→盒={robot.cube_box_dist_cm():.2f}cm", flush=True)

    min_dist, min_step = 1e9, -1
    close_step, close_dist = None, None
    max_z, max_z_step = 0.0, -1
    err_peak, err_joint = 0.0, ""
    placed_step, hold = None, 0
    tr = {"q7": [], "tcp": [], "grip": []} if args.trace else None

    policy = None
    if args.compare_frame >= 0:
        robot.set_camera_anchor()   # 初始化相机帧计数/同步簿记
        policy = RemotePolicy(args.host, args.port)
        print(f"[对比] 回放到帧 {args.compare_frame} 后调一次策略，与示范动作比较", flush=True)

    for i in range(n):
        target = states[i]
        robot.send_action(target, i)

        cur = robot.read_state()
        err = np.abs(cur[:ARM_DIM] - target[:ARM_DIM])
        if err.max() > err_peak:
            err_peak, err_joint = float(err.max()), robot.right_joint_names[int(err.argmax())]

        dist = robot.tcp_cube_cm()
        if tr is not None:
            tr["q7"].append(np.asarray(cur[:ARM_DIM], dtype=np.float64))
            tr["tcp"].append(dist)
            tr["grip"].append(float(np.mean(target[ARM_DIM:])))
        if dist < min_dist:
            min_dist, min_step = dist, i
        z = robot.cube_height()
        if z > max_z:
            max_z, max_z_step = z, i
        if close_step is None and float(np.mean(target[ARM_DIM:])) > grip_mid:
            close_step, close_dist = i, dist

        if robot.in_box():
            hold += 1
            if hold >= cfg.success_hold and placed_step is None:
                placed_step = i
        else:
            hold = 0

        if i % 30 == 0:
            print(f"[回放] 帧 {i:4d}  TCP→方块={dist:6.2f}cm  方块z={z:.3f}m"
                  f"  方块→盒={robot.cube_box_dist_cm():6.2f}cm"
                  f"  夹爪={float(np.mean(target[ARM_DIM:])):+.4f}"
                  f"  跟踪误差={err.max():.4f}", flush=True)

        if policy is not None and i == args.compare_frame:
            _compare_at(robot, policy, states, i, args, grip_mid)
            break

    print("=" * 66, flush=True)
    if tr is not None and tr["q7"]:
        np.savez(args.trace, **{k: np.asarray(v) for k, v in tr.items()})
        print(f"[trace] 已保存 {args.trace}: {len(tr['q7'])} 帧", flush=True)
    print(f"[统计] 最近 TCP→方块 = {min_dist:.2f}cm（帧 {min_step}）", flush=True)
    print(f"[统计] 示范合爪帧 = {close_step}，合爪时 TCP→方块 = "
          f"{'—' if close_dist is None else f'{close_dist:.2f}cm'}", flush=True)
    print(f"[统计] 方块最高 z = {max_z:.3f}m（帧 {max_z_step}，桌面 0.770m）", flush=True)
    print(f"[统计] 跟踪误差峰值 = {err_peak:.4f} rad（{err_joint}）", flush=True)
    print(f"[统计] 落盒判定 = {'成功于帧 ' + str(placed_step) if placed_step is not None else '未触发'}"
          f"，最终方块→盒={robot.cube_box_dist_cm():.2f}cm", flush=True)
    lifted = max_z > 0.80
    print(f"[结论] 示范回放{'成功夹起并搬运方块' if lifted else '未能夹起方块'}"
          f"（方块最大抬升 {max_z - 0.770:.3f}m）", flush=True)

    robot.disconnect()


if __name__ == "__main__":
    main()
