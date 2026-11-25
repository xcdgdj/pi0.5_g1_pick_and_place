"""模型闭环 → 偏离方块时暂停 → Pico 接管完成完整抓放 → 录制整段数据。

对应任务流程：
    1. 模型 rollout 从数据集 ep0 初始位姿开始，录制同步开始（HDF5 + 两路视频）
    2. 每步监测 TCP→方块距离；模型贴近后一旦回退（默认 >2.5cm），
       或按右手柄 Grip，或已把方块搬到盒上方 → 立即暂停模型，PD 保持姿态
    3. 倒计时 3s（此时握起右手柄）→ 右臂 OSC 重锚当前末端，交给人
    4. 人用 Pico 完成：张爪(A) → 接近 → 合爪(扳机) → 抬起 → 搬到盒上方
       → 松爪(A) → 撤手
    5. 左 Grip 保存，右 Grip 丢弃；保存的是从第 0 步到结束的完整轨迹，
       同时在 record/dagger.json 里记下接管行号，转换时用 --trim-takeover
       裁掉模型段、只用人的专家段训练

按键：
    右手柄位姿   控制右臂
    右扳机       闭合（深度）      A 全开（松爪用）      B 中位
    左手柄 Grip  保存（仅接管后）
    右手柄 Grip  模型阶段=立即接管；接管后=丢弃

用法：
    python dagger_release_collect.py --level pi --agent_name g1_pick_usda
    python dagger_release_collect.py --dry-run    # 无 Pico 自检（存到 pi_dryrun）

前置：OrcaStudio 打开 pi level 并 Play；策略服务在线；Pico 已连接。
"""

import argparse
import faulthandler
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import mujoco

from conf import g1_dex1_osc_conf
from infer_g1_pick import ARM_DIM, OrcaGymG1, RemotePolicy, RolloutConfig

GROUPS = {
    "none": [],
    "sh_pitch": [0],
    "sh_yaw": [2],
    "sh_yaw+sh_pitch": [0, 2],
}

TASK_DESCRIPTION = "Pick up the red cube and place it in the box."


def _assert_joystick_port_free(port: int = 8001) -> None:
    """8001 被上次未退出的采集进程占用时直接退出。

    PicoJoystick 在后台线程里 bind，失败只会让该线程退出，主流程会照常
    跑完但没有任何手柄输入（表现为手柄完全没反应）。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        print(f"[错误] 端口 {port} 已被占用（上次采集进程可能未退出）。"
              f"先执行 pkill -f dagger_release_collect.py 再重跑", flush=True)
        raise SystemExit(2)
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description="模型悬停/偏离 → Pico 接管完整抓放 → 录制")
    p.add_argument("--level", default="pi")
    p.add_argument("--agent-name", default="g1_pick_usda")
    p.add_argument("--episode", type=int, default=0, help="初始姿态与替换引导所用数据集集号")
    p.add_argument("--exec-steps", type=int, default=10, help="每块执行步数（剩余丢弃重新推理）")
    p.add_argument("--override-group", default="sh_yaw+sh_pitch", choices=sorted(GROUPS),
                   help="接近段替换引导；纯模型抓不起方块时用它")
    p.add_argument("--override-start", type=int, default=35)
    p.add_argument("--override-end", type=int, default=150)
    p.add_argument("--model-timeout", type=int, default=480, help="模型阶段最大控制步（30Hz）")
    p.add_argument("--deviate-cm", type=float, default=2.5,
                   help="相对最近距离回退多少厘米判定为偏离")
    p.add_argument("--handover-delay", type=float, default=3.0)
    p.add_argument("--takeover-timeout", type=float, default=180.0)
    p.add_argument("--host", default="192.168.110.51")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--lerobot-root", default="/home/user/pi/lerobot_g1_pick")
    p.add_argument("--orcagym-addr", default="localhost:50051")
    p.add_argument("--task-config", default="example.yaml")
    p.add_argument("--cam-sync-ms", type=int, default=None)
    p.add_argument("--init-state", default=None,
                   help="从保存的仿真状态开始（npz 含 qpos/qvel/q9），跳过模型阶段直接接管")
    p.add_argument("--init-hold", type=int, default=10, help="恢复状态后保持若干步以稳定/渲染")
    p.add_argument("--dry-run", action="store_true", help="无 Pico：脚本开爪验证录制管线")
    p.add_argument("--dryrun-seconds", type=float, default=12.0)
    args = p.parse_args()
    if not args.dry_run:
        _assert_joystick_port_free(8001)

    kw = {}
    if args.cam_sync_ms:
        kw["cam_sync_ms"] = args.cam_sync_ms
    cfg = RolloutConfig(lerobot_root=args.lerobot_root, episode=args.episode,
                        orcagym_addr=args.orcagym_addr, task_config=args.task_config,
                        dump_dir=None, **kw)
    robot = OrcaGymG1(cfg)
    helpers = robot.helpers
    env, manager = robot.env, robot.manager

    # ── 录制存储（与遥操同格式）─────────────────────────────────────────────
    out_level = "pi_dryrun" if args.dry_run else args.level
    dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "dataset", args.agent_name, out_level)
    data_storage = helpers.G1Dex1DataStorage(
        dataset_path=dataset_path, hdf5_path="record/proprio_stats.hdf5")
    data_storage.set_video_path("video")
    manager.set_data_storage(data_storage)
    manager.save_video = True

    camera_keys = [c.strip() for c in cfg.cameras.split(",") if c.strip()]
    stream_cameras = helpers.ensure_camera_streams(env, camera_keys)
    data_storage.stream_cameras = stream_cameras
    helpers._enable_simulate_index(env)   # 帧头 sim_idx 需≥0，否则视频区间失效
    manager.set_task(helpers.EmptyTask(env))

    # ── Pico 与遥操控制器 ───────────────────────────────────────────────────
    state = {"phase": "model", "cmd": None, "handover_req": False}
    pico = r_arm = r_grip = None
    prev_grip = {"l": False, "r": False}
    if not args.dry_run:
        manager.scene_manager.get_scene_data(
            os.path.basename(os.path.realpath(__file__)), "beginscene")
        print("=" * 60, flush=True)
        print("  等待 Pico 连接（戴上头显、握起手柄）...", flush=True)
        print("=" * 60, flush=True)
        pico = helpers.PicoJoystickDevice(helpers.PicoJoystick())
        manager.device = pico
        K = helpers.PicoJoystickKey
        helpers.add_dex1_force_pico_controller(
            manager, env, g1_dex1_osc_conf.gripper_l, g1_dex1_osc_conf.base_body,
            pico, [K.X, K.Y, K.L_TRIGGER])
        r_grip = helpers.add_dex1_force_pico_controller(
            manager, env, g1_dex1_osc_conf.gripper_r, g1_dex1_osc_conf.base_body,
            pico, [K.A, K.B, K.R_TRIGGER])
        helpers.add_arm_osc_stable_pico_controller(
            manager, env, g1_dex1_osc_conf.l_arm, g1_dex1_osc_conf.base_body,
            pico, K.L_TRANSFORM)
        r_arm = helpers.add_arm_osc_stable_pico_controller(
            manager, env, g1_dex1_osc_conf.r_arm, g1_dex1_osc_conf.base_body,
            pico, K.R_TRANSFORM)

        def _edge(key, handler):
            def cb(pressed):
                if pressed and not prev_grip[key]:
                    handler()
                prev_grip[key] = bool(pressed)
            return cb

        def _on_right_grip():
            if state["phase"] == "model":
                state["handover_req"] = True
            else:
                state["cmd"] = "discard"

        def _on_left_grip():
            if state["phase"] == "takeover":
                state["cmd"] = "save"

        pico.bind_grip_button_event(K.R_GRIPBUTTON, _edge("r", _on_right_grip))
        pico.bind_grip_button_event(K.L_GRIPBUTTON, _edge("l", _on_left_grip))
        manager.set_init_ctrl()
        print("[按键] 右Grip=模型阶段立即接管 / 接管后丢弃", flush=True)
        print("       右扳机=合爪  A=全开松爪  左Grip=保存", flush=True)
        print("[提示] 模型阶段请松开右手柄，收到接管提示后再握起", flush=True)

    # ── 模型阶段 + 全程录制 ─────────────────────────────────────────────────
    policy = None if args.init_state else RemotePolicy(args.host, args.port)
    demo = robot.states9
    idx = GROUPS[args.override_group]
    print(f"[引导] 替换={args.override_group} {idx or '无'}，窗口 步 {args.override_start}~{args.override_end}",
          flush=True)

    grip_hold_sim = None
    robot.settle()
    if args.init_state:
        z = np.load(args.init_state)
        q9 = z["q9"]
        robot._md.qpos[:] = z["qpos"]
        robot._md.qvel[:] = z["qvel"]
        mujoco.mj_forward(robot._mj, robot._md)
        q01 = float(robot.ns["q01"][ARM_DIM])
        q99 = float(robot.ns["q99"][ARM_DIM])
        lo, hi = cfg.grip_range
        if "last_action" in z:
            # 用保存的模型目标作为保持指令：夹爪目标需压向方块产生夹持力，
            # 若直接回填实际 qpos 则手指只停在原位、不施力，方块会滑落
            act = z["last_action"]
            a_hold = act[:ARM_DIM].copy()
            g_hold = [float(act[ARM_DIM]), float(act[ARM_DIM + 1])]
        else:
            a_hold = q9[:ARM_DIM].copy()
            g_hold = [q01 + (float(q9[ARM_DIM]) - lo) / (hi - lo) * (q99 - q01)] * 2
        for i in range(args.init_hold):
            robot.send_action(np.r_[a_hold, g_hold], i)
        # 交接时沿用这个夹持指令，勿用实测开度（零误差=零夹持力，方块会滑落）
        grip_hold_sim = float(robot.dataset_grip_to_sim(float(np.mean(g_hold))))
        s = robot.read_state()
        print(f"[初始状态] {args.init_state}: 恢复差 max={np.abs(s - q9).max():.4f}"
              f"  方块z={robot.cube_height():.3f}m  距盒心={robot.cube_box_dist_cm():.1f}cm"
              f"  TCP→方块={robot.tcp_cube_cm():.2f}cm", flush=True)
    robot.set_camera_anchor()

    record_start = datetime.now()
    initial_qpos = manager._capture_initial_joint_qpos()   # 第 0 步姿态
    data_storage.begin_save_video(env)

    t_next = time.time()

    def pace(period=1.0 / 30.0):
        nonlocal t_next
        t_next += period
        d = t_next - time.time()
        if d > 0:
            time.sleep(d)
        else:
            t_next = time.time()

    def rec(a9, s):
        robot.send_action(a9, s)          # 内部已按控制步推进并 render
        data_storage.collection_data(data_storage.obs_callback(env), env)

    step, d_min, d_min_step = 0, 1e9, -1
    trigger = (f"初始状态({os.path.basename(args.init_state)})"
               if args.init_state else None)
    last9 = robot.read_state()
    if trigger:
        print(f"[初始状态] 跳过模型阶段：{trigger}", flush=True)

    def check(s):
        nonlocal d_min, d_min_step
        if pico is not None:
            pico.update()                 # 派发按键事件（右Grip 立即接管）
        if state["handover_req"]:
            return "手动(右Grip)"
        d = robot.tcp_cube_cm()
        if d < d_min:
            d_min, d_min_step = d, s
        if s - d_min_step >= 3 and d_min <= 9.0 and d - d_min >= args.deviate_cm:
            return f"偏离(最近{d_min:.1f}cm@步{d_min_step} → 当前{d:.1f}cm)"
        if 0.80 < robot.cube_height() < 0.93 and robot.cube_box_dist_cm() < 10.0:
            return "盒上方悬停"
        return None

    chunk = None
    while step < args.model_timeout and trigger is None:
        if chunk is None:
            obs = robot.get_observation(step)
            res = {}
            th = threading.Thread(
                target=lambda: res.update(chunk=policy.select_action(obs)[0]))
            th.start()
            # 推理期间保持上一动作并按 30Hz 继续录制，避免视频断帧
            while th.is_alive():
                rec(last9, step)
                step += 1
                trigger = check(step)
                if trigger:
                    break
                pace()
            if trigger:
                break
            if "chunk" not in res:
                print("[错误] 策略推理失败（服务端异常/断连），本集丢弃", flush=True)
                data_storage.clear_data()
                data_storage._episode_start_idx = {}
                robot.disconnect()
                return 1
            chunk = res["chunk"]
        n = min(args.exec_steps, len(chunk), args.model_timeout - step)
        for t in range(n):
            a9 = chunk[t].copy()
            if idx and args.override_start <= step < args.override_end and step < len(demo):
                a9[idx] = demo[step][idx]
            rec(a9, step)
            last9 = a9
            step += 1
            if step % 10 == 0:
                print(f"  步 {step:3d}  TCP→方块={robot.tcp_cube_cm():6.2f}cm"
                      f"  方块z={robot.cube_height():.3f}m"
                      f"  距盒心={robot.cube_box_dist_cm():5.1f}cm", flush=True)
            trigger = check(step)
            if trigger:
                break
            pace()
        chunk = None

    if trigger is None:
        print(f"[结果] 模型 {args.model_timeout} 步内未触发接管"
              f"（方块z={robot.cube_height():.3f}m 距盒心={robot.cube_box_dist_cm():.1f}cm），本集丢弃",
              flush=True)
        data_storage.clear_data()
        data_storage._episode_start_idx = {}
        robot.disconnect()
        return 1

    print("=" * 60, flush=True)
    print(f"  触发接管 @步{step}: {trigger}", flush=True)
    print(f"  方块z={robot.cube_height():.3f}m 距盒心={robot.cube_box_dist_cm():.1f}cm"
          f" TCP→方块={robot.tcp_cube_cm():.2f}cm", flush=True)
    print("=" * 60, flush=True)

    # ── 倒计时（模型 PD 保持，仍录制）──────────────────────────────────────
    if not args.dry_run:
        manager.scene_manager.show_ui_message(
            1, f"Takeover in {args.handover_delay:.0f}s, grip RIGHT controller", "0xffff00", 5)
        print(f"  {args.handover_delay:.0f} 秒后接管，请现在握起右手柄", flush=True)
    for _ in range(int(args.handover_delay * 30)):
        if pico is not None:
            pico.update()
        rec(last9, step)
        step += 1
        pace()

    # 接管后第一行在整段轨迹中的行号：转换数据时可据此裁掉模型段，
    # 只用人的专家段做训练（模型段仍保留在原始文件里便于复盘）
    takeover_row = None if args.dry_run else len(data_storage.data["time_step"])

    # ── 接管 ────────────────────────────────────────────────────────────────
    if not args.dry_run:
        q_now = robot.read_state()
        r_arm.reset()                        # 重锚当前末端，手柄不跳变
        r_grip.q_target = (grip_hold_sim if grip_hold_sim is not None
                           else float(np.mean(q_now[ARM_DIM:])))   # 保持当前开度，A 键全开
        state["phase"] = "takeover"
        manager.scene_manager.show_ui_message(
            1, "TAKEOVER: trigger=close A=open LEFT-grip=save", "0x00ff00", 5)
        print("[接管] 已生效：先按 A 张爪→接近→扳机合爪→抬起→盒上方→A 松爪→撤手→左Grip保存",
              flush=True)

        t_begin, n_rec, last_print = time.time(), 0, time.time()
        while state["cmd"] is None:
            t0 = time.time()
            action = manager.run_controllers()      # 内部会 update 设备并派发按键
            term = trunc = False
            for _ in range(5):                      # 5×1ms = 5ms，对齐遥操 200Hz
                _, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            env.render()
            data_storage.collection_data(data_storage.obs_callback(env), env)
            n_rec += 1
            if term or trunc:
                print("[接管] 仿真提前终止，丢弃本集", flush=True)
                state["cmd"] = "discard"
                break
            if time.time() - last_print > 0.5:
                last_print = time.time()
                print(f"  [录] {time.time() - t_begin:5.1f}s  方块z={robot.cube_height():.3f}m"
                      f"  距盒心={robot.cube_box_dist_cm():5.1f}cm"
                      f"  落盒={'Y' if robot.in_box() else 'N'}"
                      f"  ({n_rec} 行)", flush=True)
            if time.time() - t_begin > args.takeover_timeout:
                print("[接管] 超时，自动保存", flush=True)
                state["cmd"] = "save"
            d = 0.005 - (time.time() - t0)
            if d > 0:
                time.sleep(d)
        cmd = state["cmd"]
    else:
        # dry-run：保持 1s → 全开 → 再录到超时，仅验证 HDF5/视频落盘
        print("[DRY-RUN] 1s 后脚本全开夹爪（验证录制管线，数据存 pi_dryrun）", flush=True)
        t_begin = time.time()
        while time.time() - t_begin < args.dryrun_seconds:
            if time.time() - t_begin > 1.0:
                grip_ds = float(robot.ns["q01"][ARM_DIM])
            else:
                grip_ds = float(np.mean(last9[ARM_DIM:]))
            a9 = np.concatenate([robot.read_state()[:ARM_DIM], [grip_ds, grip_ds]])
            rec(a9, step)
            step += 1
            if step % 30 == 0:
                print(f"  [dry] {time.time() - t_begin:5.1f}s  方块z={robot.cube_height():.3f}m"
                      f"  距盒心={robot.cube_box_dist_cm():5.1f}cm"
                      f"  落盒={'Y' if robot.in_box() else 'N'}", flush=True)
            pace()
        cmd = "save"

    # ── 保存 / 丢弃 ─────────────────────────────────────────────────────────
    if cmd == "save":
        # save_data 内部会切到下一个 uuid 目录，先记下本集路径
        unit_dir = data_storage.get_current_unit_path()
        saved_h5 = data_storage.get_hdf5_absolute_path()
        data_storage.stop_save_video(env)
        record_end = datetime.now()
        meta = manager._collect_sim_metadata()
        meta["frame_skip"] = 5          # 接管段 5×1ms/行，与遥操数据口径一致
        data_storage.save_data(
            task_info=manager.task.get_task_info(),
            scene_info=manager.scene_manager.get_scene_info(),
            task_description=TASK_DESCRIPTION,
            record_start_time=record_start.isoformat(),
            record_end_time=record_end.isoformat(),
            initial_joint_qpos=initial_qpos,
            **meta,
        )
        # 接管点标记：转换器据此裁剪出「只用专家段」的数据集
        if takeover_row is not None:
            os.makedirs(os.path.join(unit_dir, "record"), exist_ok=True)
            with open(os.path.join(unit_dir, "record", "dagger.json"), "w") as fp:
                json.dump(
                    {
                        "takeover_row": int(takeover_row),
                        "trigger_step": int(step),
                        "trigger": trigger,
                    },
                    fp,
                    ensure_ascii=False,
                )
            print(f"[结果] 接管点 行{takeover_row}（转换时 --trim-takeover 可裁掉模型段）",
                  flush=True)
        print(f"[结果] 已保存 {saved_h5}", flush=True)
        print(f"[结果] 结束时 方块z={robot.cube_height():.3f}m"
              f" 距盒心={robot.cube_box_dist_cm():.1f}cm"
              f" 落盒={'Y' if robot.in_box() else 'N'}", flush=True)
    else:
        data_storage.clear_data()
        data_storage._episode_start_idx = {}
        print("[结果] 本集已丢弃", flush=True)

    # 清理阶段若卡在 gRPC，20s 后打印各线程栈并强制退出，避免残留进程占着 8001
    faulthandler.dump_traceback_later(20, exit=True)
    robot.disconnect()
    try:
        env.reset()
        time.sleep(0.1)
        env.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl-C 若只打断主线程，解释器仍会等待非 daemon 线程（Pico 8001 服务等），
        # 终端表现为"终止不了"；这里直接强退，释放端口
        print("\n[中断] Ctrl-C：强制退出", flush=True)
        os._exit(130)
