"""回放 HDF5 遥操作数据：把录制时的电机指令按固定帧率直接灌给 OrcaGym。

不接任何策略模型，也不需要 Pico。默认用位置跟踪（PD）回放：
    手臂  /action/joint/position (N,14)  关节目标角 → PD 力矩 → *_mctrl
    夹爪  /action/effector/motor (N,2)   夹持力指令 → *_dex1_motor
因为 *_mctrl 是 <motor>（ctrl 即力矩），不能把角度直接写进 ctrl，
所以用 tau = kp*(q_des - q) - kd*dq 做关节 PD 跟踪。
增益默认按模型等效惯量逐关节计算（kd = 2ζ√(kp·J)，并限制 kd·dt/J < 1）：
固定的大 kp/kd 在小惯量腕关节上会超过离散稳定界限，导致持续抖动、
稳态偏差几厘米，表现为"手臂还没到方块夹爪就闭合"。--manual-gains 可改回 --kp/--kd。
--mode torque 可切回开环力矩回放（复现原始录制的力矩序列）。
基座/腰/腿用 equality 约束钉住（与采集时一致），否则机器人会瘫倒。
起始姿态优先用 HDF5 里的 initial_joint_qpos 恢复，保证第一帧对齐。

用法:
    python replay_hdf5.py --level pi --agent_name g1_pick_usda --episode 1
    python replay_hdf5.py --level pi --agent_name g1_pick_usda          # 默认回放最新一集
    python replay_hdf5.py --level pi --agent_name g1_pick_usda --fps 60 --loop
    python replay_hdf5.py --level pi --agent_name g1_pick_usda --mode torque   # 开环力矩
"""

import argparse
import importlib.util
import json
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import h5py
import numpy as np
from yaml import load, Loader

from scene.scene_manager import SceneManager
from conf import g1_dex1_osc_conf
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1Replay",
    log_file="g1_replay.log",
    log_dir=os.path.join(base_dir, "logs"),
    console_level="INFO",
    file_level="INFO",
    force_reinit=True,
)


def _load_collection_helpers():
    """复用采集脚本里的姿态钉住/求解器调参函数（避免复制 200 行）。"""
    path = os.path.join(base_dir, "g1_dex1_collection_tele.py")
    spec = importlib.util.spec_from_file_location("g1_collect_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay_obs_callback(env) -> dict:
    """观测回调只用于搭出非空 observation_space，回放本身不读观测。"""
    names = [
        env.joint(name)
        for name in (
            g1_dex1_osc_conf.l_arm["joint_names"]
            + g1_dex1_osc_conf.r_arm["joint_names"]
            + g1_dex1_osc_conf.gripper_l["joint_names"]
            + g1_dex1_osc_conf.gripper_r["joint_names"]
        )
    ]
    qpos = env.query_joint_qpos(names)
    return {
        "/observation/joint/position": np.array(
            [qpos[name] for name in names], dtype=np.float32
        ).flatten()
    }


def find_episode(dataset_root: str, episode: str | None) -> str:
    if episode:
        path = episode
        if os.path.isfile(path):
            return path
        candidate = os.path.join(dataset_root, episode, "record", "proprio_stats.hdf5")
        if os.path.isfile(candidate):
            return candidate
        raise SystemExit(f"找不到 episode: {episode}（已尝试 {candidate}）")
    files = [
        os.path.join(dataset_root, name, "record", "proprio_stats.hdf5")
        for name in os.listdir(dataset_root)
        if os.path.isfile(
            os.path.join(dataset_root, name, "record", "proprio_stats.hdf5")
        )
    ] if os.path.isdir(dataset_root) else []
    if not files:
        raise SystemExit(f"目录下没有可回放的 episode: {dataset_root}")
    return max(files, key=os.path.getmtime)


def restore_pose(env, initial_qpos: dict | None):
    if not initial_qpos:
        return
    env.set_joint_qpos(
        {name: np.asarray(values, dtype=np.float64) for name, values in initial_qpos.items()}
    )
    env.mj_forward()


def main():
    parser = argparse.ArgumentParser(description="HDF5 遥操作数据回放（OrcaGym 直灌电机指令）")
    parser.add_argument("--level", default="pi", help="OrcaStudio 场景名称")
    parser.add_argument("--agent_name", default="g1_pick_usda", help="场景中的 agent 名称")
    parser.add_argument("--episode", default=None, help="episode 目录名 / HDF5 路径，缺省为最新一集")
    parser.add_argument("--fps", type=float, default=30.0, help="回放帧率（录制时约 28fps）")
    parser.add_argument(
        "--mode",
        choices=["pd", "torque"],
        default="pd",
        help="pd=追踪录制的位置序列（默认）；torque=开环复现录制力矩",
    )
    parser.add_argument("--kp", type=float, default=300.0, help="PD 比例增益")
    parser.add_argument("--kd", type=float, default=20.0, help="PD 微分增益")
    parser.add_argument("--max-tau", type=float, default=25.0, help="大臂关节力矩上限 N·m")
    parser.add_argument("--max-tau-wrist", type=float, default=5.0, help="腕关节力矩上限 N·m")
    parser.add_argument("--settle-frames", type=int, default=15, help="起手前 PD 稳住的帧数")
    parser.add_argument("--blend-frames", type=int, default=10, help="从初始姿态过渡到目标的帧数")
    parser.add_argument("--debug-pd", action="store_true", help="打印 PD 力矩/速度/饱和情况")
    parser.add_argument(
        "--no-ff",
        dest="feedforward",
        action="store_false",
        help="关闭力矩前馈（默认 = 录制力矩 + PD 修正）",
    )
    parser.add_argument("--loop", action="store_true", help="循环回放直到 Ctrl+C")
    parser.add_argument(
        "--hold-frame",
        type=int,
        default=-1,
        help="诊断：摆到该帧录制角度后静态 PD 保持 120 帧并打印误差，不跑完整回放",
    )
    parser.add_argument(
        "--auto-gains",
        dest="auto_gains",
        action="store_true",
        default=True,
        help="按模型等效惯量逐关节计算 kp/kd（默认开启）",
    )
    parser.add_argument(
        "--manual-gains",
        dest="auto_gains",
        action="store_false",
        help="关闭自动增益，改用 --kp/--kd",
    )
    parser.add_argument("--wn", type=float, default=60.0, help="--auto-gains 的目标自然频率 rad/s")
    parser.add_argument("--zeta", type=float, default=1.0, help="--auto-gains 的阻尼比")
    parser.add_argument("--fast", action="store_true", help="不按 fps sleep，尽快跑完（诊断用）")
    parser.add_argument("--task_config", default="example.yaml", help="SceneManager 配置文件")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    dataset_root = os.path.join(base_dir, "dataset", args.agent_name, args.level)
    h5_path = find_episode(dataset_root, args.episode)

    with h5py.File(h5_path, "r") as f:
        motors = f["action/joint/motor"][:]
        gripper_motors = f["action/effector/motor"][:]
        frame_skip = int(f["frame_skip"][()]) if "frame_skip" in f else 5
        # HDF5 里的 dt 是控制周期（frame_skip * time_step = 0.005），
        # 物理步长要再除回去（0.001），否则每帧推进 25ms 仿真时间，动作会慢 5 倍。
        control_dt = float(f["dt"][()]) if "dt" in f else 0.005
        # 手臂关节实测位置序列（rad），顺序 = 左臂 7 + 右臂 7，与 motors 一致
        qpos_targets = f["action/joint/position"][:]
        initial_qpos = (
            json.loads(f["initial_joint_qpos"][()].decode())
            if "initial_joint_qpos" in f
            else None
        )
        print("HDF5 keys:", flush=True)
        for key in sorted(f.keys()):
            obj = f[key]
            shape = getattr(obj, "shape", None)
            if shape is not None and shape != ():
                print(f"  {key:32s} {str(shape):14s} {obj.dtype}", flush=True)
            else:
                print(f"  {key:32s} (标量/对象)", flush=True)
    physics_dt = control_dt / max(frame_skip, 1)
    n_frames = motors.shape[0]
    if motors.shape[1] != 14 or gripper_motors.shape[1] != 2:
        raise SystemExit(f"字段维度不符: motor={motors.shape}, gripper={gripper_motors.shape}")
    if qpos_targets.shape != motors.shape:
        raise SystemExit(f"位置序列维度不符: {qpos_targets.shape} != {motors.shape}")

    print("=" * 60, flush=True)
    print(f"  回放文件  {h5_path}", flush=True)
    print(f"  帧数      {n_frames}  ({n_frames / args.fps:.1f}s @ {args.fps:g}fps)", flush=True)
    print(f"  控制周期  frame_skip={frame_skip}, 物理步长={physics_dt}", flush=True)
    if args.mode == "pd":
        gain_txt = (
            f"自动增益 wn={args.wn:g} zeta={args.zeta:g}"
            if args.auto_gains
            else f"手动增益 kp={args.kp:g} kd={args.kd:g}"
        )
        print(f"  回放方式  PD 跟踪位置  {gain_txt}", flush=True)
    else:
        print("  回放方式  开环力矩（录制值直灌）", flush=True)
    print("=" * 60, flush=True)

    helpers = _load_collection_helpers()

    default_joint_values = {}
    for joint_name in (
        g1_dex1_osc_conf.l_arm["joint_names"] + g1_dex1_osc_conf.r_arm["joint_names"]
    ):
        default_joint_values[joint_name] = 0.0

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    if "data_collection" in config:
        config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="Replay",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=_replay_obs_callback,
        env_index=0,
        # 一次 step 只推进一个物理步（1ms），PD 在 1kHz 上闭环；
        # 录制的每个控制帧（5ms）在循环里拆成 substeps 次子步执行。
        frame_skip=1,
        time_step=physics_dt,
        orcagym_addr=args.orcagym_addr,
        scene_manager=scene_manager,
    )
    env = manager.env

    arm_actuators = (
        g1_dex1_osc_conf.l_arm["motors_names"] + g1_dex1_osc_conf.r_arm["motors_names"]
    )
    gripper_actuators = (
        g1_dex1_osc_conf.gripper_l["actuator_names"]
        + g1_dex1_osc_conf.gripper_r["actuator_names"]
    )
    arm_ids = [env.model.actuator_name2id(env.actuator(name)) for name in arm_actuators]
    gripper_ids = [
        env.model.actuator_name2id(env.actuator(name)) for name in gripper_actuators
    ]
    arm_joint_names = [
        env.joint(name)
        for name in (
            g1_dex1_osc_conf.l_arm["joint_names"]
            + g1_dex1_osc_conf.r_arm["joint_names"]
        )
    ]
    # 每个录制帧对应的物理子步数（frame_skip=5 → 5 × 1ms = 5ms）
    substeps = max(int(frame_skip), 1)
    # 与 XML forcerange 一致：大臂/肩/肘 ±25，腕 pitch/yaw ±5
    tau_limit = np.array(
        [args.max_tau] * 5 + [args.max_tau_wrist] * 2 + [args.max_tau] * 5
        + [args.max_tau_wrist] * 2,
        dtype=np.float64,
    )

    def prepare_scene():
        env.reset()
        time.sleep(0.1)
        manager.update_scene()
        restore_pose(env, initial_qpos)
        # 先摆好姿态再钉住，腰/腿的参考值才与录制时一致
        helpers.apply_grasp_solver_tuning(env)
        helpers.pin_all_joints(env, args.agent_name)

    prepare_scene()

    ctrl = np.zeros(env.nu, dtype=np.float32)
    period = 1.0 / args.fps

    def read_arm_state():
        qpos_now = env.query_joint_qpos(arm_joint_names)
        qvel_now = env.query_joint_qvel(arm_joint_names)
        q = np.array(
            [np.asarray(qpos_now[name]).ravel()[0] for name in arm_joint_names],
            dtype=np.float64,
        )
        dq = np.array(
            [np.asarray(qvel_now[name]).ravel()[0] for name in arm_joint_names],
            dtype=np.float64,
        )
        return q, dq

    # 每个关节独立的 PD 增益（--auto-gains 时按等效惯量计算，见下方）
    kp_vec = np.full(14, args.kp, dtype=np.float64)
    kd_vec = np.full(14, args.kd, dtype=np.float64)

    def joint_pd(target, ff=None):
        q, dq = read_arm_state()
        tau = kp_vec * (target - q) - kd_vec * dq
        if args.feedforward and ff is not None:
            # 录制力矩作为前馈：它正是产生该轨迹的真实输入，
            # PD 只负责补偿残差，避免纯 PD 在大加速度段饱和滞后
            tau = tau + ff
        return np.clip(tau, -tau_limit, tau_limit), q, dq

    def arm_target_from_initial():
        if not initial_qpos:
            return None
        values = []
        for name in arm_joint_names:
            if name not in initial_qpos:
                return None
            values.append(float(np.asarray(initial_qpos[name]).ravel()[0]))
        return np.array(values, dtype=np.float64)

    def report_pose(expected: np.ndarray, label: str):
        q, _ = read_arm_state()
        diff = q - expected
        for idx in range(min(3, len(arm_joint_names))):
            print(
                f"    [{label}] {arm_joint_names[idx].split('_', 1)[-1]:28s}"
                f" 期望={expected[idx]:+.4f}  实际={q[idx]:+.4f}  差={diff[idx]:+.4f}",
                flush=True,
            )
        worst = int(np.argmax(np.abs(diff)))
        print(
            f"    [{label}] 14 关节最大偏差 {np.abs(diff).max():.4f} rad @ "
            f"{arm_joint_names[worst].split('_', 1)[-1]}",
            flush=True,
        )

    # ── 末端-方块距离监测 ────────────────────────────────────────────────
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    mj, md = gym._mjModel, gym._mjData
    cube_bid = next(
        (
            i for i in range(mj.nbody)
            if "cube" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
        ),
        -1,
    )
    cube_name = (
        mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, cube_bid)
        if cube_bid >= 0
        else "未找到"
    )
    r_ee_site = env.site(g1_dex1_osc_conf.r_arm["ee_site_name"])
    print(f"\n[距离监测] 方块 body={cube_name}, 右臂 TCP site={r_ee_site}", flush=True)

    def tcp_cube_distance() -> float:
        if cube_bid < 0:
            return float("nan")
        tcp = np.asarray(
            env.query_site_pos_and_quat([r_ee_site])[r_ee_site]["xpos"], dtype=np.float64
        ).ravel()[:3]
        cube = np.asarray(md.xpos[cube_bid], dtype=np.float64)
        return float(np.linalg.norm(tcp - cube))

    # ── 基座姿态 & FK：区分"数据本身没到位" vs "回放跟踪没到位" ──────────
    site_id = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, r_ee_site))
    fb_key = next(
        (k for k in (initial_qpos or {}) if k.endswith("floating_base_joint")), None
    )
    base_recorded = (
        np.asarray(initial_qpos[fb_key], dtype=np.float64) if fb_key else None
    )
    base_adr = -1
    if fb_key is not None:
        fb_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, fb_key))
        if fb_jid >= 0:
            base_adr = int(mj.jnt_qposadr[fb_jid])
    arm_qpos_adr = [
        int(
            mj.jnt_qposadr[
                mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            ]
        )
        for joint_name in arm_joint_names
    ]
    scratch = mujoco.MjData(mj)

    def fk_tcp(arm_q: np.ndarray, base_q: np.ndarray | None = None) -> np.ndarray:
        """临时把手臂关节（可选基座）替换成给定值做正运动学，返回 TCP 世界坐标。"""
        scratch.qpos[:] = md.qpos
        if base_q is not None and base_adr >= 0:
            scratch.qpos[base_adr : base_adr + 7] = base_q
        for adr, val in zip(arm_qpos_adr, arm_q):
            scratch.qpos[adr] = val
        mujoco.mj_kinematics(mj, scratch)
        return np.asarray(scratch.site_xpos[site_id], dtype=np.float64).copy()

    def cube_pos() -> np.ndarray:
        return np.asarray(md.xpos[cube_bid], dtype=np.float64)

    def dist_to_cube(p: np.ndarray) -> float:
        return float(np.linalg.norm(p - cube_pos()))

    def right_related(geom_name: str) -> bool:
        key = geom_name.lower()
        if any(t in key for t in ("foot", "ankle", "hip", "knee", "waist")):
            return False
        return any(t in key for t in ("right", "dex1", "cube", "box", "table"))

    # ── 逐关节 PD 增益：按模型等效惯量 + 力矩上限设计 ─────────────────────
    if args.auto_gains:
        arm_dof_adr = [
            int(mj.jnt_dofadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in arm_joint_names
        ]
        scratch.qpos[:] = md.qpos
        if base_recorded is not None and base_adr >= 0:
            scratch.qpos[base_adr : base_adr + 7] = base_recorded
        for adr, val in zip(arm_qpos_adr, qpos_targets[0]):
            scratch.qpos[adr] = val
        mujoco.mj_forward(mj, scratch)
        # MuJoCo 4.x 的 d.M 是压缩三角形格式，逐列用 mj_mulM 取稠密惯量矩阵
        mass_mat = np.zeros((mj.nv, mj.nv), dtype=np.float64)
        for col in range(mj.nv):
            unit = np.zeros(mj.nv, dtype=np.float64)
            unit[col] = 1.0
            res = np.zeros(mj.nv, dtype=np.float64)
            mujoco.mj_mulM(mj, scratch, res, unit)
            mass_mat[:, col] = res
        joint_j = np.array([mass_mat[d, d] for d in arm_dof_adr])
        kp_vec = joint_j * args.wn**2
        kd_vec = 2.0 * args.zeta * joint_j * args.wn
        # 离散稳定上限：kd*dt/J 必须明显小于 2
        kd_cap = 1.0 * joint_j / physics_dt
        kd_vec = np.minimum(kd_vec, kd_cap)
        print(
            f"[自动增益] wn={args.wn:g}rad/s zeta={args.zeta:g} "
            f"（J 为等效转动惯量，含下游连杆与 armature）",
            flush=True,
        )
        for k in range(14):
            print(
                f"    {arm_joint_names[k].split('_', 1)[-1]:28s}"
                f" J={joint_j[k]:.5f}  kp={kp_vec[k]:8.2f}  kd={kd_vec[k]:6.3f}"
                f"  力矩上限={tau_limit[k]:.0f}  kd*dt/J={kd_vec[k] * physics_dt / joint_j[k]:.3f}",
                flush=True,
            )

    def base_pose_report(label: str):
        if base_recorded is None or base_adr < 0:
            print(f"[基座 {label}] HDF5 无 floating_base_joint，跳过", flush=True)
            return
        cur = np.asarray(md.qpos[base_adr : base_adr + 7], dtype=np.float64)
        dpos = np.linalg.norm(cur[:3] - base_recorded[:3]) * 100
        dot = float(np.clip(abs(np.dot(cur[3:], base_recorded[3:])), -1.0, 1.0))
        dang = 2.0 * np.degrees(np.arccos(dot))
        print(
            f"[基座 {label}] 录制 pos={np.round(base_recorded[:3], 4)} "
            f"当前 pos={np.round(cur[:3], 4)}  差={dpos:.2f}cm  姿态差={dang:.2f}°",
            flush=True,
        )

    # 夹爪指令开始闭合的帧（右侧，第 2 列）
    grip_r = np.asarray(gripper_motors[:, 1], dtype=np.float64)
    changes = np.where(np.abs(np.diff(grip_r)) > 0.5)[0] + 1
    holding = np.where(grip_r > 5.0)[0]
    print(
        "[夹爪指令] 右夹爪力 > 5N 的首帧: "
        + (str(int(holding[0])) if len(holding) else "无")
        + f"; 力突变帧(前 8 个): {changes[:8].tolist()}",
        flush=True,
    )
    if len(holding):
        idx0 = int(holding[0])
        # 闭合前的"预夹紧"：最后一段低力平台（>1.2N 但不夹紧）
        pre = np.where((grip_r[:idx0] > 1.2) & (grip_r[:idx0] < 5.0))[0]
        if len(pre):
            print(
                f"            预夹紧(>{1.2:.1f}N)首帧 {int(pre[0])} "
                f"力={grip_r[pre[0]]:.1f}N；正式闭合帧 {idx0} 力={grip_r[idx0]:.1f}N",
                flush=True,
            )
        print(
            f"            第 {idx0} 帧力={grip_r[idx0]:.1f}N, "
            f"此后 60 帧内最大 {grip_r[idx0:idx0 + 60].max():.1f}N",
            flush=True,
        )
    close_frames = (
        {max(int(holding[0]) - 30, 0), max(int(holding[0]) - 10, 0), int(holding[0])}
        if len(holding)
        else set()
    )

    # ── 数据自身：按录制基座+录制关节做 FK，夹爪闭合时 TCP 离方块多远 ────
    if len(holding) and base_recorded is not None:
        idx0 = int(holding[0])
        lo, hi = max(idx0 - 60, 0), min(idx0 + 60, n_frames)
        d_data = np.array(
            [dist_to_cube(fk_tcp(qpos_targets[k], base_recorded)) for k in range(lo, hi)]
        )
        best = lo + int(np.argmin(d_data))
        print(
            f"[数据自身] 闭合帧 {idx0}: 录制轨迹 TCP→方块={d_data[idx0 - lo] * 100:.2f}cm；"
            f"±60 帧内最近 {d_data.min() * 100:.2f}cm @ 帧 {best}",
            flush=True,
        )

    arm_initial = arm_target_from_initial()
    if arm_initial is not None:
        print("\n[对齐检查]", flush=True)
        print(
            f"    HDF5 第 0 帧目标 vs initial_joint_qpos 最大差 "
            f"{np.abs(qpos_targets[0] - arm_initial).max():.4f} rad",
            flush=True,
        )
        report_pose(arm_initial, "reset+restore 后")

    if args.hold_frame >= 0:
        i = int(min(args.hold_frame, n_frames - 1))
        print(f"\n[静态保持测试] 帧 {i}", flush=True)
        env.set_joint_qpos(
            {
                name: np.asarray([qpos_targets[i][k]], dtype=np.float64)
                for k, name in enumerate(arm_joint_names)
            }
        )
        env.mj_forward()
        report_pose(qpos_targets[i], "摆位后")
        base_pose_report("摆位后")
        target = qpos_targets[i]
        for k in range(120):
            for _ in range(substeps):
                tau, q, dq = joint_pd(target, ff=motors[i])
                ctrl[:] = 0.0
                ctrl[arm_ids] = tau
                env.step(ctrl)
            env.render()
            if k % 10 == 0 or k == 119:
                q_mj = np.array([md.qpos[adr] for adr in arm_qpos_adr])
                print(
                    f"  hold {k:3d}: 右臂误差 max={np.abs(target[7:] - q[7:]).max():.4f} "
                    f"mean={np.abs(target[7:] - q[7:]).mean():.4f} rad"
                    f"  query-vs-mjdata={np.abs(q - q_mj).max():.2e}"
                    f"  TCP→方块={tcp_cube_distance() * 100:.2f}cm",
                    flush=True,
                )
        for k7 in range(7, 14):
            print(
                f"    {arm_joint_names[k7].split('_', 1)[-1]:28s}"
                f" 目标={target[k7]:+.4f} 实际={q[k7]:+.4f}"
                f" 差={target[k7] - q[k7]:+.4f} tau={tau[k7]:+.2f} dq={dq[k7]:+.3f}",
                flush=True,
            )
        return

    round_idx = 0
    try:
        while True:
            round_idx += 1
            print(f"\n>>> 第 {round_idx} 轮回放开始", flush=True)
            print(f"    方块世界坐标={np.round(cube_pos(), 4)}", flush=True)
            base_pose_report("回放开始")

            # 起手前用 PD 把双臂稳在初始姿态，消除 reset 后重力下坠/接触冲量
            if args.mode == "pd" and arm_initial is not None:
                for k in range(args.settle_frames):
                    for _ in range(substeps):
                        tau, _, dq = joint_pd(arm_initial)
                        ctrl[:] = 0.0
                        ctrl[arm_ids] = tau
                        env.step(ctrl)
                        env.render()
                    if args.debug_pd and (k % 5 == 0 or k == args.settle_frames - 1):
                        print(
                            f"    [settle {k:02d}] |tau|max={np.abs(tau).max():5.2f} "
                            f"|dq|max={np.abs(dq).max():6.3f} "
                            f"饱和={int(np.sum(np.abs(tau) >= tau_limit - 1e-6))}",
                            flush=True,
                        )
                    time.sleep(period)
                print(f"    settle {args.settle_frames} 帧后:", flush=True)
                report_pose(arm_initial, "settle 后")
                base_pose_report("settle 后")

            next_tick = time.perf_counter()
            err_sum, err_count = 0.0, 0
            for i in range(n_frames):
                sim_time_before = env.data.time
                if args.mode == "pd":
                    # 前 blend 帧从初始姿态平滑过渡到录制目标，避免第一帧猛拽
                    target = qpos_targets[i]
                    if arm_initial is not None and i < args.blend_frames:
                        alpha = (i + 1) / args.blend_frames
                        target = (1.0 - alpha) * arm_initial + alpha * target
                    for s in range(substeps):
                        tau, q, _ = joint_pd(target, ff=motors[i])
                        ctrl[:] = 0.0
                        ctrl[arm_ids] = tau
                        ctrl[gripper_ids] = gripper_motors[i]
                        env.step(ctrl)
                        if s == substeps - 1:
                            env.render()
                    err_sum += float(np.abs(target - q).mean())
                    err_count += 1
                else:
                    ctrl[:] = 0.0
                    ctrl[arm_ids] = motors[i]
                    ctrl[gripper_ids] = gripper_motors[i]
                    for _ in range(substeps):
                        env.step(ctrl)
                    env.render()

                sim_dt = env.data.time - sim_time_before
                if not args.fast:
                    next_tick += period
                    wait = next_tick - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    else:
                        next_tick = time.perf_counter()

                if args.mode == "pd" and i in close_frames:
                    print(f"    [闭合帧 {i}] 右臂逐关节 目标/实际/误差:", flush=True)
                    for idx in range(7, 14):
                        print(
                            f"      {arm_joint_names[idx].split('_', 1)[-1]:28s}"
                            f" 目标={target[idx]:+.4f} 实际={q[idx]:+.4f}"
                            f" 差={target[idx] - q[idx]:+.4f}",
                            flush=True,
                        )
                    tcp_now = np.asarray(
                        env.query_site_pos_and_quat([r_ee_site])[r_ee_site]["xpos"],
                        dtype=np.float64,
                    ).ravel()[:3]
                    d_actual = dist_to_cube(tcp_now)
                    d_base = dist_to_cube(fk_tcp(qpos_targets[i]))          # 基座用当前、关节用录制
                    d_data = (
                        dist_to_cube(fk_tcp(qpos_targets[i], base_recorded))  # 基座+关节都用录制
                        if base_recorded is not None
                        else float("nan")
                    )
                    print(
                        f"      TCP→方块: 实际={d_actual * 100:.2f}cm | "
                        f"关节完美={d_base * 100:.2f}cm | 数据本身={d_data * 100:.2f}cm",
                        flush=True,
                    )
                    # 任务空间滞后匹配：当前实际 TCP 对应录制轨迹的第几帧
                    best_k, best_e = 0, float("inf")
                    for k in range(0, min(91, i + 1)):
                        e = float(np.linalg.norm(fk_tcp(qpos_targets[i - k]) - tcp_now))
                        if e < best_e:
                            best_k, best_e = k, e
                    print(
                        f"      TCP 滞后匹配: 最佳滞后 {best_k} 帧"
                        f"（{best_k / args.fps * 1000:.0f}ms）, 残差 {best_e * 100:.2f}cm"
                        f"；第 {i} 帧自身残差 "
                        f"{float(np.linalg.norm(fk_tcp(qpos_targets[i]) - tcp_now)) * 100:.2f}cm",
                        flush=True,
                    )
                    print(
                        f"      TCP-方块 向量={np.round(tcp_now - cube_pos(), 4)}"
                        f"  TCP z={tcp_now[2]:.4f}",
                        flush=True,
                    )
                    tau_c, _, dq_c = joint_pd(target, ff=motors[i])
                    sat = np.where(np.abs(tau_c) >= tau_limit - 1e-6)[0]
                    print(
                        f"      力矩饱和关节: "
                        + (
                            ", ".join(
                                f"{arm_joint_names[k].split('_', 1)[-1]}"
                                f"({tau_c[k]:+.1f})"
                                for k in sat
                            )
                            if len(sat)
                            else "无"
                        )
                        + f"；|dq|max={np.abs(dq_c).max():.2f}rad/s",
                        flush=True,
                    )
                    for c in range(md.ncon):
                        g1, g2 = int(md.contact[c].geom1), int(md.contact[c].geom2)
                        n1 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
                        n2 = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"
                        if right_related(n1) or right_related(n2):
                            print(
                                f"      接触 {n1} <-> {n2} dist={md.contact[c].dist:+.4f}m",
                                flush=True,
                            )
                    base_pose_report(f"闭合帧 {i}")

                if (i + 1) % 10 == 0 or i + 1 == n_frames:
                    dist = tcp_cube_distance()
                    d_data_now = (
                        dist_to_cube(fk_tcp(qpos_targets[i], base_recorded))
                        if base_recorded is not None
                        else float("nan")
                    )
                    if args.mode == "pd":
                        r_tgt = target[7:]
                        r_act = q[7:]
                        r_err = np.abs(r_tgt - r_act)
                        joint_txt = (
                            f"  右臂误差 max={r_err.max():.3f} mean={r_err.mean():.3f} rad"
                            f"  实际仿真增量={sim_dt * 1000:.1f}ms"
                        )
                    else:
                        joint_txt = f"  实际仿真增量={sim_dt * 1000:.1f}ms"
                    print(
                        f"  帧 {i + 1}/{n_frames} sim={env.data.time:.2f}s"
                        f"  TCP→方块={dist * 100:.2f}cm"
                        f" (数据={d_data_now * 100:.2f}cm){joint_txt}"
                        f"  夹爪力R={gripper_motors[i, 1]:.1f}N"
                        + (
                            f"  基座偏移="
                            f"{np.linalg.norm(np.asarray(md.qpos[base_adr:base_adr + 3]) - base_recorded[:3]) * 100:.2f}cm"
                            if base_recorded is not None and base_adr >= 0
                            else ""
                        ),
                        flush=True,
                    )
                    err_sum, err_count = 0.0, 0
            print("", flush=True)
            if not args.loop:
                break
            prepare_scene()
    except KeyboardInterrupt:
        print("\n已停止回放", flush=True)
    finally:
        orca_logger.info("回放结束（仿真停留在最后一帧状态）")


if __name__ == "__main__":
    main()
