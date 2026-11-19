"""用训练好的 pi0.5 策略在 OrcaGym 中闭环推理（右臂 + DEX1 夹爪抓红方块）。

控制链路（三步，缺一不可）:
  1. 反归一化：模型输出 9 维 [-1,1] 动作 → q01/q99 线性映射回物理量
     （臂 = 关节角 rad，夹爪 = 手指开度 m）。openpi 服务端默认已在
     Unnormalize/AbsoluteActions 里做过一次，此时用 --server-output physical 跳过。
  2. 角度 → 力矩：*_mctrl 是 <motor>（ctrl 即力矩），绝不能直接写角度。
     手臂走逐关节 PD（默认按模型等效惯量自动整定，与 replay_hdf5.py 一致），
     夹爪走官方 DEX1 官方阻抗律 f = kp*(q_d - q) - kd*dq，饱和到 ±120N。
  3. 初始姿态对齐：从 LeRobot 数据集第 0 帧 state（或 HDF5 initial_joint_qpos）
     强制 env.set_joint_qpos + mj_forward，保证与训练起点一致。

用法:
    # 不接模型，只验控制链路（用录制动作冒充模型输出，走真实反归一化路径）
    python infer_g1_pick.py --policy dummy --episode 0

    # 接 openpi 策略服务（先在另一终端跑 scripts/serve_policy.py）
    python infer_g1_pick.py --policy server --port 8000 --episode 0
"""

import argparse
import collections
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import pyarrow.parquet as pq

from scene.scene_manager import SceneManager
from conf import g1_dex1_osc_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
PROMPT = "Pick up the red cube and place it in the box."
ARM_DIM = 7
ACTION_DIM = 9


def _load_collection_helpers():
    """复用采集脚本里的姿态钉住/求解器调参函数。"""
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "g1_dex1_collection_tele.py")
    spec = importlib.util.spec_from_file_location("g1_collect_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs_callback(env) -> dict:
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


# ── 归一化 / 反归一化 ────────────────────────────────────────────────────
def load_norm_stats(lerobot_root: str) -> dict[str, np.ndarray]:
    path = Path(lerobot_root) / "norm_stats.json"
    if not path.is_file():
        raise SystemExit(f"缺少归一化统计: {path}（先跑 compute_norm_stats.py）")
    d = json.loads(path.read_text())["norm_stats"]
    return {
        "a_q01": np.asarray(d["actions"]["q01"], dtype=np.float64),
        "a_q99": np.asarray(d["actions"]["q99"], dtype=np.float64),
    }


def denormalize(actions: np.ndarray, ns: dict[str, np.ndarray]) -> np.ndarray:
    """[-1,1] → 物理量：x = (a+1)/2 * (q99-q01+1e-6) + q01（逐维）。

    与 openpi transforms.Normalize(use_quantiles=True) 严格互逆（分母含 1e-6）:
        norm:  a = 2*(x-q01)/(q99-q01+1e-6) - 1
    """
    a = np.asarray(actions, dtype=np.float64)
    q01, q99 = ns["a_q01"], ns["a_q99"]
    return (a + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01


def normalize(values: np.ndarray, ns: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    q01, q99 = ns["a_q01"], ns["a_q99"]
    return 2.0 * (x - q01) / (q99 - q01 + 1e-6) - 1.0


# 训练侧 DeltaActions：前 7 维存"相对当前状态的增量"，夹爪 2 维存绝对值
DELTA_MASK = np.array([True] * ARM_DIM + [False] * (ACTION_DIM - ARM_DIM))


def to_delta(actions_abs: np.ndarray, state9: np.ndarray) -> np.ndarray:
    out = np.asarray(actions_abs, dtype=np.float64).copy()
    out[:, DELTA_MASK] -= np.asarray(state9, dtype=np.float64)[DELTA_MASK]
    return out


def from_delta(actions_delta: np.ndarray, state9: np.ndarray) -> np.ndarray:
    """AbsoluteActions：把增量加回当前状态，得到绝对目标角。"""
    out = np.asarray(actions_delta, dtype=np.float64).copy()
    out[:, DELTA_MASK] += np.asarray(state9, dtype=np.float64)[DELTA_MASK]
    return out


# ── 图像预处理：与训练一致（等比缩放 + 上下补黑边，不拉伸）───────────────
def resize_with_pad(bgr: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    return out


class CameraGrabber:
    """从 OrcaStudio 的 WebSocket H.264 流里持续抓最新一帧（后台线程）。"""

    HEADER = 12  # 8B timestamp + 4B simulate_index

    def __init__(self, env, camera_name: str, label: str):
        import av  # noqa: F401  (确保解码依赖可用)

        self.label = label
        rec = env.get_recorder_manager().get_recorder(camera_name)
        if rec is None:
            raise RuntimeError(f"相机 {label} 没有录制器")
        self._uri = f"ws://localhost:{int(rec.color_port)}"
        self._frame: np.ndarray | None = None
        self._seq = 0        # 最新已解码帧的序号
        self._buf: collections.deque = collections.deque(maxlen=8)  # (序号, 帧)
        self._count = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import av
        import websockets.sync.client

        while self._running:
            try:
                with websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None
                ) as ws:
                    # 每条 WebSocket 消息就是一帧完整的 H.264 NAL（Annex-B 基本流）。
                    # 必须用持久化的 CodecContext 增量解码，保持 DPB 参考帧状态；
                    # 不能把流式数据喂给 av.open() 解析容器 —— 那样只会解出首帧。
                    # 参考 orca_gym/recorder/camera_recorder.py:_decode_nal。
                    codec = av.CodecContext.create("h264", "r")
                    while self._running:
                        data = ws.recv()
                        if isinstance(data, str) or len(data) < self.HEADER:
                            continue
                        try:
                            for frame in codec.decode(av.Packet(data[self.HEADER :])):
                                self._frame = frame.to_ndarray(format="bgr24")
                                self._count += 1
                                self._seq = self._count
                                self._buf.append((self._seq, self._frame))
                        except av.FFmpegError:
                            continue  # 单帧解码失败保留上一帧，不重连
            except Exception as exc:
                if not self._running:
                    break
                print(f"[相机 {self.label}] 流断开: {exc}，1s 后重连", flush=True)
                time.sleep(1.0)

    def latest(self) -> np.ndarray | None:
        return self._frame

    def at_ordinal(self, ordinal: int, timeout: float = 1.0) -> np.ndarray | None:
        """取第 ordinal 帧（按本路相机的收帧顺序计数）。

        流是"每次 env.render() 推一帧"，所以帧序号 == 控制步序号。按序号取帧
        而不是"取最新一帧"，两路相机才不会差一步，同一 seed 才能逐位复现。
        """
        t0 = time.time()
        while time.time() - t0 < timeout and self._running:
            if self._seq >= ordinal:
                for o, f in reversed(self._buf):
                    if o == ordinal:
                        return f
                if self._buf:      # 目标帧已被挤出小缓冲：退回缓冲里最老的一帧
                    return self._buf[0][1]
            time.sleep(0.002)
        return self._frame

    def wait_fresh(self, timeout: float = 0.25) -> np.ndarray | None:
        """等到比调用时刻更新的一帧，避免把滞后的旧帧喂给策略。"""
        base = self._count
        t0 = time.time()
        while self._count == base and time.time() - t0 < timeout and self._running:
            time.sleep(0.002)
        return self._frame

    def stop(self):
        self._running = False


# ── 策略客户端 ───────────────────────────────────────────────────────────
class DummyPolicy:
    """用录制的 9 维动作冒充模型输出，但故意走完整的归一化管线
    （取增量 → 归一化到 [-1,1]），从而把推理脚本的反归一化路径真实跑一遍。
    不需要 checkpoint，只验控制链路。"""

    def __init__(self, actions: np.ndarray, ns: dict, horizon: int):
        self._actions = actions
        self._ns = ns
        self._horizon = horizon
        self._i = 0
        self.roundtrip_err = None
        self.last_norm_range = None

    def infer(self, obs) -> dict:
        if self._i >= len(self._actions):
            raise StopIteration
        state9 = np.asarray(obs["state"], dtype=np.float64)
        chunk = self._actions[self._i : self._i + self._horizon]
        self._i += len(chunk)
        # 与训练/服务端一致：前 7 维取增量 → 用同一份 q01/q99 归一化
        normed = normalize(to_delta(chunk, state9), self._ns)
        self.last_norm_range = (float(normed.min()), float(normed.max()))
        if self.roundtrip_err is None:
            back = from_delta(denormalize(normed, self._ns), state9)
            self.roundtrip_err = float(np.abs(back - chunk).max())
        return {"actions": normed}


class ServerPolicy:
    """openpi serve_policy.py 的 websocket 客户端（msgpack-numpy 协议）。"""

    def __init__(self, host: str, port: int):
        client_src = "/home/user/openpi/packages/openpi-client/src"
        if client_src not in sys.path:
            sys.path.insert(0, client_src)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        print(f"[策略] 连接 ws://{host}:{port} ...", flush=True)
        self._policy = WebsocketClientPolicy(host=host, port=port)
        print(f"[策略] 已连接，metadata={self._policy.get_server_metadata()}", flush=True)

    def infer(self, obs) -> dict:
        return self._policy.infer(obs)


def build_policy_obs(
    cameras: dict, state9: np.ndarray, prompt: str, targets: dict | None = None
) -> dict:
    """组装服务端期望的观测键。

    serve_policy.py 调 create_trained_policy 时没有传 repack_transforms，
    所以服务端不执行 RepackTransform，客户端必须直接给 repack 之后的结构：
        {"state": (9,), "images": {"base_0_rgb": CHW, ...}, "prompt": str}

    targets 不为空时，每路相机都取「基准序号 + 步数」的那一帧，保证两路是
    同一个控制步的画面（否则 base 与 wrist 差一步，同 seed 也无法复现）。
    """
    obs = {
        "state": np.asarray(state9, dtype=np.float32),
        "images": {},
        "prompt": prompt,
    }
    for key, grab in cameras.items():
        frame = grab.at_ordinal(targets[key]) if targets else grab.wait_fresh()
        if frame is None:
            raise RuntimeError(f"相机 {key} 还没有帧")
        rgb = cv2.cvtColor(resize_with_pad(frame, 224), cv2.COLOR_BGR2RGB)
        obs["images"][key] = np.transpose(rgb, (2, 0, 1))  # CHW uint8
    return obs


def build_demo_obs(demo_frames: dict, states9: np.ndarray, prompt: str, idx: int) -> dict:
    """开环：观测全部取自演示数据，仿真反馈不参与决策（只执行模型动作）。"""
    obs = {
        "state": np.asarray(states9[idx], dtype=np.float32),
        "images": {},
        "prompt": prompt,
    }
    for key, frames in demo_frames.items():
        bgr = frames[min(idx, len(frames) - 1)]
        rgb = cv2.cvtColor(resize_with_pad(bgr, 224), cv2.COLOR_BGR2RGB)
        obs["images"][key] = np.transpose(rgb, (2, 0, 1))
    return obs


def main():
    ap = argparse.ArgumentParser(description="pi0.5 策略闭环推理（G1 右臂抓方块）")
    ap.add_argument("--level", default="pi")
    ap.add_argument("--agent_name", default="g1_pick_usda")
    ap.add_argument("--lerobot_root", default="/home/user/pi/lerobot_g1_pick")
    ap.add_argument(
        "--norm_root",
        default=None,
        help="归一化统计来源（默认同 --lerobot_root；验证集不含 norm_stats，应指向训练集）",
    )
    ap.add_argument("--episode", type=int, default=0, help="取该集第 0 帧做初始姿态；dummy 模式还取它的动作")
    ap.add_argument("--policy", choices=["dummy", "server"], default="server")
    ap.add_argument("--host", default="192.168.110.57")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--server-output",
        choices=["physical", "normalized"],
        default="physical",
        help="server 返回的是物理量（openpi 默认，已 Unnormalize+Absolute）还是 [-1,1]",
    )
    ap.add_argument(
        "--openloop-demo",
        action="store_true",
        help="开环验证：模型输入全部取自演示数据（图像 + state），"
             "仿真反馈不参与决策，只把模型动作灌进仿真",
    )
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--cameras", default="Camera,wrist_r", help="base,腕部相机逻辑名（与采集一致）")
    ap.add_argument("--no-cameras", action="store_true", help="dummy 模式下不接相机")
    ap.add_argument("--dump-obs", default=None, help="把第 0 步的观测图存到该目录（排查画面不一致）")
    ap.add_argument(
        "--dump-dir",
        default="./debug_images",
        help="每次推理时把两路 224x224 观测图存到该目录（step_XXXX_base.png / step_XXXX_wrist.png）",
    )
    ap.add_argument("--dump-every", type=int, default=10, help="每隔多少步 dump 一次图像")
    ap.add_argument(
        "--blend-steps",
        type=int,
        default=5,
        help="跨块平滑：新块前 N 步与上一块对应位置做线性加权（0 = 关闭）",
    )
    ap.add_argument(
        "--blend-alpha",
        type=float,
        default=0.5,
        help="跨块平滑：重叠区起始时旧块权重（线性衰减到 0）",
    )
    ap.add_argument(
        "--stall-window",
        type=int,
        default=60,
        help="看门狗：观察多少步内的关节变化判断是否停滞（0 = 关闭看门狗）",
    )
    ap.add_argument("--stall-tol", type=float, default=0.01, help="看门狗：窗口内关节最大变化小于该值算停滞（rad）")
    ap.add_argument("--stall-box-cm", type=float, default=8.0, help="看门狗：方块距盒子小于该值才允许接管（cm）")
    ap.add_argument("--release-steps", type=int, default=60, help="看门狗：强制释放持续多少步")
    ap.add_argument("--release-target", type=float, default=-0.020, help="看门狗：强制释放时下发的夹爪目标（数据集量纲，-0.020 为全开）")
    ap.add_argument(
        "--execute-steps",
        type=int,
        default=10,
        help="每个动作块只执行前 N 步就重新推理（滚动时域，抑制流匹配采样漂移；设为 50 即整块开环）",
    )
    ap.add_argument(
        "--hdf5_init",
        default=None,
        help="可选：从 HDF5 的 initial_joint_qpos 恢复完整姿态（含基座与左臂，场景与录制完全一致）",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--warmup-demo",
        type=int,
        default=0,
        help="诊断用：前 N 步改为执行演示动作（验证是否起始阶段偏离流形）",
    )
    ap.add_argument("--horizon", type=int, default=50, help="每次推理输出的动作块长度")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--kp", type=float, default=300.0)
    ap.add_argument("--kd", type=float, default=20.0)
    ap.add_argument("--manual-gains", dest="auto_gains", action="store_false", help="关闭自动增益")
    ap.add_argument(
        "--no-gravity-comp",
        dest="gravity_comp",
        action="store_false",
        help="关闭重力/科氏补偿（只留纯 PD）",
    )
    ap.add_argument("--wn", type=float, default=60.0)
    ap.add_argument("--zeta", type=float, default=1.0)
    ap.add_argument("--max-tau", type=float, default=25.0)
    ap.add_argument("--max-tau-wrist", type=float, default=5.0)
    ap.add_argument(
        "--grip-range",
        type=float,
        nargs=2,
        default=(-0.020, 0.0245),
        help="DEX1 手指机械行程（开 / 合），用于把数据集里的指位映射成夹持力指令",
    )
    ap.add_argument("--settle-frames", type=int, default=15)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--task_config", default="example.yaml")
    ap.add_argument("--orcagym_addr", default="localhost:50051")
    args = ap.parse_args()

    # ── 数据集：初始姿态 + dummy 动作用 ─────────────────────────────────
    parquet = (
        Path(args.lerobot_root) / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    )
    table = pq.read_table(parquet)
    states9 = np.stack(table.column("observation.state").to_pylist())
    demo_actions9 = np.stack(table.column("action").to_pylist())
    init_state9 = states9[0].astype(np.float64)
    ns = load_norm_stats(args.norm_root or args.lerobot_root)

    demo_frames = None
    demo_len = len(states9)
    if args.openloop_demo:
        demo_frames = {}
        for cam in ("base_0_rgb", "right_wrist_0_rgb"):
            path = (Path(args.lerobot_root) / "videos" / "chunk-000"
                    / f"observation.images.{cam}" / f"episode_{args.episode:06d}.mp4")
            cap = cv2.VideoCapture(str(path))
            buf = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                buf.append(bgr)
            cap.release()
            demo_frames[cam] = buf
        demo_len = min(len(states9), *(len(v) for v in demo_frames.values()))
        print(f"[开环] 演示数据对齐长度 {demo_len}（state {len(states9)}，"
              f"base {len(demo_frames['base_0_rgb'])}，"
              f"wrist {len(demo_frames['right_wrist_0_rgb'])}）", flush=True)

    print("=" * 66, flush=True)
    print(f"  推理集    {parquet}", flush=True)
    print(f"  模式      {args.policy}"
          f"{'（server 输出=' + args.server_output + '）' if args.policy == 'server' else '（录制动作冒充模型）'}",
          flush=True)
    print(f"  归一化    q01/q99 = {np.round(ns['a_q01'], 4).tolist()} / {np.round(ns['a_q99'], 4).tolist()}",
          flush=True)
    print(f"  动作维度  {ACTION_DIM}（右臂 7 + 夹爪 2）", flush=True)
    print("=" * 66, flush=True)

    helpers = _load_collection_helpers()

    default_joint_values = {
        name: 0.0
        for name in g1_dex1_osc_conf.l_arm["joint_names"] + g1_dex1_osc_conf.r_arm["joint_names"]
    }
    from yaml import load, Loader

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.task_config), encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    if "data_collection" in config:
        config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="Infer",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=_obs_callback,
        env_index=0,
        frame_skip=1,          # 每个 env.step 只推进 1ms，PD 在 1kHz 闭环
        time_step=0.001,
        orcagym_addr=args.orcagym_addr,
        scene_manager=scene_manager,
    )
    env = manager.env

    arm_actuators = g1_dex1_osc_conf.l_arm["motors_names"] + g1_dex1_osc_conf.r_arm["motors_names"]
    gripper_actuators = (
        g1_dex1_osc_conf.gripper_l["actuator_names"] + g1_dex1_osc_conf.gripper_r["actuator_names"]
    )
    arm_ids = [env.model.actuator_name2id(env.actuator(n)) for n in arm_actuators]
    gripper_ids = [env.model.actuator_name2id(env.actuator(n)) for n in gripper_actuators]
    arm_joint_names = [
        env.joint(n)
        for n in g1_dex1_osc_conf.l_arm["joint_names"] + g1_dex1_osc_conf.r_arm["joint_names"]
    ]
    right_joint_names = arm_joint_names[-ARM_DIM:]
    grip_joint_names = [env.joint(n) for n in g1_dex1_osc_conf.gripper_r["joint_names"]]
    # 数据集 fps=30（帧间距实测 33.3ms），所以一个控制步必须推进 33ms，
    # 否则策略会以 6.7 倍速被执行：每块 10 帧只覆盖 50ms 而不是 333ms，
    # 机械臂跟不上计划，夹爪就会在半路合拢。33 × 1ms = 33ms。
    substeps = 33
    physics_dt = 0.001

    env.reset()
    time.sleep(0.1)
    manager.update_scene()
    helpers.apply_grasp_solver_tuning(env)
    helpers.pin_all_joints(env, args.agent_name)

    # ── ③ 初始姿态对齐：把数据集第 0 帧的 9 维 state 写进仿真 ────────────
    def read_state9() -> np.ndarray:
        q_arm = env.query_joint_qpos(right_joint_names)
        q_grip = env.query_joint_qpos(grip_joint_names)
        return np.array(
            [float(np.asarray(q_arm[n]).ravel()[0]) for n in right_joint_names]
            + [float(np.asarray(q_grip[n]).ravel()[0]) for n in grip_joint_names],
            dtype=np.float64,
        )

    def align_initial_pose(target9: np.ndarray, label: str):
        want = {n: np.asarray([target9[i]], dtype=np.float64) for i, n in enumerate(right_joint_names)}
        want.update({n: np.asarray([target9[ARM_DIM + i]], dtype=np.float64)
                     for i, n in enumerate(grip_joint_names)})
        env.set_joint_qpos(want)
        env.mj_forward()
        got = read_state9()
        print(f"[初始姿态 {label}] 数据集第 0 帧 = {np.round(target9, 4).tolist()}", flush=True)
        print(f"[初始姿态 {label}] 对齐后实际   = {np.round(got, 4).tolist()}", flush=True)
        print(f"[初始姿态 {label}] 最大偏差 {np.abs(got - target9).max():.6f}"
              f"（{'已对齐' if np.abs(got - target9).max() < 1e-4 else '未对齐!'}）", flush=True)

    align_initial_pose(init_state9, "数据集")

    if args.hdf5_init:
        import h5py

        with h5py.File(args.hdf5_init, "r") as f:
            init_qpos = json.loads(f["initial_joint_qpos"][()].decode())
        env.set_joint_qpos(
            {k: np.asarray(v, dtype=np.float64) for k, v in init_qpos.items()}
        )
        env.mj_forward()
        init_state9 = read_state9()
        print(f"[初始姿态 HDF5] 已恢复完整姿态（含基座/左臂）：{args.hdf5_init}", flush=True)
        align_initial_pose(init_state9, "HDF5")

    # ── ② 角度 → 力矩：逐关节 PD（自动增益）+ 夹爪阻抗 ───────────────────
    import mujoco

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    mj, md = gym._mjModel, gym._mjData
    tau_limit = np.array(
        [args.max_tau] * 5 + [args.max_tau_wrist] * 2 + [args.max_tau] * 5 + [args.max_tau_wrist] * 2,
        dtype=np.float64,
    )
    kp_vec = np.full(14, args.kp, dtype=np.float64)
    kd_vec = np.full(14, args.kd, dtype=np.float64)
    dof_adr = [
        int(mj.jnt_dofadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in arm_joint_names
    ]
    if args.auto_gains:
        scratch = mujoco.MjData(mj)
        scratch.qpos[:] = md.qpos
        mujoco.mj_forward(mj, scratch)  # 必须先 forward，d.qM 才会有内容
        mass_mat = np.zeros((mj.nv, mj.nv), dtype=np.float64)
        for col in range(mj.nv):
            unit = np.zeros(mj.nv, dtype=np.float64)
            unit[col] = 1.0
            res = np.zeros(mj.nv, dtype=np.float64)
            mujoco.mj_mulM(mj, scratch, res, unit)
            mass_mat[:, col] = res
        joint_j = np.array([mass_mat[d, d] for d in dof_adr])
        kp_vec = joint_j * args.wn**2
        kd_vec = np.minimum(2.0 * args.zeta * joint_j * args.wn, 1.0 * joint_j / physics_dt)
        print(f"[自动增益] wn={args.wn:g} zeta={args.zeta:g}，右臂 7 关节：", flush=True)
        for k in range(7, 14):
            print(f"    {arm_joint_names[k].split('_', 1)[-1]:26s} J={joint_j[k]:.5f}"
                  f" kp={kp_vec[k]:8.2f} kd={kd_vec[k]:6.3f} 上限={tau_limit[k]:.0f}", flush=True)

    def read_arm_qd():
        q = env.query_joint_qpos(arm_joint_names)
        dq = env.query_joint_qvel(arm_joint_names)
        return (
            np.array([float(np.asarray(q[n]).ravel()[0]) for n in arm_joint_names]),
            np.array([float(np.asarray(dq[n]).ravel()[0]) for n in arm_joint_names]),
        )

    def arm_torque(target14: np.ndarray) -> np.ndarray:
        """角度目标 → 关节力矩，绝不把角度直接写进 ctrl。

        tau = kp*(q_d - q) - kd*dq + 重力/科氏补偿
        没有前馈时，纯 PD 会有 ~0.03-0.06 rad 的重力下垂（末端差 1cm 量级），
        补上 qfrc_bias 才能像 OSC 控制器那样把末端送到位。
        """
        q, dq = read_arm_qd()
        tau = kp_vec * (target14 - q) - kd_vec * dq
        if args.gravity_comp:
            tau = tau + np.asarray(md.qfrc_bias, dtype=np.float64)[dof_adr]
        return np.clip(tau, -tau_limit, tau_limit)

    grip_kp = float(g1_dex1_osc_conf.gripper_r["kp"])
    grip_kd = float(g1_dex1_osc_conf.gripper_r["kd"])
    # 数据集里的夹爪通道是"被方块挡住后的实测指位"（约 -0.0207 开 / +0.0052 合），
    # 直接当位置目标下发，接触后 PD 误差≈0 → 夹不紧。这里把它线性映射到机械行程
    # （-0.020 开 / +0.0245 合），闭合时才有持续的压紧力。
    g_q01, g_q99 = float(ns["a_q01"][ARM_DIM]), float(ns["a_q99"][ARM_DIM])
    grip_open_mech, grip_close_mech = args.grip_range
    print(f"[夹爪] 数据集开合范围 [{g_q01:+.4f}, {g_q99:+.4f}] → 映射到机械行程 "
          f"[{grip_open_mech:+.4f}, {grip_close_mech:+.4f}]，kp={grip_kp:g} kd={grip_kd:g}", flush=True)

    def gripper_force(target_q: float) -> float:
        """指位目标 → 夹持力：先把开合度映射到机械行程，再走 DEX1 阻抗律，±120N。"""
        a = float(np.clip((target_q - g_q01) / max(g_q99 - g_q01, 1e-6), 0.0, 1.0))
        q_cmd = grip_open_mech + a * (grip_close_mech - grip_open_mech)
        q = env.query_joint_qpos([grip_joint_names[0]])[grip_joint_names[0]]
        dq = env.query_joint_qvel([grip_joint_names[0]])[grip_joint_names[0]]
        f = grip_kp * (q_cmd - float(np.asarray(q).ravel()[0])) \
            - grip_kd * float(np.asarray(dq).ravel()[0])
        return float(np.clip(f, -120.0, 120.0))

    ctrl = np.zeros(env.nu, dtype=np.float32)
    period = 1.0 / args.fps

    # 末端到红方块的距离（判断推理是否真的走到位再闭合）
    cube_bid = next(
        (i for i in range(mj.nbody)
         if "cube" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()),
        -1,
    )
    ee_sid = env.site(g1_dex1_osc_conf.r_arm["ee_site_name"])

    def tcp_cube_cm() -> float:
        if cube_bid < 0:
            return float("nan")
        tcp = np.asarray(env.query_site_pos_and_quat([ee_sid])[ee_sid]["xpos"]).ravel()[:3]
        return float(np.linalg.norm(tcp - np.asarray(md.xpos[cube_bid], dtype=np.float64))) * 100.0

    # 盒子（用于判断是否已经到达盒子上方 / 是否把方块放进去）
    box_bids = [
        i for i in range(mj.nbody)
        if "box" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
    ]
    print(f"[场景] 盒子候选 body: {[(i, mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i)) for i in box_bids]}",
          flush=True)

    def cube_box_cm() -> float:
        """方块到最近盒子中心的水平距离（cm）；没有盒子返回 nan。"""
        if cube_bid < 0 or not box_bids:
            return float("nan")
        c = np.asarray(md.xpos[cube_bid], dtype=np.float64)
        return float(min(np.linalg.norm(c[:2] - np.asarray(md.xpos[b], dtype=np.float64)[:2]) for b in box_bids)) * 100.0

    # 数据里只有右臂：左臂保持复位姿态不动，避免甩动改变俯视相机的画面
    q_left = env.query_joint_qpos(arm_joint_names[:ARM_DIM])
    left_arm_hold = np.array(
        [float(np.asarray(q_left[n]).ravel()[0]) for n in arm_joint_names[:ARM_DIM]],
        dtype=np.float64,
    )
    home14 = np.concatenate([left_arm_hold, init_state9[:ARM_DIM]])
    print(f"[左臂] 保持复位姿态 {np.round(left_arm_hold, 3).tolist()}", flush=True)

    # 起手稳定：PD 顶住初始目标（角度 → 力矩，不直接写 ctrl）
    for _ in range(args.settle_frames):
        for _s in range(substeps):
            ctrl[:] = 0.0
            ctrl[arm_ids] = arm_torque(home14)
            ctrl[gripper_ids] = [0.0, gripper_force(float(init_state9[ARM_DIM]))]
            env.step(ctrl)
        env.render()

    # ── 相机 ─────────────────────────────────────────────────────────────
    cameras = {}
    if not args.no_cameras and not args.openloop_demo:
        try:
            logical = [c.strip() for c in args.cameras.split(",") if c.strip()]
            resolved = helpers._resolve_cameras(env, logical)
            print(f"[相机] 名称解析 {dict(zip(logical, [resolved[c] for c in logical]))}", flush=True)
            for key, lg in zip(("base_0_rgb", "right_wrist_0_rgb"), logical):
                cam = resolved[lg]
                try:
                    env.start_streaming(cam, capture_rgb=True)
                except ValueError:  # 端口冲突时退回默认端口
                    env.start_streaming(
                        cam, capture_rgb=True,
                        color_port=helpers._CAMERA_FALLBACK_PORTS.get(lg, 7070),
                    )
                cameras[key] = CameraGrabber(env, cam, key)
                print(f"[相机] {key} ← {cam}（已连接）", flush=True)
            # 引擎只在 render() 时编码推流帧，等首帧期间必须持续渲染
            deadline = time.time() + 8.0
            while time.time() < deadline and any(g.latest() is None for g in cameras.values()):
                env.render()
                time.sleep(0.05)
            missing = [k for k, g in cameras.items() if g.latest() is None]
            print(f"[相机] 首帧就绪 {[k for k in cameras if k not in missing]}"
                  f"{'，缺 ' + str(missing) if missing else ''}", flush=True)
            if missing and args.policy == "server":
                raise RuntimeError(f"相机无帧: {missing}")
        except Exception as exc:
            print(f"[相机] 启动失败: {exc}；dummy 模式可加 --no-cameras 跳过", flush=True)
            if args.policy == "server":
                raise

    # ── 策略 ─────────────────────────────────────────────────────────────
    if args.policy == "dummy":
        policy = DummyPolicy(demo_actions9, ns, args.horizon)
    else:
        policy = ServerPolicy(args.host, args.port)

    def left_arm_target() -> np.ndarray:
        return left_arm_hold

    queue: collections.deque = collections.deque()
    step = 0
    cam_anchor: int | None = None      # 相机帧索引锚点：推理时取 simulate_index = 锚点 + step
    cam_anchor_ready = False
    prev_chunk: np.ndarray | None = None   # 上一块（已平滑后的绝对动作）
    prev_start = 0                          # 上一块对应的起始步
    first_close_dumped = False
    grip_mid = 0.5 * (float(ns["a_q01"][ARM_DIM]) + float(ns["a_q99"][ARM_DIM]))
    # 停滞看门狗：策略在盒子上方会进入"保持不动"的不动点，夹爪永远不松；
    # 这里检测到"已到盒子上方 + 关节不再变化"就接管夹爪强制张开。
    q_hist: collections.deque = collections.deque(maxlen=max(args.stall_window, 1))
    release_left = 0
    release_done = False
    err_peak, err_peak_step, err_peak_joint = 0.0, 0, ""
    t_start = time.time()
    print("-" * 66, flush=True)
    print("[推理] 开始循环（Ctrl+C 停止）", flush=True)
    try:
        while step < args.max_steps:
            if (
                demo_actions9 is not None
                and step < args.warmup_demo
                and not queue
            ):
                # 诊断：前 N 步先用演示动作，把机器人送入演示流形后再交给策略
                queue.append(demo_actions9[step])
            if not queue:
                if cameras and not cam_anchor_ready:
                    cam_anchor_ready = True
                    t0 = time.time()
                    while time.time() - t0 < 3.0 and any(
                        g._count == 0 for g in cameras.values()
                    ):
                        time.sleep(0.01)
                    cam_anchor = {k: g._count for k, g in cameras.items()}
                    print(f"[相机] 对齐基准帧序号 {cam_anchor}（之后取 基准+步数 的帧）",
                          flush=True)
                targets = (
                    None if cam_anchor is None
                    else {k: cam_anchor[k] + step for k in cameras}
                )
                if args.openloop_demo:
                    obs = build_demo_obs(demo_frames, states9, args.prompt, min(step, demo_len - 1))
                else:
                    obs = build_policy_obs(cameras, read_state9(), args.prompt, targets)
                out = policy.infer(obs)
                chunk = np.array(out["actions"], dtype=np.float64)  # copy：服务端返回的数组只读，平滑要原地写
                raw_range = (float(chunk.min()), float(chunk.max()))
                if args.policy == "dummy" or args.server_output == "normalized":
                    # ① 反归一化：[-1,1] → 增量物理量；再按 AbsoluteActions 加回当前状态
                    chunk = from_delta(denormalize(chunk, ns), obs["state"])
                # ── 跨块平滑：新块前 N 步与上一块在时间轴上对齐后线性混合 ──
                # 上一块从 prev_start 开始，本块从 step 开始，时间偏移 shift = step - prev_start；
                # 本块第 i 步对应上一块第 shift+i 步，重叠区旧块权重从 alpha 线性衰减到 0。
                if prev_chunk is not None and args.blend_steps > 0:
                    shift = step - prev_start
                    k = min(args.blend_steps, chunk.shape[0], max(prev_chunk.shape[0] - shift, 0))
                    if k > 0:
                        w_old = np.linspace(args.blend_alpha, 0.0, k)[:, None]
                        chunk[:k] = (1.0 - w_old) * chunk[:k] + w_old * prev_chunk[shift : shift + k]
                        print(f"[平滑] 步 {step:4d} 与上一块重叠 {k} 步混合"
                              f"（旧块权重 {args.blend_alpha:.2f}→0）", flush=True)
                prev_chunk, prev_start = chunk.copy(), step
                print(f"[推理] 第 {step:4d} 步拿到动作块 {chunk.shape}，"
                      f"模型输出范围 [{raw_range[0]:+.3f}, {raw_range[1]:+.3f}]"
                      f" → 目标角 [{chunk.min():+.3f}, {chunk.max():+.3f}]", flush=True)
                if step == 0 and cube_bid >= 0:
                    print(f"[场景] 方块世界坐标 {np.round(md.xpos[cube_bid][:3], 4).tolist()}"
                          f"（每集应一致，变了说明场景没重置）", flush=True)
                if step == 0 and demo_actions9 is not None and len(demo_actions9) > 1:
                    # 与服务端同源对比：模型第一个动作块 vs 演示动作
                    n = min(len(chunk), len(demo_actions9))
                    print(f"[对比] 模型 chunk[0]  = {np.round(chunk[0], 3).tolist()}", flush=True)
                    print(f"[对比] 演示 chunk[0]  = {np.round(demo_actions9[0], 3).tolist()}", flush=True)
                    print(f"[对比] 前 {n} 帧臂 MAE = "
                          f"{np.abs(chunk[:n, :ARM_DIM] - demo_actions9[:n, :ARM_DIM]).mean():.4f} rad", flush=True)
                if args.policy == "dummy" and policy.roundtrip_err is not None:
                    print(f"[反归一化] 归一化→反归一化→加回状态 往返最大误差 "
                          f"{policy.roundtrip_err:.3e}（应 ~1e-15）", flush=True)
                if cameras and step == 0:
                    for k in ("base_0_rgb", "right_wrist_0_rgb"):
                        img = obs["images"].get(k)
                        if img is not None:
                            print(f"[观测] {k} shape={img.shape} {img.dtype} "
                                  f"mean={img.mean():.1f}", flush=True)
                    print(f"[观测] state={obs['state'].round(4).tolist()}"
                          f"  prompt={obs['prompt']!r}", flush=True)
                    if args.dump_obs:
                        os.makedirs(args.dump_obs, exist_ok=True)
                        for k, img in obs["images"].items():
                            rgb = np.transpose(img, (1, 2, 0))
                            cv2.imwrite(os.path.join(args.dump_obs, f"{k}.png"),
                                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        print(f"[观测] 已保存到 {args.dump_obs}", flush=True)
                # ── 调试：真实 qpos / 未来 10 步夹爪 / 隐含增量 / 画面 dump ──
                q_real = read_state9()[:ARM_DIM]
                just_closed = (not first_close_dumped) and float(chunk[0, ARM_DIM]) > grip_mid
                if just_closed:
                    first_close_dumped = True
                    print(f"[调试] 首次出现闭合指令 @ 步 {step}：chunk[0] 夹爪={chunk[0, ARM_DIM]:+.4f}"
                          f"（闭合阈值 {grip_mid:+.4f}）TCP→方块={tcp_cube_cm():.2f}cm", flush=True)
                if step % 10 == 0 or just_closed:
                    print(f"[调试] 步 {step:4d} 真实qpos[:7]   = {np.round(q_real, 4).tolist()}", flush=True)
                    print(f"[调试] 步 {step:4d} chunk[0:10,7:9] = {np.round(chunk[:10, ARM_DIM:], 4).tolist()}", flush=True)
                    print(f"[调试] 步 {step:4d} 隐含增量[0:5]  = {np.round(chunk[:5, :ARM_DIM] - q_real, 4).tolist()}", flush=True)
                if args.dump_dir and (step < 3 or step % args.dump_every == 0 or just_closed):
                    os.makedirs(args.dump_dir, exist_ok=True)
                    saved = []
                    for key, img in obs["images"].items():
                        short = "base" if key == "base_0_rgb" else "wrist"
                        rgb = np.transpose(img, (1, 2, 0))
                        path = os.path.join(args.dump_dir, f"step_{step:04d}_{short}.png")
                        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        saved.append(path)
                    print(f"[调试] 步 {step:4d} 已保存 {saved}", flush=True)
                queue.extend(chunk[: args.execute_steps])

            action9 = queue.popleft()
            if cameras and step % 30 == 0:
                counts = {f"{k}(seq={g._seq})": g._count for k, g in cameras.items()}
                print(f"[相机] 步 {step:3d} 累计帧数 {counts}", flush=True)
            # ── 停滞看门狗：到盒子上方且手臂停止运动 → 强制松爪 ──
            if args.stall_window > 0 and release_left == 0:
                q_hist.append(read_state9()[:ARM_DIM])
                if (
                    len(q_hist) == args.stall_window
                    and float(np.ptp(np.stack(q_hist), axis=0).max()) < args.stall_tol
                    and cube_box_cm() < args.stall_box_cm
                    and float(np.mean(action9[ARM_DIM:])) > grip_mid
                ):
                    release_left = args.release_steps
                    print(f"[看门狗] 步 {step}：盒子上方停滞 {args.stall_window} 步"
                          f"（关节变化 < {args.stall_tol} rad），方块→盒子={cube_box_cm():.2f}cm，"
                          f"接管夹爪强制张开 {release_left} 步", flush=True)
            if release_left > 0:
                # 手臂保持当前姿态（冻结目标 = 当前 qpos），只改夹爪
                target14 = np.concatenate([left_arm_target(), read_state9()[:ARM_DIM]])
                grip_target = args.release_target
                release_left -= 1
                if release_left == 0:
                    release_done = True
            else:
                target14 = np.concatenate([left_arm_target(), action9[:ARM_DIM]])
                grip_target = float(np.mean(action9[ARM_DIM:]))
            for s in range(substeps):
                ctrl[:] = 0.0
                ctrl[arm_ids] = arm_torque(target14)
                ctrl[gripper_ids] = [gripper_force(-0.020), gripper_force(grip_target)]
                env.step(ctrl)
                if s == substeps - 1:
                    env.render()
            step += 1
            q_now, _ = read_arm_qd()
            err_now = np.abs(q_now[7:] - action9[:ARM_DIM])
            if err_now.max() > err_peak:
                err_peak = float(err_now.max())
                err_peak_step = step
                err_peak_joint = right_joint_names[int(err_now.argmax())]
            if step % 30 == 0:
                err = err_now
                print(f"[推理] 步 {step:4d}  TCP→方块={tcp_cube_cm():5.2f}cm"
                      f"  方块z={float(md.xpos[cube_bid][2]) if cube_bid >= 0 else float('nan'):.3f}m"
                      f"  方块→盒子={cube_box_cm():5.2f}cm"
                      f"  跟踪误差 max={err.max():.4f} mean={err.mean():.4f}"
                      f"  右夹爪目标={grip_target:+.4f}", flush=True)
                if args.openloop_demo:
                    ref = states9[min(step, demo_len - 1)]
                    rep = np.abs(read_state9() - ref)
                    print(f"[开环] 步 {step:4d} 仿真 vs 演示关节偏差 max={rep.max():.4f} "
                          f"mean={rep.mean():.4f} rad（臂 {rep[:ARM_DIM].max():.4f} / "
                          f"夹爪 {rep[ARM_DIM:].max():.4f}）", flush=True)
            if release_done:
                print(f"[看门狗] 步 {step}：释放完成，流程结束"
                      f"（方块z={float(md.xpos[cube_bid][2]) if cube_bid >= 0 else float('nan'):.3f}m，"
                      f"方块→盒子={cube_box_cm():.2f}cm）", flush=True)
                break
            time.sleep(period)
    except StopIteration:
        print("[推理] dummy 动作已播完", flush=True)
    except KeyboardInterrupt:
        print("\n[推理] 手动中断", flush=True)
    finally:
        for grab in cameras.values():
            grab.stop()

    print(f"[完成] 共 {step} 步，{time.time() - t_start:.1f}s", flush=True)
    if step:
        print(f"[跟踪误差] 全程峰值 {err_peak:.4f} rad @ 步 {err_peak_step} "
              f"({err_peak_joint.replace(f'{args.agent_name}_', '').split('_joint')[0]})", flush=True)


if __name__ == "__main__":
    main()
