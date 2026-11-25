"""G1 右臂抓方块 —— 仿真闭环推理（按 LeRobot rollout/base 范式重写）。

对照 lerobot-rollout 的三个抽象（见 https://huggingface.co/docs/lerobot/inference）：

    Robot  : OrcaGymG1      get_observation() / send_action()
    Policy : RemotePolicy   select_action(obs) -> 动作块
    Loop   : run_rollout    30 Hz 定时循环，一次推理执行 N 帧

与官方 base 策略的同步推理（sync backend）一致：主循环阻塞在策略调用上，
拿到动作块后逐帧下发，每帧间隔 1/fps。

底层仍必须是力矩控制：G1 六个臂关节在 MuJoCo 里是 <motor>（ctrl 即力矩），
所以 send_action 内部做 PD + 重力补偿，绝不把角度写进 ctrl。

用法:
    python infer_g1_pick.py --host 192.168.110.51 --port 8000 --episode 0
"""

import argparse
import collections
import dataclasses
import importlib.util
import json
import os
import struct
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
IMAGE_SIZE = 224
SUBSTEPS = 33          # 33 × 1 ms = 33 ms，对齐数据集 fps=30
PHYSICS_DT = 0.001


# ── 配置 ─────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class RolloutConfig:
    host: str = "192.168.110.51"
    port: int = 8000
    agent_name: str = "g1_pick_usda"
    lerobot_root: str = "/home/user/pi/lerobot_g1_pick"
    norm_root: str | None = None     # 验证集不含 norm_stats 时指向训练集
    episode: int = 0                 # 取该集第 0 帧做初始姿态
    prompt: str = PROMPT
    cameras: str = "Camera,wrist_r"
    fps: float = 30.0
    execution_steps: int = 10        # 每块执行多少帧后重新推理
    blend_steps: int = 0             # 跨块平滑：0 = 基准（不混合）；>0 时只混前 7 个关节
    blend_alpha: float = 0.5         # 重叠区起始时旧块权重（线性衰减到 0）
    max_steps: int = 400
    orcagym_addr: str = "localhost:50051"
    task_config: str = "example.yaml"
    dump_dir: str | None = "./debug_images"
    trace: str | None = None         # 逐帧记录 q/动作/TCP/预测块 到 .npz，用于与示范对照
    capture_dir: str | None = None   # 首次"远处合爪"时保存该次推理的观测/输出（失败现场）
    capture_tcp_cm: float = 6.0      # 合爪时 TCP→方块 超过该距离才视为失败现场
    cam_stale_ms: int = 150          # 单路画面年龄上限（实测传输+解码约 70ms，30Hz 约 2 帧）
    cam_sync_ms: int = 60            # 两路相机彼此的采集时间差上限（约 2 帧）
    state_img_tol_ms: int = 5        # 画面捕获允许早于最后一次物理推进的毫秒数
    # 成功判定：方块完全落在盒口内、静止、且连续稳定
    success_radius: float = 0.10     # 找不到盒壁几何时的水平回退判据（m）
    success_floor_tol: float = 0.01   # 方块底面允许高出盒腔内底的高度（m）
    success_speed: float = 0.03      # 视为静止的最大线速度（m/s）
    success_hold: int = 30           # 连续满足多少控制步才算稳定（30 步 = 1s）
    # 控制参数
    wn: float = 60.0
    zeta: float = 1.0
    max_tau: float = 25.0
    max_tau_wrist: float = 5.0
    kp: float = 300.0
    kd: float = 20.0
    auto_gains: bool = True
    gravity_comp: bool = True
    grip_range: tuple[float, float] = (-0.020, 0.0245)
    settle_frames: int = 15


# ── 工具 ─────────────────────────────────────────────────────────────────
def load_collection_helpers():
    """复用采集脚本里的相机解析 / 求解器调参 / 关节钉住。"""
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "g1_dex1_collection_tele.py")
    spec = importlib.util.spec_from_file_location("g1_collect_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_norm_stats(lerobot_root: str) -> dict[str, np.ndarray]:
    path = Path(lerobot_root) / "norm_stats.json"
    if not path.is_file():
        raise SystemExit(f"缺少归一化统计: {path}")
    d = json.loads(path.read_text())["norm_stats"]
    return {
        "q01": np.asarray(d["actions"]["q01"], dtype=np.float64),
        "q99": np.asarray(d["actions"]["q99"], dtype=np.float64),
    }


def dataset_grip_to_sim(q_ds: float, ns: dict, grip_range: tuple[float, float]) -> float:
    """数据集夹爪值（m）→ 仿真机械行程目标角。

    初始化和运行时必须共用这一个映射，否则夹爪起点与模型看到的输入不一致。
    """
    q01 = float(ns["q01"][ARM_DIM])
    q99 = float(ns["q99"][ARM_DIM])
    a = float(np.clip((q_ds - q01) / max(q99 - q01, 1e-6), 0.0, 1.0))
    return grip_range[0] + a * (grip_range[1] - grip_range[0])


def resize_with_pad(bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """等比缩放 + 居中补黑边，与训练侧预处理一致（不拉伸）。"""
    h, w = bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[(size - nh) // 2 : (size - nh) // 2 + nh, (size - nw) // 2 : (size - nw) // 2 + nw] = resized
    return out


def _annexb_scan(payload: bytes) -> tuple[int, bool]:
    """统计 Annex-B 里的 VCL slice 数，并检查是否含 B 帧 slice。

    H.264 只有 B 帧会让「解码顺序 ≠ 显示顺序」；没有 B 帧时解码输出顺序就是
    采集顺序，FIFO 配帧才成立。B 帧检测是这条假设能否成立的关键证据。
    """
    n_vcl, has_b, i = 0, False, payload.find(b"\x00\x00\x01")
    while i >= 0 and i + 4 < len(payload):
        nal_type = payload[i + 3] & 0x1F
        if nal_type in (1, 5):          # 非 IDR / IDR slice
            n_vcl += 1
            if nal_type == 1 and _slice_is_b(payload[i + 4 : i + 12]):
                has_b = True
        i = payload.find(b"\x00\x00\x01", i + 3)
    return n_vcl, has_b


def _slice_is_b(head: bytes) -> bool:
    """从 slice header 开头读 first_mb_in_slice / slice_type（Exp-Golomb）。"""
    rbsp, zeros = bytearray(), 0
    for b in head:                      # 去掉防竞争字节 00 00 03
        if zeros >= 2 and b == 0x03:
            zeros = 0
            continue
        rbsp.append(b)
        zeros = zeros + 1 if b == 0 else 0
    bits = [(byte >> (7 - k)) & 1 for byte in rbsp for k in range(8)]
    pos, val = 0, 0
    for _ in range(2):                  # first_mb_in_slice, slice_type
        zeros = 0
        while pos < len(bits) and bits[pos] == 0:
            zeros += 1
            pos += 1
        if pos >= len(bits):
            return False
        pos += 1
        suffix = 0
        for _ in range(zeros):
            if pos >= len(bits):
                return False
            suffix = (suffix << 1) | bits[pos]
            pos += 1
        val = (1 << zeros) - 1 + suffix
    return val in (1, 6)               # 1/6 = B slice


class CameraStream:
    """从 OrcaStudio 的 WebSocket H.264 流持续解码（后台线程）。

    帧格式: [8B 时间戳 uint64 LE][4B simulate_index int32 LE][H.264 NAL]
    （见 orca_gym/recorder/camera_recorder.py::_process_frame）
    """

    HEADER = 12  # 8B timestamp + 4B simulate_index
    MAX_PENDING = 1  # 解码器最多只能延迟 1 条消息；不存在重排时空闲积压恒为 0

    def __init__(self, env, camera_name: str, label: str, stale_s: float):
        self.label = label
        rec = env.get_recorder_manager().get_recorder(camera_name)
        if rec is None:
            raise RuntimeError(f"相机 {label} 没有录制器")
        self._uri = f"ws://localhost:{int(rec.color_port)}"
        self._lock = threading.Lock()
        # 画面 / 帧计数 / 采集时间戳必须整组原子读取，否则可能把这一帧的画面
        # 和下一帧的时间戳配在一起
        self._snap: tuple[np.ndarray | None, int, float | None] = (None, 0, None)
        self.last_sim = None          # 最新一帧的 simulate_index（推理模式下恒 -1）
        self.last_ts_epoch = None     # 帧头采集时间戳（仅当与本地墙钟同源时有效）
        self._last_raw_ts = None
        self._ts_scale = None         # 时间戳单位（ns/us/ms/s，按帧间隔自动推断）
        self.dropped = 0              # 因积压被丢弃的帧数
        # 消息 ↔ 解码帧的对应关系统计：用来确认"一条消息 = 一帧"是否成立
        self.msgs = 0                 # 收到的 WS 消息数
        self.msgs_no_frame = 0        # 未解出帧的消息数（解码器延迟输出）
        self.msgs_multi_frame = 0     # 一条消息解出多帧的次数（重排/积压回吐）
        # 尚未对上解码帧的消息时间戳（FIFO）：解码器延迟输出时按到达顺序配帧
        self._pending_ts: collections.deque[int] = collections.deque()
        # FIFO 配对只在"消息↔帧一对一"时才成立；一旦违反就置位并拒绝使用画面
        self.frames_total = 0
        self.pair_error: str | None = None
        self.clock_resets = 0
        self.vcl_hist: dict[int, int] = {}   # 每条消息的 VCL slice 数分布
        self.b_slices = 0                    # 含 B 帧 slice 的消息数
        # 扫描结果与解码结果的一致性统计
        self.msg_vcl_no_frame = 0            # 有 slice 但本次没出帧（解码延迟）
        self.msg_frame_no_vcl = 0            # 没有 slice 却出帧（扫描/解码不一致）
        self.frames_no_ts = 0                # 出了帧但找不到对应时间戳（已丢弃）
        self.stale_s = stale_s
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        import av
        import websockets.sync.client

        while self._running:
            try:
                with websockets.sync.client.connect(self._uri, compression=None, max_size=None) as ws:
                    # 重连后旧连接的待配对时间戳、时钟推断全部作废，
                    # 否则相机时间戳重置时旧状态会让新帧一直被当成过期帧丢弃
                    if self.msgs:
                        self._reset_clock_state("断线重连")
                    # 每条消息是一帧完整 H.264 NAL，必须用持久的 CodecContext
                    # 增量解码以保持参考帧状态（参见 camera_recorder.py::_decode_nal）
                    codec = av.CodecContext.create("h264", "r")
                    while self._running:
                        data = ws.recv()
                        if isinstance(data, str) or len(data) < self.HEADER:
                            continue
                        try:
                            raw_ts = struct.unpack_from("<Q", data, 0)[0]
                            sim_idx = struct.unpack_from("<i", data, 8)[0]
                            payload = data[self.HEADER :]
                            decoded = list(codec.decode(av.Packet(payload)))
                            self.msgs += 1
                            # 逐条消息检查"是否只有一个画面单元、是否含 B 帧"：
                            # 没有 B 帧才意味着解码输出顺序=采集顺序，FIFO 配帧成立
                            n_vcl, has_b = _annexb_scan(payload)
                            self.vcl_hist[n_vcl] = self.vcl_hist.get(n_vcl, 0) + 1
                            if has_b:
                                self.b_slices += 1
                                self.pair_error = (
                                    "流中含 B 帧 slice：解码顺序可能≠采集顺序，"
                                    "按消息顺序 FIFO 配帧不成立"
                                )
                            elif n_vcl > 1 and self.pair_error is None:
                                self.pair_error = (
                                    f"一条消息含 {n_vcl} 个 VCL slice，"
                                    f"无法确认它是单个完整画面单元"
                                )
                            # 只有确认含画面 slice 的消息才把时间戳入队：
                            # 纯 SPS/PPS 等非画面消息若也入队，后续画面会取走它的时间戳
                            if n_vcl > 0:
                                self._pending_ts.append(raw_ts)
                            if not decoded:
                                self.msgs_no_frame += 1
                                if n_vcl > 0:
                                    self.msg_vcl_no_frame += 1
                                # 堆积超过 MAX_PENDING 说明解码器在攒帧，FIFO 配对不可靠
                                if len(self._pending_ts) > self.MAX_PENDING:
                                    self.pair_error = (
                                        f"连续 {len(self._pending_ts)} 条画面消息未解出帧"
                                        f"（解码器延迟输出，消息与帧无法一一对应）"
                                    )
                                continue
                            if len(decoded) > 1:
                                self.msgs_multi_frame += 1
                                self.pair_error = (
                                    f"一条消息解出 {len(decoded)} 帧"
                                    f"（消息与帧非一对一，时间戳无法可靠配对）"
                                )
                            if n_vcl == 0 and not self._pending_ts:
                                # 本条没有画面 slice，队列里也没有上一条消息留下的时间戳：
                                # 既说不出这帧是哪次采集的，也不符合"一消息一画面"，
                                # 才判为扫描/解码不一致。若队列非空，说明是解码延迟吐帧，
                                # FIFO 能正确配对，不算错。
                                self.msg_frame_no_vcl += 1
                                self.pair_error = (
                                    "扫描到 0 个 VCL slice 却解出图像帧，且没有待配时间戳："
                                    "Annex-B 扫描与解码结果不一致，时间戳无法配对"
                                )
                            for frame in decoded:
                                if not self._pending_ts:
                                    # 没有可信时间戳就不入库，绝不用当前消息的时间凑
                                    self.frames_no_ts += 1
                                    continue
                                frame_ts = self._pending_ts.popleft()
                                now = time.time()
                                self._update_clock(frame_ts, now)
                                # 采集时刻无法确认的帧不入库：到达时间不能当采集时间用
                                if self.last_ts_epoch is None:
                                    continue
                                # 丢弃积压帧：采集时间戳已过期就不更新画面，
                                # 让上层按"画面年龄"显式失败而不是拿旧图凑合
                                if now - self.last_ts_epoch > self.stale_s:
                                    self.dropped += 1
                                    continue
                                img = frame.to_ndarray(format="bgr24")
                                with self._lock:
                                    self._snap = (img, self._snap[1] + 1, self.last_ts_epoch)
                                    self.frames_total += 1
                                    self.last_sim = sim_idx
                        except av.FFmpegError:
                            continue
            except Exception as exc:
                if not self._running:
                    break
                print(f"[相机 {self.label}] 流断开: {exc}，1s 后重连", flush=True)
                time.sleep(1.0)

    def _update_clock(self, raw_ts: int, now: float):
        """解析帧头时间戳：先按帧间隔推断单位，再判断能否与本地墙钟对齐。"""
        if self._last_raw_ts is not None and raw_ts < self._last_raw_ts:
            # 时间戳回退=相机时钟被重置：旧推断与新帧不同源，必须整体作废，
            # 否则新帧会被算成"未来的帧"或永远过期
            self._reset_clock_state("时间戳回退/时钟重置")
            return
        if self._last_raw_ts is not None and raw_ts > self._last_raw_ts:
            delta = raw_ts - self._last_raw_ts
            for scale in (1e-9, 1e-6, 1e-3, 1.0):
                if 0.001 < delta * scale < 2.0:   # 帧间隔落在 1ms~2s 即认为单位吻合
                    self._ts_scale = scale
                    break
        self._last_raw_ts = raw_ts
        if self._ts_scale is None:
            return
        ts = raw_ts * self._ts_scale
        # 同源时 now - ts 应是一个毫秒级小量；否则（如 monotonic 计）不可比
        self.last_ts_epoch = ts if -1.0 < now - ts < 300.0 else None

    def _reset_clock_state(self, reason: str):
        """断线重连 / 时钟重置：作废旧画面、清空待配对时间戳、重新推断时钟。"""
        with self._lock:
            self._snap = (None, self._snap[1], None)
        self._pending_ts.clear()
        self._last_raw_ts = None
        self._ts_scale = None
        self.last_ts_epoch = None
        self.clock_resets += 1
        if self.msgs:
            print(f"[相机 {self.label}] {reason}：已清空待配对时间戳并重新推断时钟", flush=True)

    @property
    def pairing_ok(self) -> bool:
        return self.pair_error is None

    @property
    def pending(self) -> int:
        """还没配上解码帧的消息时间戳数量（正常应 ≤ MAX_PENDING）。"""
        return len(self._pending_ts)

    def snapshot(self) -> tuple[np.ndarray | None, int, float | None]:
        """原子读取 (画面, 帧计数, 采集时间戳)——三者一定来自同一帧。"""
        with self._lock:
            return self._snap

    @property
    def has_capture_clock(self) -> bool:
        """帧头时间戳是否与本地墙钟同源（否则无法确认图像采集时刻）。"""
        return self.snapshot()[2] is not None

    def stop(self):
        self._running = False


# ── Robot ────────────────────────────────────────────────────────────────
class OrcaGymG1:
    """OrcaGym 里的 G1：提供 get_observation / send_action 两个口子。

    内部负责：
      · 建立 SceneManager + DataCollectionManager
      · 初始姿态对齐（数据集第 0 帧）
      · 角度 → 力矩（PD + 重力补偿）与夹爪阻抗
      · 两路相机取流
    """

    def __init__(self, cfg: RolloutConfig):
        self.cfg = cfg
        self.ns = load_norm_stats(cfg.norm_root or cfg.lerobot_root)
        self.helpers = load_collection_helpers()
        self._setup_data()
        self._setup_env()
        self._setup_control()
        self._setup_cameras()

    # ── 数据 ──
    def _setup_data(self):
        parquet = Path(self.cfg.lerobot_root) / "data" / "chunk-000" / f"episode_{self.cfg.episode:06d}.parquet"
        table = pq.read_table(parquet)
        states = np.stack(table.column("observation.state").to_pylist())
        actions = np.stack(table.column("action").to_pylist())
        # ── 数据集维度自检：9 维、行数一致、字段顺序与 conf 逐项一致 ──
        for name, arr in (("observation.state", states), ("action", actions)):
            if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
                raise SystemExit(f"数据集 {name} 形状 {arr.shape}，期望 (N, {ACTION_DIM})")
        if states.shape[0] != actions.shape[0]:
            raise SystemExit(f"数据集 state/action 行数不一致: {states.shape[0]} vs {actions.shape[0]}")
        expect = (list(g1_dex1_osc_conf.r_arm["joint_names"])
                  + list(g1_dex1_osc_conf.gripper_r["joint_names"]))
        if len(expect) != ACTION_DIM:
            raise SystemExit(f"conf 右手关节共 {len(expect)} 个，与 ACTION_DIM={ACTION_DIM} 不一致")
        meta_path = Path(self.cfg.lerobot_root) / "meta" / "info.json"
        if not meta_path.is_file():
            raise SystemExit(f"缺少 {meta_path}，无法核对字段顺序")
        meta = json.loads(meta_path.read_text())
        for key in ("observation.state", "action"):
            names = list(meta["features"][key]["names"])
            if names != expect:
                raise SystemExit(
                    f"数据集 {key} 字段顺序与 conf 不一致:\n  数据 {names}\n  期望 {expect}"
                )
        print(f"[数据] {parquet.name} state={states.shape} action={actions.shape}，"
              f"{ACTION_DIM} 维字段顺序与 conf r_arm+gripper_r 逐项一致", flush=True)
        self.states9 = states
        self.demo_actions9 = actions
        self.parquet = parquet

    # ── 仿真 ──
    def _setup_env(self):
        default_joint_values = {
            name: 0.0
            for name in g1_dex1_osc_conf.l_arm["joint_names"] + g1_dex1_osc_conf.r_arm["joint_names"]
        }
        from yaml import Loader, load

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), self.cfg.task_config),
                  encoding="utf-8") as f:
            config = load(f, Loader=Loader)
        if "data_collection" in config:
            config["data_collection"]["agent_joint_prefix"] = f"{self.cfg.agent_name}_"
        else:
            config["data_collection"] = {"agent_joint_prefix": f"{self.cfg.agent_name}_"}

        scene_manager = SceneManager(self.cfg.orcagym_addr, config=config)
        self.manager = DataCollectionManager(
            agent_name=self.cfg.agent_name,
            env_name="Infer",
            entry_point=ENTRY_POINT,
            default_joint_values=default_joint_values,
            obs_callback=_obs_callback,
            env_index=0,
            frame_skip=1,
            time_step=PHYSICS_DT,
            orcagym_addr=self.cfg.orcagym_addr,
            scene_manager=scene_manager,
        )
        self.env = self.manager.env

        self.arm_actuators = g1_dex1_osc_conf.l_arm["motors_names"] + g1_dex1_osc_conf.r_arm["motors_names"]
        self.gripper_actuators = (
            g1_dex1_osc_conf.gripper_l["actuator_names"] + g1_dex1_osc_conf.gripper_r["actuator_names"]
        )
        self.arm_ids = [self.env.model.actuator_name2id(self.env.actuator(n)) for n in self.arm_actuators]
        self.gripper_ids = [self.env.model.actuator_name2id(self.env.actuator(n)) for n in self.gripper_actuators]
        self.arm_joint_names = [
            self.env.joint(n)
            for n in g1_dex1_osc_conf.l_arm["joint_names"] + g1_dex1_osc_conf.r_arm["joint_names"]
        ]
        self.right_joint_names = self.arm_joint_names[-ARM_DIM:]
        self.grip_joint_names = [self.env.joint(n) for n in g1_dex1_osc_conf.gripper_r["joint_names"]]

        self.env.reset()
        time.sleep(0.1)
        self.manager.update_scene()
        self.helpers.apply_grasp_solver_tuning(self.env)
        self.helpers.pin_all_joints(self.env, self.cfg.agent_name)

        # 初始姿态：把数据集第 0 帧的 9 维 state 写进仿真。
        # 夹爪两维先经与运行时相同的映射，再写入关节。
        raw9 = self.states9[0].astype(np.float64)
        target9 = raw9.copy()
        for i in range(ACTION_DIM - ARM_DIM):
            target9[ARM_DIM + i] = dataset_grip_to_sim(raw9[ARM_DIM + i], self.ns, self.cfg.grip_range)
        want = {n: np.asarray([target9[i]]) for i, n in enumerate(self.right_joint_names)}
        want.update({
            n: np.asarray([target9[ARM_DIM + i]]) for i, n in enumerate(self.grip_joint_names)
        })
        self.env.set_joint_qpos(want)
        self.env.mj_forward()
        got = self.read_state()
        print(f"[初始姿态] 数据集 {np.round(raw9, 4).tolist()}", flush=True)
        print(f"[初始姿态] 写入仿真 {np.round(target9, 4).tolist()}", flush=True)
        print(f"[初始姿态] 实际 {np.round(got, 4).tolist()}  "
              f"最大偏差 {np.abs(got - target9).max():.6f}", flush=True)

    # ── 控制 ──
    def _setup_control(self):
        import mujoco

        cfg = self.cfg
        gym = getattr(self.env, "gym", None) or getattr(getattr(self.env, "unwrapped", self.env), "gym", None)
        self._mj, self._md = gym._mjModel, gym._mjData

        self.tau_limit = np.array(
            [cfg.max_tau] * 5 + [cfg.max_tau_wrist] * 2 + [cfg.max_tau] * 5 + [cfg.max_tau_wrist] * 2,
            dtype=np.float64,
        )
        self.kp_vec = np.full(14, cfg.kp, dtype=np.float64)
        self.kd_vec = np.full(14, cfg.kd, dtype=np.float64)
        self.dof_adr = [
            int(self._mj.jnt_dofadr[mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in self.arm_joint_names
        ]
        if cfg.auto_gains:
            # kp = J·wn², kd = 2ζJ·wn，并钳位 kd·dt/J ≤ 1 保证离散稳定
            scratch = mujoco.MjData(self._mj)
            scratch.qpos[:] = self._md.qpos
            mujoco.mj_forward(self._mj, scratch)
            mass = np.zeros((self._mj.nv, self._mj.nv), dtype=np.float64)
            for col in range(self._mj.nv):
                unit = np.zeros(self._mj.nv, dtype=np.float64)
                unit[col] = 1.0
                res = np.zeros(self._mj.nv, dtype=np.float64)
                mujoco.mj_mulM(self._mj, scratch, res, unit)
                mass[:, col] = res
            joint_j = np.array([mass[d, d] for d in self.dof_adr])
            self.kp_vec = joint_j * cfg.wn**2
            self.kd_vec = np.minimum(2.0 * cfg.zeta * joint_j * cfg.wn, joint_j / PHYSICS_DT)
            print(f"[增益] wn={cfg.wn:g} zeta={cfg.zeta:g}，右臂 7 关节", flush=True)
            for k in range(ARM_DIM, 14):
                print(f"    {self.arm_joint_names[k].split('_', 1)[-1]:26s} J={joint_j[k]:.5f}"
                      f" kp={self.kp_vec[k]:8.2f} kd={self.kd_vec[k]:6.3f}", flush=True)

        self.grip_kp = float(g1_dex1_osc_conf.gripper_r["kp"])
        self.grip_kd = float(g1_dex1_osc_conf.gripper_r["kd"])
        self.grip_max = float(g1_dex1_osc_conf.gripper_r["max_force"])
        self.step_wall = 0.0
        self.g_q01 = float(self.ns["q01"][ARM_DIM])
        self.g_q99 = float(self.ns["q99"][ARM_DIM])
        self.grip_open, self.grip_close = cfg.grip_range
        # 左夹爪：仿真行程的开爪角（机械量，不经过数据集统计映射）
        self.left_grip_joint_names = [
            self.env.joint(n) for n in g1_dex1_osc_conf.gripper_l["joint_names"]
        ]
        self.left_grip_open = float(g1_dex1_osc_conf.gripper_l["q_open"])

        # 左臂保持复位，避免甩动改变俯视相机画面（数据里只有右臂）
        q_left = self.env.query_joint_qpos(self.arm_joint_names[:ARM_DIM])
        self.left_arm_hold = np.array(
            [float(np.asarray(q_left[n]).ravel()[0]) for n in self.arm_joint_names[:ARM_DIM]],
            dtype=np.float64,
        )

        # 场景物体句柄
        mj = self._mj
        self.cube_bid = next(
            (i for i in range(mj.nbody)
             if "cube" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()),
            -1,
        )
        self.box_bids = [
            i for i in range(mj.nbody)
            if "box" in (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
        ]
        self.ee_sid = self.env.site(g1_dex1_osc_conf.r_arm["ee_site_name"])

        def body_name(bid: int) -> str:
            return mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, bid) or f"#{bid}"

        self.box_bid = self.box_bids[0] if self.box_bids else -1
        print(f"[场景] 方块 body={body_name(self.cube_bid) if self.cube_bid >= 0 else '未找到'}"
              f"  盒子 body={[body_name(b) for b in self.box_bids]}", flush=True)
        for bid in [b for b in [self.cube_bid, *self.box_bids] if b >= 0]:
            xyz = np.round(np.asarray(self._md.xpos[bid], dtype=np.float64), 4)
            print(f"[场景]   {body_name(bid)} xyz={xyz.tolist()}", flush=True)
        self.box_z_range = self._body_z_range(self.box_bid) if self.box_bid >= 0 else None
        self.box_floor_z = self._box_floor_z(self.box_bid) if self.box_bid >= 0 else None
        cube_z_range = self._body_z_range(self.cube_bid) if self.cube_bid >= 0 else None
        self.cube_z_half = ((cube_z_range[1] - cube_z_range[0]) / 2.0
                            if cube_z_range is not None else 0.0)
        if self.box_z_range is not None:
            print(f"[场景] 盒子竖直范围 [{self.box_z_range[0]:.3f}, {self.box_z_range[1]:.3f}]m"
                  f"  盒腔内底 z={self.box_floor_z:.3f}m"
                  f"  方块半高 {self.cube_z_half:.3f}m", flush=True)
        if self.box_floor_z is not None:
            print(f"[场景] 成功要求：方块底部落在 {self.box_floor_z:.3f}~"
                  f"{self.box_floor_z + self.cfg.success_floor_tol:.3f}m 内（实际落进盒腔），"
                  f"而不是停在盒沿上", flush=True)
        self.cube_half_xy, _ = (self._body_xy_extents(self.cube_bid)
                                if self.cube_bid >= 0 else ((0.0, 0.0), (0.0, 0.0)))
        self.box_half_xy, self.box_inner_xy = (self._body_xy_extents(self.box_bid)
                                               if self.box_bid >= 0 else (None, None))
        if self.box_inner_xy is not None:
            print(f"[场景] 盒口内半径 x={self.box_inner_xy[0]:.3f} y={self.box_inner_xy[1]:.3f}m"
                  f"，方块半宽 x={self.cube_half_xy[0]:.3f} y={self.cube_half_xy[1]:.3f}m"
                  f"（完全落入要求 |Δx|≤{self.box_inner_xy[0] - self.cube_half_xy[0]:.3f}"
                  f" |Δy|≤{self.box_inner_xy[1] - self.cube_half_xy[1]:.3f}m）", flush=True)

    def _body_xy_extents(self, bid: int):
        """body 及其子 body 的水平 (外轮廓半宽, 内开口半宽)。

        盒壁按"薄壁 geom"识别（水平最小半厚 < 3cm）：内表面 = |壁心偏移| − 半厚，
        这样开口尺寸直接来自场景几何，不依赖硬编码。
        """
        mj, md = self._mj, self._md
        cx, cy = float(md.xpos[bid][0]), float(md.xpos[bid][1])
        half = [0.0, 0.0]
        walls = [np.inf, np.inf]
        for g in range(mj.ngeom):
            b = int(mj.geom_bodyid[g])
            while b > 0 and b != bid:
                b = int(mj.body_parentid[b])
            if b != bid:
                continue
            hx, hy = float(mj.geom_size[g][0]), float(mj.geom_size[g][1])
            dx = abs(float(md.geom_xpos[g][0]) - cx)
            dy = abs(float(md.geom_xpos[g][1]) - cy)
            half[0], half[1] = max(half[0], dx + hx), max(half[1], dy + hy)
            if min(hx, hy) < 0.03:
                if hx < hy:
                    walls[0] = min(walls[0], dx - hx)
                else:
                    walls[1] = min(walls[1], dy - hy)
        inner = [walls[i] if np.isfinite(walls[i]) else half[i] for i in range(2)]
        return tuple(half), tuple(inner)

    def _box_floor_z(self, bid: int):
        """盒腔内底面高度：盒体几何里"宽底"(非薄壁)的最高上表面。

        方块必须落在这个面上才算进盒腔；只按盒子整体高度范围判断的话，
        方块静止在盒沿上也会被误判为成功。
        """
        import mujoco

        mj, md = self._mj, self._md
        best = None
        for g in range(mj.ngeom):
            b = int(mj.geom_bodyid[g])
            while b > 0 and b != bid:
                b = int(mj.body_parentid[b])
            if b != bid:
                continue
            hx, hy = float(mj.geom_size[g][0]), float(mj.geom_size[g][1])
            if min(hx, hy) < 0.03:  # 薄壁=盒壁，不是盒底
                continue
            if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                hz = float(mj.geom_size[g][2])
            else:
                hz = float(mj.geom_rbound[g])
            top = float(md.geom_xpos[g][2]) + hz
            best = top if best is None else max(best, top)
        return best

    def _body_z_range(self, bid: int):
        """body 及其子 body 的几何竖直范围（按 geom 类型的 z 半长，忽略几何自身旋转）。"""
        import mujoco

        mj, md = self._mj, self._md
        lo, hi = np.inf, -np.inf
        for g in range(mj.ngeom):
            b = int(mj.geom_bodyid[g])
            while b > 0 and b != bid:
                b = int(mj.body_parentid[b])
            if b != bid:
                continue
            z = float(md.geom_xpos[g][2])
            if mj.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                hz = float(mj.geom_size[g][2])
            else:
                hz = float(mj.geom_rbound[g])
            lo, hi = min(lo, z - hz), max(hi, z + hz)
        return (lo, hi) if lo < hi else None

    def _read_arm(self):
        q = self.env.query_joint_qpos(self.arm_joint_names)
        dq = self.env.query_joint_qvel(self.arm_joint_names)
        return (
            np.array([float(np.asarray(q[n]).ravel()[0]) for n in self.arm_joint_names]),
            np.array([float(np.asarray(dq[n]).ravel()[0]) for n in self.arm_joint_names]),
        )

    def read_state(self) -> np.ndarray:
        q_arm = self.env.query_joint_qpos(self.right_joint_names)
        q_grip = self.env.query_joint_qpos(self.grip_joint_names)
        return np.array(
            [float(np.asarray(q_arm[n]).ravel()[0]) for n in self.right_joint_names]
            + [float(np.asarray(q_grip[n]).ravel()[0]) for n in self.grip_joint_names],
            dtype=np.float64,
        )

    def _arm_torque(self, target14: np.ndarray) -> np.ndarray:
        q, dq = self._read_arm()
        tau = self.kp_vec * (target14 - q) - self.kd_vec * dq
        if self.cfg.gravity_comp:
            tau = tau + np.asarray(self._md.qfrc_bias, dtype=np.float64)[self.dof_adr]
        return np.clip(tau, -self.tau_limit, self.tau_limit)

    def dataset_grip_to_sim(self, q_ds: float) -> float:
        """数据集夹爪值 → 仿真机械行程目标角（与初始化共用同一函数）。"""
        return dataset_grip_to_sim(q_ds, self.ns, self.cfg.grip_range)

    def grip_force_for(self, joint_names, q_sim: float) -> float:
        """给定仿真目标角，按 DEX1 位置阻抗律算 motor 力，饱和 ±max_force。

        输入必须是仿真行程角（机械量）；数据集值请先过 dataset_grip_to_sim。
        """
        q = self.env.query_joint_qpos([joint_names[0]])[joint_names[0]]
        dq = self.env.query_joint_qvel([joint_names[0]])[joint_names[0]]
        f = self.grip_kp * (q_sim - float(np.asarray(q).ravel()[0])) \
            - self.grip_kd * float(np.asarray(dq).ravel()[0])
        return float(np.clip(f, -self.grip_max, self.grip_max))

    # ── 相机 ──
    def _setup_cameras(self):
        logical = [c.strip() for c in self.cfg.cameras.split(",") if c.strip()]
        resolved = self.helpers._resolve_cameras(self.env, logical)
        self.cameras = {}
        for key, lg in zip(("base_0_rgb", "right_wrist_0_rgb"), logical):
            cam = resolved[lg]
            try:
                self.env.start_streaming(cam, capture_rgb=True)
            except ValueError:
                self.env.start_streaming(
                    cam, capture_rgb=True,
                    color_port=self.helpers._CAMERA_FALLBACK_PORTS.get(lg, 7070),
                )
            self.cameras[key] = CameraStream(self.env, cam, key, self.cfg.cam_stale_ms / 1000.0)
            print(f"[相机] {key} ← {cam}", flush=True)
        deadline = time.time() + 8.0
        while time.time() < deadline and any(c.snapshot()[0] is None for c in self.cameras.values()):
            self.env.render()
            time.sleep(0.05)
        missing = [k for k, c in self.cameras.items() if c.snapshot()[0] is None]
        if missing:
            raise RuntimeError(f"相机无帧: {missing}")
        # 无法确认采集时刻就等于无法判断画面是否对应本次 state——直接拒绝运行
        no_clock = [k for k, c in self.cameras.items() if not c.has_capture_clock]
        if no_clock:
            raise RuntimeError(
                f"无法确认图像采集时刻: {no_clock} 的帧头时间戳与本地墙钟不同源"
                f"（到达时间不能当采集时间用），图像-状态同步判据失效"
            )
        # 消息↔帧一对一自检：只有一对一，FIFO 时间戳配对才成立
        for key, c in self.cameras.items():
            if not c.pairing_ok:
                raise RuntimeError(f"相机 {key} 时间戳配对不可靠: {c.pair_error}")
            print(f"[相机] {key} 消息/帧自检：消息{c.msgs} 已解码帧{c.frames_total} "
                  f"待配对时间戳{c.pending} 每消息VCL{c.vcl_hist} B帧{c.b_slices}"
                  f" 扫描一致性[VCL无帧{c.msg_vcl_no_frame} 帧无VCL{c.msg_frame_no_vcl}"
                  f" 无戳帧{c.frames_no_ts}]"
                  f"（待配对须 ≤{CameraStream.MAX_PENDING}，多帧必须为 0）", flush=True)
        self._anchor: dict[str, int] | None = None

    # ── 对外接口（对齐 LeRobot Robot）──
    def set_camera_anchor(self):
        """推理循环开始前的相机同步准备。

        实测两个事实（2026-11-19）：
          1) 纯推理（非录制）模式下引擎发的 simulate_index 恒为 -1，
             按仿真帧号配帧不可行；
          2) 相机流有自身帧率上限，控制循环快于实时时会丢帧
             （10 次 render 只收到 5 帧的实测），按收帧计数硬配同样不可行。
        因此改用"新鲜度"同步：每次取观测短等一帧新画面，用画面年龄
        （帧解码时刻距 state 读取时刻）作为同步判据，超过阈值直接报错。
        """
        self._prev_count = {k: max(0, c.snapshot()[1] - 1) for k, c in self.cameras.items()}
        sim_ok = all(c.last_sim is not None and c.last_sim >= 0 for c in self.cameras.values())
        self.cam_age_ms: dict[str, int] = {}
        self.cam_sync_ms = float("nan")
        self.state_img_ms = float("nan")
        self.img_pre_step_ms = float("nan")
        print(f"[相机] simulate_index {'可用' if sim_ok else '恒为 -1（引擎未提供，无法按仿真帧号配帧）'}，"
              f"改用新鲜度同步（阈值 {self.cfg.cam_stale_ms}ms），起始帧号 {self._prev_count}", flush=True)
        for k, c in self.cameras.items():
            src = (f"帧头时间戳（单位 {c._ts_scale:g}s）" if c.has_capture_clock
                   else "无法确认图像采集时刻（帧头时间戳与墙钟不同源）")
            print(f"[相机]   {k} 时钟源={src}", flush=True)

    def _grab_images(self):
        """一次性抓两路画面，且必须是"最后一次物理推进之后"采集的那一帧。

        物理只在 send_action 里推进，所以：
          · state 在两次物理推进之间是冻结的，"图→state"的时间差不影响配对；
          · 但若用到的画面是在那次推进之前采集的，它显示的是上一个姿态，
            与本次 state 差一帧——必须等引擎把新画面送出来。
        两路同时等待，保证它们在时间上靠拢。
        """
        deadline = time.time() + 0.8   # 相机偶发 300~500ms 出帧停顿，等足再报失步
        prev = {k: s.snapshot()[1] for k, s in self.cameras.items()}
        last_render = 0.0
        while time.time() < deadline:
            snaps = {k: s.snapshot() for k, s in self.cameras.items()}
            if all(sn[1] > prev[k] and sn[2] is not None and sn[2] >= self.step_wall - 0.003
                   for k, sn in snaps.items()):
                break
            now = time.time()
            if now - last_render >= 0.03:   # 按 30Hz 补渲染，避免高频 render 堆积编码队列
                self.env.render()  # render 会让录制器再抓一帧
                last_render = now
            time.sleep(0.002)
        images, stamps, ages = {}, {}, {}
        now = time.time()
        for key, stream in self.cameras.items():
            # 画面、计数、时间戳一次性取，保证来自同一帧（年龄也用它算）
            frame, count, ts = stream.snapshot()
            if frame is None:
                raise RuntimeError(f"相机 {key} 无帧")
            if not stream.pairing_ok:
                raise RuntimeError(f"相机 {key} 时间戳配对不可靠: {stream.pair_error}")
            if ts is None:
                raise RuntimeError(f"无法确认 {key} 的图像采集时刻（帧头时间戳与墙钟不同源）")
            ages[key] = max(0.0, (now - ts) * 1000.0)
            if ages[key] > self.cfg.cam_stale_ms:
                raise RuntimeError(
                    f"相机 {key} 画面年龄 {ages[key]:.0f}ms 超过阈值 {self.cfg.cam_stale_ms}ms"
                    f"——图像与状态失步，停止下发动作"
                )
            self._prev_count[key] = count
            rgb = cv2.cvtColor(resize_with_pad(frame, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
            images[key] = np.transpose(rgb, (2, 0, 1))  # CHW uint8
            stamps[key] = ts
        return images, ages, stamps

    def get_observation(self, step: int = 0) -> dict:
        # 顺序很重要：先取图、再读 state。state 是瞬时量，放在取图之后读，
        # 才能保证配对的 state 绝不早于画面采集时刻（否则会"旧状态 + 新画面"）。
        images, ages, stamps = self._grab_images()
        t_state = time.time()
        obs = {"state": self.read_state().astype(np.float32), "images": images, "prompt": self.cfg.prompt}
        # 画面采集时刻 ↔ 本次 state 的间隔（画面来自取图阶段，state 在其后读取）
        self.state_img_ms = max((t_state - s) * 1000.0 for s in stamps.values()) if stamps else float("nan")
        # 物理只在 send_action 里推进，所以画面必须在上一次物理推进之后采集，
        # 否则画面显示的是更早的姿态，与本次 state 对不上。
        self.img_pre_step_ms = (
            (self.step_wall - min(stamps.values())) * 1000.0 if stamps else float("nan")
        )
        self.cam_age_ms = {k: int(v) for k, v in ages.items()}
        self.cam_sync_ms = (abs(stamps["base_0_rgb"] - stamps["right_wrist_0_rgb"]) * 1000.0
                            if len(stamps) == 2 else float("nan"))
        if len(stamps) == 2 and self.cam_sync_ms > self.cfg.cam_sync_ms:
            raise RuntimeError(
                f"两路相机时间差 {self.cam_sync_ms:.0f}ms 超过阈值 {self.cfg.cam_sync_ms}ms"
                f"——俯视与腕部画面不同步，停止下发动作"
            )
        if self.img_pre_step_ms > self.cfg.state_img_tol_ms:
            raise RuntimeError(
                f"画面比最后一次物理推进早 {self.img_pre_step_ms:.0f}ms"
                f"——图像对应旧姿态，与本次 state 无法配对"
            )
        return obs

    def _substeps(self, step: int) -> int:
        """30 Hz = 33.33 ms/步；33 子步 = 33 ms，每 3 步插 1 个 34 子步，
        平均 33.33 ms，长期零漂移。"""
        return SUBSTEPS + (1 if step % 3 == 2 else 0)

    def send_action(self, action9: np.ndarray, step: int = 0):
        """9 维绝对动作 → 力矩，推进一个控制步（33/34 子步交替）。"""
        target14 = np.concatenate([self.left_arm_hold, np.asarray(action9[:ARM_DIM], dtype=np.float64)])
        grip_target = float(np.mean(action9[ARM_DIM:]))
        ctrl = np.zeros(self.env.nu, dtype=np.float32)
        n_sub = self._substeps(step)
        for s in range(n_sub):
            ctrl[:] = 0.0
            ctrl[self.arm_ids] = self._arm_torque(target14)
            ctrl[self.gripper_ids] = [
                self.grip_force_for(self.left_grip_joint_names, self.left_grip_open),
                self.grip_force_for(self.grip_joint_names, self.dataset_grip_to_sim(grip_target)),
            ]
            self.env.step(ctrl)
            if s == n_sub - 1:
                self.step_wall = time.time()  # 物理推进完成时刻：画面必须在此之后采集
                self.env.render()

    def settle(self):
        init9 = self.states9[0]
        for _ in range(self.cfg.settle_frames):
            for _s in range(SUBSTEPS):
                ctrl = np.zeros(self.env.nu, dtype=np.float32)
                ctrl[self.arm_ids] = self._arm_torque(
                    np.concatenate([self.left_arm_hold, init9[:ARM_DIM]])
                )
                ctrl[self.gripper_ids] = [
                    self.grip_force_for(self.left_grip_joint_names, self.left_grip_open),
                    self.grip_force_for(self.grip_joint_names, self.dataset_grip_to_sim(float(init9[ARM_DIM]))),
                ]
                self.env.step(ctrl)
            self.step_wall = time.time()
            self.env.render()

    def disconnect(self):
        for stream in getattr(self, "cameras", {}).values():
            stream.stop()

    # ── 指标 ──
    def tcp_cube_cm(self) -> float:
        if self.cube_bid < 0:
            return float("nan")
        tcp = np.asarray(self.env.query_site_pos_and_quat([self.ee_sid])[self.ee_sid]["xpos"]).ravel()[:3]
        return float(np.linalg.norm(tcp - np.asarray(self._md.xpos[self.cube_bid], dtype=np.float64))) * 100.0

    def cube_height(self) -> float:
        return float(self._md.xpos[self.cube_bid][2]) if self.cube_bid >= 0 else float("nan")

    def cube_speed(self) -> float:
        """方块线速度（m/s，世界系）。"""
        if self.cube_bid < 0:
            return float("nan")
        return float(np.linalg.norm(np.asarray(self._md.cvel[self.cube_bid], dtype=np.float64)[3:]))

    def in_box(self) -> bool:
        """方块是否已放进盒子：完全落在盒口内、竖直在盒内、且基本静止。"""
        if self.cube_bid < 0 or self.box_bid < 0:
            return False
        cube = np.asarray(self._md.xpos[self.cube_bid], dtype=np.float64)
        box = np.asarray(self._md.xpos[self.box_bid], dtype=np.float64)
        dx, dy = abs(float(cube[0] - box[0])), abs(float(cube[1] - box[1]))
        if self.box_inner_xy is not None:
            # 方块整体（不是中心）必须在开口内，开口尺寸来自盒壁几何
            if (dx + self.cube_half_xy[0] > self.box_inner_xy[0]
                    or dy + self.cube_half_xy[1] > self.box_inner_xy[1]):
                return False
        elif float(np.hypot(dx, dy)) >= self.cfg.success_radius:
            return False
        # 竖直：方块底面必须贴在盒腔内底上。只判断中心高度或"高出上沿多少"的话，
        # 方块稳定停在盒沿上也会被算成功。
        z = float(cube[2])
        if self.box_floor_z is not None:
            bottom = z - self.cube_z_half
            if not (self.box_floor_z - 0.01 <= bottom
                    <= self.box_floor_z + self.cfg.success_floor_tol):
                return False
        elif self.box_z_range is not None:
            z_lo, z_hi = self.box_z_range
            if not (z_lo - 0.01) < z < z_hi:
                return False
        # 还在滑动/被拖动时不算放置完成
        return self.cube_speed() <= self.cfg.success_speed

    def cube_z_bounds(self) -> tuple[float, float]:
        """方块底面 / 顶面高度（用于日志对照盒底与盒沿）。"""
        if self.cube_bid < 0:
            return (float("nan"), float("nan"))
        z = float(self._md.xpos[self.cube_bid][2])
        return z - self.cube_z_half, z + self.cube_z_half

    def cube_box_dist_cm(self) -> float:
        if self.cube_bid < 0 or self.box_bid < 0:
            return float("nan")
        cube = np.asarray(self._md.xpos[self.cube_bid], dtype=np.float64)
        box = np.asarray(self._md.xpos[self.box_bid], dtype=np.float64)
        return float(np.linalg.norm(cube[:2] - box[:2])) * 100.0


# ── Policy ───────────────────────────────────────────────────────────────
class RemotePolicy:
    """openpi serve_policy 的 websocket 客户端（msgpack-numpy 协议）。"""

    def __init__(self, host: str, port: int):
        client_src = "/home/user/openpi/packages/openpi-client/src"
        if client_src not in sys.path:
            sys.path.insert(0, client_src)
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        print(f"[策略] 连接 ws://{host}:{port} ...", flush=True)
        self._policy = WebsocketClientPolicy(host=host, port=port)
        print(f"[策略] 已连接 metadata={self._policy.get_server_metadata()}", flush=True)

    def select_action(self, obs: dict) -> tuple[np.ndarray, float]:
        """返回（绝对动作块, 推理耗时秒）。

        服务端输出变换链是 G1PickCubeOutputs(裁回 9 维) → Unnormalize →
        AbsoluteActions，出来已是物理绝对角；这里不能再叠加 state
        （否则目标翻倍，跟踪误差会到 3 rad 量级）。
        """
        t0 = time.time()
        out = self._policy.infer(obs)
        latency = time.time() - t0
        chunk = np.array(out["actions"], dtype=np.float64)
        # action_dim=32 是模型内部维度，客户端必须收到裁剪后的 (T, 9)。
        # 形状不对（多 batch 维 / 未裁剪 / 空）一律拒绝下发。
        if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != ACTION_DIM:
            raise RuntimeError(
                f"服务端动作形状 {chunk.shape} 不符，期望 (T≥1, {ACTION_DIM})——拒绝下发"
            )
        return chunk, latency


# ── 主循环 ───────────────────────────────────────────────────────────────
class Cadence:
    """30 Hz 节拍统计，对齐 lerobot 的 cadence reporting。"""

    def __init__(self, fps: float):
        self.period = 1.0 / fps
        self.ticks = 0
        self.over = 0
        self.latencies: list[float] = []
        self.t_start = time.time()
        self._next = time.time()

    def note_latency(self, latency: float):
        """每块推理记录一次（不是每帧）。"""
        self.latencies.append(latency)

    def reset(self):
        """推理等阻塞调用结束后重置节拍基准。

        不重置的话 _next 还停在推理前的旧时间表上，tick 会不睡连发，
        把阻塞期欠的时间一次性"追赶"出去——动作会瞬间密集下发。
        """
        self._next = time.time()

    def tick(self):
        self.ticks += 1
        if time.time() > self._next:
            self.over += 1
        self._next += self.period
        sleep = self._next - time.time()
        if sleep > 0:
            time.sleep(sleep)

    def report(self):
        elapsed = time.time() - self.t_start
        lat = np.array(self.latencies) * 1000 if self.latencies else np.zeros(1)
        print(
            f"[节拍] {self.ticks / elapsed:5.2f} Hz（目标 {1 / self.period:.0f} Hz）"
            f" · {self.ticks} ticks / {elapsed:.1f}s"
            f" · 超预算 {self.over}/{self.ticks}"
            f" · 推理 {np.median(lat):.1f} ms（最坏 {lat.max():.1f}）",
            flush=True,
        )


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
    return {"/observation/joint/position": np.array(
        [qpos[name] for name in names], dtype=np.float32).flatten()}


def run_rollout(cfg: RolloutConfig):
    robot = OrcaGymG1(cfg)
    policy = RemotePolicy(cfg.host, cfg.port)
    cadence = Cadence(cfg.fps)
    grip_mid = 0.5 * (float(robot.ns["q01"][ARM_DIM]) + float(robot.ns["q99"][ARM_DIM]))

    print("=" * 66, flush=True)
    print(f"  数据集   {robot.parquet}", flush=True)
    print(f"  控制     {cfg.fps:.0f} Hz，每块执行 {cfg.execution_steps} 帧后重推理", flush=True)
    print("=" * 66, flush=True)

    robot.settle()
    robot.set_camera_anchor()
    step, first_close, err_peak, err_peak_joint = 0, None, 0.0, ""
    prev_chunk, prev_start = None, 0
    hold, success_step = 0, None
    tr = ({"step": [], "q7": [], "q9": [], "act9": [], "grip_sim": [], "tcp": [],
           "chunk_step": [], "chunk": []} if cfg.trace else None)
    captured = False
    try:
        while step < cfg.max_steps:
            obs = robot.get_observation(step)
            actions, latency = policy.select_action(obs)
            cadence.note_latency(latency)
            cadence.reset()  # 推理阻塞结束，重挂节拍基准，防止追赶旧时间表连发
            # 跨块平滑：本块第 i 步对应上一块第 shift+i 步，重叠区旧块权重线性衰减
            if prev_chunk is not None and cfg.blend_steps > 0:
                shift = step - prev_start
                k = min(cfg.blend_steps, len(actions), max(len(prev_chunk) - shift, 0))
                if k > 0:
                    w_old = np.linspace(cfg.blend_alpha, 0.0, k)[:, None]
                    # 只混前 7 个关节角；夹爪是离散语义（开/合），跨块平均会拖慢合爪
                    actions[:k, :ARM_DIM] = (
                        (1.0 - w_old) * actions[:k, :ARM_DIM]
                        + w_old * prev_chunk[shift : shift + k, :ARM_DIM]
                    )
            prev_chunk, prev_start = actions.copy(), step
            # 失败现场：本块在远处（TCP→方块 > 阈值）就给出合爪，保存这次推理的输入与输出
            if cfg.capture_dir and not captured:
                k_close = next((i for i, a in enumerate(actions[: cfg.execution_steps])
                                if float(a[ARM_DIM]) > grip_mid), None)
                if k_close is not None:
                    tcp_cm = robot.tcp_cube_cm()
                    if tcp_cm > cfg.capture_tcp_cm:
                        os.makedirs(cfg.capture_dir, exist_ok=True)
                        payload = {
                            "step": step, "state": obs["state"], "prompt": obs["prompt"],
                            "chunk": actions.copy(), "close_index": k_close,
                            "tcp_cm": tcp_cm, "sim_q": robot.read_state(),
                            "latency": latency,
                        }
                        for k, im in obs["images"].items():
                            payload[f"img_{k}"] = im
                        out = os.path.join(cfg.capture_dir, "fail_obs.npz")
                        np.savez(out, **payload)
                        captured = True
                        print(f"[捕获] 步 {step} 块内 t={k_close} 即合爪，TCP→方块={tcp_cm:.2f}cm，"
                              f"已保存 {out}", flush=True)
            if tr is not None:
                tr["chunk_step"].append(step)
                tr["chunk"].append(actions.copy())
            if step == 0 and cfg.dump_dir:
                os.makedirs(cfg.dump_dir, exist_ok=True)
                for key, img in obs["images"].items():
                    short = "base" if key == "base_0_rgb" else "wrist"
                    cv2.imwrite(
                        os.path.join(cfg.dump_dir, f"step_0000_{short}.png"),
                        cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR),
                    )
                print(f"[观测] step={obs['state'].round(4).tolist()} prompt={obs['prompt']!r}", flush=True)

            for action9 in actions[: cfg.execution_steps]:
                if step >= cfg.max_steps:
                    break
                robot.send_action(action9, step)
                step += 1

                if first_close is None and float(action9[ARM_DIM]) > grip_mid:
                    first_close = step
                    print(f"[事件] 步 {step} 首次合爪，TCP→方块={robot.tcp_cube_cm():.2f}cm", flush=True)

                q_now, _ = robot._read_arm()
                err = float(np.abs(q_now[ARM_DIM:] - action9[:ARM_DIM]).max())
                if tr is not None:
                    tr["step"].append(step)
                    tr["q7"].append(np.asarray(q_now[ARM_DIM:], dtype=np.float64))
                    tr["q9"].append(np.asarray(robot.read_state(), dtype=np.float64))
                    tr["act9"].append(np.asarray(action9, dtype=np.float64))
                    tr["grip_sim"].append(robot.dataset_grip_to_sim(
                        float(np.mean(action9[ARM_DIM:]))))
                    tr["tcp"].append(robot.tcp_cube_cm())
                if err > err_peak:
                    err_peak = err
                    err_peak_joint = robot.right_joint_names[int(
                        np.abs(q_now[ARM_DIM:] - action9[:ARM_DIM]).argmax())]

                # 成功判定：方块进入盒内并连续保持 success_hold 步
                if robot.in_box():
                    hold += 1
                    if hold >= cfg.success_hold and success_step is None:
                        success_step = step
                        zb, zt = robot.cube_z_bounds()
                        print(f"[成功] 步 {step} 方块在盒内已稳定 {hold} 步"
                              f"（{hold / cfg.fps:.2f}s），距盒心 {robot.cube_box_dist_cm():.1f}cm"
                              f"，方块底/顶={zb:.3f}/{zt:.3f}m"
                              f"（盒底 {robot.box_floor_z:.3f} / 盒沿 {robot.box_z_range[1]:.3f}m）",
                              flush=True)
                else:
                    hold = 0

                if step % 30 == 0:
                    age = " ".join(f"{k.split('_')[0]}{v}ms" for k, v in robot.cam_age_ms.items())
                    print(f"[循环] 步 {step:4d}  TCP→方块={robot.tcp_cube_cm():5.2f}cm"
                          f"  方块→盒={robot.cube_box_dist_cm():5.2f}cm"
                          f"  方块z={robot.cube_height():.3f}m"
                          f"  夹爪={float(np.mean(action9[ARM_DIM:])):+.4f}"
                          f"  跟踪误差={err:.4f}"
                          f"  画面龄[{age}]  同步{robot.cam_sync_ms:.0f}ms"
                          f"  图-状态{robot.state_img_ms:.0f}ms"
                          f"  图早于步{robot.img_pre_step_ms:+.0f}ms", flush=True)
                cadence.tick()
    except KeyboardInterrupt:
        print("\n[循环] 手动中断", flush=True)
    finally:
        stats = {k.split("_")[0]: f"消息{c.msgs} 帧{c.frames_total} 无帧{c.msgs_no_frame} "
                                   f"多帧{c.msgs_multi_frame} 丢弃{c.dropped} "
                                   f"重连{c.clock_resets} VCL{c.vcl_hist} B帧{c.b_slices} "
                                   f"一致性[VCL无帧{c.msg_vcl_no_frame} "
                                   f"帧无VCL{c.msg_frame_no_vcl} 无戳帧{c.frames_no_ts}] "
                                   f"配对{'OK' if c.pairing_ok else 'FAIL:' + str(c.pair_error)}"
                 for k, c in robot.cameras.items()}
        print(f"[相机] {stats}", flush=True)
        if tr is not None and tr["step"]:
            np.savez(cfg.trace, **{k: np.asarray(v) for k, v in tr.items()})
            print(f"[trace] 已保存 {cfg.trace}: {len(tr['step'])} 步 / {len(tr['chunk_step'])} 块", flush=True)
        robot.disconnect()

    print(f"[完成] {step} 步，{time.time() - cadence.t_start:.1f}s", flush=True)
    if step:
        verdict = (f"放置成功（步 {success_step}，盒内稳定 {cfg.success_hold} 步）"
                   if success_step is not None else "放置失败（超时/未进盒）")
        print(f"[结果] {verdict}"
              f"  首次合爪={'步 ' + str(first_close) if first_close else '未合爪'}"
              f"  最终TCP→方块={robot.tcp_cube_cm():.2f}cm"
              f"  方块→盒={robot.cube_box_dist_cm():.2f}cm"
              f"  方块z={robot.cube_height():.3f}m"
              f"  跟踪误差峰值={err_peak:.4f} rad"
              f"（{err_peak_joint.replace(f'{cfg.agent_name}_', '') if err_peak_joint else '-'}）",
              flush=True)
    cadence.report()


def main():
    p = argparse.ArgumentParser(description="G1 抓方块：仿真闭环推理（LeRobot rollout 范式）")
    d = RolloutConfig()
    p.add_argument("--host", default=d.host)
    p.add_argument("--port", type=int, default=d.port)
    p.add_argument("--agent-name", default=d.agent_name)
    p.add_argument("--lerobot-root", default=d.lerobot_root)
    p.add_argument("--norm-root", default=d.norm_root)
    p.add_argument("--episode", type=int, default=d.episode)
    p.add_argument("--prompt", default=d.prompt)
    p.add_argument("--cameras", default=d.cameras)
    p.add_argument("--fps", type=float, default=d.fps)
    p.add_argument("--execution-steps", type=int, default=d.execution_steps)
    p.add_argument("--blend-steps", type=int, default=d.blend_steps)
    p.add_argument("--blend-alpha", type=float, default=d.blend_alpha)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--orcagym-addr", default=d.orcagym_addr)
    p.add_argument("--task-config", default=d.task_config)
    p.add_argument("--dump-dir", default=d.dump_dir)
    p.add_argument("--trace", default=d.trace)
    p.add_argument("--capture-dir", default=d.capture_dir)
    p.add_argument("--capture-tcp-cm", type=float, default=d.capture_tcp_cm)
    p.add_argument("--cam-stale-ms", type=int, default=d.cam_stale_ms)
    p.add_argument("--cam-sync-ms", type=int, default=d.cam_sync_ms)
    p.add_argument("--state-img-tol-ms", type=int, default=d.state_img_tol_ms)
    p.add_argument("--success-radius", type=float, default=d.success_radius)
    p.add_argument("--success-floor-tol", type=float, default=d.success_floor_tol)
    p.add_argument("--success-speed", type=float, default=d.success_speed)
    p.add_argument("--success-hold", type=int, default=d.success_hold)
    p.add_argument("--wn", type=float, default=d.wn)
    p.add_argument("--zeta", type=float, default=d.zeta)
    p.add_argument("--max-tau", type=float, default=d.max_tau)
    p.add_argument("--max-tau-wrist", type=float, default=d.max_tau_wrist)
    p.add_argument("--kp", type=float, default=d.kp)
    p.add_argument("--kd", type=float, default=d.kd)
    p.add_argument("--manual-gains", dest="auto_gains", action="store_false")
    p.add_argument("--no-gravity-comp", dest="gravity_comp", action="store_false")
    p.add_argument("--grip-range", type=float, nargs=2, default=list(d.grip_range))
    p.add_argument("--settle-frames", type=int, default=d.settle_frames)
    args = p.parse_args()

    cfg = RolloutConfig(**vars(args))
    cfg.grip_range = tuple(args.grip_range)
    run_rollout(cfg)


if __name__ == "__main__":
    main()
