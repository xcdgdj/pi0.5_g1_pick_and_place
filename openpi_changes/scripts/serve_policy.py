import dataclasses
import enum
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    # ★ 本地补丁：指定策略采样种子（环境变量 OPENPI_POLICY_SEED）。
    # π₀.₅ 是流匹配，动作从噪声积分出来；上游 Policy 固定用 jax.random.key(0)，
    # 且每次 infer 都 split 一次往前走 —— 所以「第几次调用」决定这一次的噪声，
    # 重启服务才回到噪声流起点。换种子 = 换一整条噪声流，用来做可复现性验证与种子筛选。
    if (_seed := os.environ.get("OPENPI_POLICY_SEED")) is not None:
        import jax

        policy._rng = jax.random.key(int(_seed))  # noqa: SLF001
        logging.info("策略采样种子已设为 %s（OPENPI_POLICY_SEED）", _seed)
    # ★ 本地补丁：噪声状态的存/取。每次 infer 前把 rng 状态打进日志（可据此复现任意一集），
    # OPENPI_POLICY_RNG_LOAD=<npy> 则把状态原样载回来 —— 等价于「load 之前存下来的噪声」。
    if os.environ.get("OPENPI_POLICY_SEED") is not None or os.environ.get("OPENPI_POLICY_RNG_LOG"):
        import jax
        import numpy as _np

        if (_f := os.environ.get("OPENPI_POLICY_RNG_LOAD")):
            policy._rng = jax.random.wrap_key_data(_np.load(_f))  # noqa: SLF001
            logging.info("已载入 RNG 状态 %s（OPENPI_POLICY_RNG_LOAD）", _f)
        _dump = os.environ.get("OPENPI_POLICY_RNG_DUMP")  # 目录：每次调用存一份 rng_<n>.npy
        if _dump:
            os.makedirs(_dump, exist_ok=True)
        _orig_infer, _n = policy.infer, [0]

        def _obs_fp(o):
            """观测指纹：state 取前 18 维四舍五入，图像取字节和 —— 用来判断两次运行看到的输入是否一致。
            相机流是异步的（约 6~7fps），每次推理看到哪一帧并不确定，这是噪声之外的另一个随机源。"""
            import hashlib

            h = hashlib.md5()
            for k in sorted(o):
                v = o[k]
                if isinstance(v, dict):
                    for kk in sorted(v):
                        h.update(_np.asarray(v[kk]).tobytes())
                elif isinstance(v, _np.ndarray):
                    h.update(_np.asarray(v).tobytes())
                else:
                    h.update(str(v).encode())
            return h.hexdigest()[:12]

        def _infer_logged(obs, **kw):
            _state = jax.random.key_data(policy._rng)  # noqa: SLF001
            logging.info("[RNG] call=%d state=%s obs=%s", _n[0], _np.asarray(_state).tolist(), _obs_fp(obs))
            if _dump:
                _np.save(os.path.join(_dump, f"rng_{_n[0]:05d}.npy"), _np.asarray(_state))
            _n[0] += 1
            return _orig_infer(obs, **kw)

        policy.infer = _infer_logged
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
