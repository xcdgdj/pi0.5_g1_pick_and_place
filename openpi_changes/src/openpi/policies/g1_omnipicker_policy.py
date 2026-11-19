"""SouthGrid 智元 G1 OmniPicker 的 openpi 数据变换（Task3 电力工器具归纳收置）。

观测/动作接口必须与 SouthGrid 的推理客户端逐字对齐，见
``SouthGrid/src/examples/inference/g1_omnipicker/eval_g1_omnipicker_tool_lerobot.py``：

    observation = {"state": (18,), "images": {"cam_head": CHW, "cam_wrist_r": CHW}, "prompt": str}
    result["actions"] -> (chunk, >=18)

18 维的语义（位置与四元数都在基座系 ``g1_omnipicker_body_link1``）::

    [0:3]   左臂位置        [3:7]   左臂四元数 xyzw
    [7:10]  右臂位置        [10:14] 右臂四元数 xyzw
    [14:16] 左夹爪 内/外    [16:18] 右夹爪 内/外   （均已归一化到 [0,1]）
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 数据集与推理客户端共用的动作维度。
ACTION_DIM = 18

# 左臂 7 维 + 左夹爪 2 维在训练数据中是恒定常量（100 集统计，标准差 ≤7mm）。
# 但推理时机器人的左臂停在另一个位姿：训练数据中左臂被设成水平后落下、稳定在下垂位
# （数据记的是下垂位 z=-0.456），而推理脚本在它落下之前就锁死在水平位（z=+0.301），
# y 更是差了 0.72 m。这 9 维会进 state，pi05 又把 state 离散成语言 token，
# 于是策略从第 0 帧起就读到训练中从未出现过的 state 串。
#
# 这些维度对任务毫无信息量（左臂全程不动），所以在变换层直接钉成数据集常量。
# 训练与推理走同一个变换，钉完两边永远一致；训练侧的改动 ≤7mm，可忽略。
_LEFT_ARM_CONST = np.array(
    [0.117616, 0.256256, -0.456056, 0.696451, 0.082139, -0.711682, 0.040714], dtype=np.float32
)
_LEFT_GRIP_CONST = np.array([0.047967, 0.047967], dtype=np.float32)


def _pin_left_arm(state: np.ndarray) -> np.ndarray:
    """把 state 里锁死的左臂/左夹爪维度钉成数据集常量。"""
    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape[-1] >= 16:
        state[..., 0:7] = _LEFT_ARM_CONST
        state[..., 14:16] = _LEFT_GRIP_CONST
    return state


def make_g1_omnipicker_example() -> dict:
    """构造一个与推理客户端格式一致的随机样例（用于 serve_policy 自检）。"""
    return {
        "state": np.random.rand(ACTION_DIM),
        "images": {
            "cam_head": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_wrist_r": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "整理工具",
    }


def _parse_image(image) -> np.ndarray:
    """统一成 uint8 的 (H, W, C)。

    训练时 LeRobot 给的是 float32 CHW，推理时客户端给的是 uint8 CHW，两条路都要能吃。
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1OmniPickerInputs(transforms.DataTransformFn):
    """把客户端/数据集格式转成模型输入格式。训练和推理都会走这里。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        head = _parse_image(data["images"]["cam_head"])
        wrist_r = _parse_image(data["images"]["cam_wrist_r"])

        # 只有头部 + 右腕两路；左腕位置补零并在非 FAST 模型上屏蔽掉。
        inputs = {
            "state": _pin_left_arm(data["state"]),
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": np.zeros_like(head),
                "right_wrist_0_rgb": wrist_r,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # actions 只有训练时才有；padding 到模型维度由 PadStatesAndActions 负责。
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1OmniPickerOutputs(transforms.DataTransformFn):
    """把模型输出裁回数据集的 18 维。仅推理时使用。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}