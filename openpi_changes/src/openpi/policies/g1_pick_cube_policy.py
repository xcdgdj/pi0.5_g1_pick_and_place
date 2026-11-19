"""宇树 G1 单臂（右臂 + DEX1 夹爪）抓取任务的 openpi 数据变换。

数据集只包含右臂与两路相机，格式与推理客户端保持一致::

    observation = {
        "state": (9,),                          # 右臂 7 关节(rad) + 右夹爪 2 关节(m)
        "images": {
            "base_0_rgb":       (H, W, 3) uint8,  # 俯视 Camera
            "right_wrist_0_rgb": (H, W, 3) uint8,  # 右腕特写 wrist_r
        },
        "prompt": str,
    }
    result["actions"] -> (chunk, 9)

模型侧固定三路图像键，因此左腕位补零并把 mask 置 False。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 数据集的 state / action 维度：右臂 7 + 右夹爪 2。
ACTION_DIM = 9


def make_delta_mask() -> tuple[bool, ...]:
    """前 7 维手臂关节取增量，后 2 维夹爪保持绝对值。"""
    return transforms.make_bool_mask(7, -2)


def _parse_image(image) -> np.ndarray:
    """统一成 uint8 的 (H, W, C)。

    训练时 LeRobot 给的是 float32 CHW，推理时客户端给的是 uint8 CHW，
    两条路径都要能处理。
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1PickCubeInputs(transforms.DataTransformFn):
    """把数据集/客户端格式转成模型输入格式。训练和推理都会走这里。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["images"]["base_0_rgb"])
        wrist = _parse_image(data["images"]["right_wrist_0_rgb"])

        # 只有俯视 + 右腕两路；左腕补零。PI0_FAST 需要三路真实图像才能
        # tokenize，这里数据集本身没有左腕，因此同样置 False。
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
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
class G1PickCubeOutputs(transforms.DataTransformFn):
    """把模型输出裁回数据集的 9 维。仅推理时使用。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIM])}
