"""极简检查：按训练完全相同的管线取 1 个 batch，打印图像与归一化 action 的 shape/min/max。

用法（必须在 openpi 环境里跑）：
    cd /home/user/openpi && .venv/bin/python /home/user/pi/g1_lerobot/print_one_batch.py
"""

import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

CONFIG_NAME = "pi05_g1_pick_cube_lora"
BATCH_SIZE = 2


def main():
    cfg = _config.get_config(CONFIG_NAME)
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    if data_cfg.norm_stats is None:
        raise SystemExit("找不到 norm_stats.json，先运行 scripts/compute_norm_stats.py")
    print(f"config={CONFIG_NAME}  batch={BATCH_SIZE}  action_horizon={cfg.model.action_horizon}")
    print(f"repo_id={data_cfg.repo_id}")
    print(f"norm_stats keys={sorted(data_cfg.norm_stats)}  use_quantile={data_cfg.use_quantile_norm}\n")

    # 与 training 完全一致的顺序：repack → data_transforms(含 DeltaActions) → Normalize → model_transforms
    dataset = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.transform_dataset(dataset, data_cfg)

    batch = _data_loader._collate_fn([dataset[i] for i in range(BATCH_SIZE)])
    obs = _model.Observation.from_dict(batch)
    actions = np.asarray(batch["actions"])

    for name, img in obs.images.items():
        img = np.asarray(img)
        print(
            f"image[{name:20s}] shape={img.shape} dtype={img.dtype} "
            f"min={img.min():+.4f} max={img.max():+.4f} mean={img.mean():+.4f}"
        )
    print(f"image_mask={ {k: bool(np.asarray(v).all()) for k, v in obs.image_masks.items()} }")

    state = np.asarray(obs.state)
    print(f"state            shape={state.shape} dtype={state.dtype} "
          f"min={state.min():+.4f} max={state.max():+.4f}")

    print(f"\nactions(归一化后) shape={actions.shape} dtype={actions.dtype} "
          f"min={actions.min():+.4f} max={actions.max():+.4f}")
    for d in range(9):
        col = actions[..., d]
        print(f"  dim {d}: min={col.min():+.4f} max={col.max():+.4f}")
    print(f"  dim 9..31 是否全为 0（padding）: {bool(np.allclose(actions[..., 9:], 0.0))}")

    if obs.tokenized_prompt is not None:
        print(f"tokenized_prompt shape={np.asarray(obs.tokenized_prompt).shape}")


if __name__ == "__main__":
    main()
