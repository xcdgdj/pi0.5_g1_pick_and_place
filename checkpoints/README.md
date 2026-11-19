# 权重未包含在本仓库

训练好的 checkpoint 位于训练机（2×4090）：

```
/home/orca/openpi/checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora/
├── 1000/ ... 20000/    每 1000 步一档，共 20 档
```

单档约数 GB，超出 GitHub 单文件限制，不随仓库分发。

## 获取方式

1. 从训练机打包（推荐只传 15000 与 19999 两档）：
   ```bash
   tar -czf pi05_g1_pick_cube_lora_15000.tar.gz \
     -C /home/orca/openpi/checkpoints/pi05_g1_pick_cube_lora/g1_pick_cube_lora 15000
   ```
2. 上传到 HuggingFace Hub 或网盘，在 README 里补充下载链接。
3. 解压到本地后，`--policy.dir` 指向该目录即可。

## 注意

- 每个 checkpoint 目录内含 `assets/<asset_id>/norm_stats.json`，**推理必须使用这一份**，不要用新采数据重算。
- 权重与 norm_stats 绑定：换数据集重训后必须使用新的 checkpoint 与其自带 stats。
