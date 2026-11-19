#!/usr/bin/env python3
"""LeRobot v2.1 数据集健康检查。

检查项:
  1. 总帧数（并和 meta/info.json 的 total_frames 对账）
  2. 所有 parquet 的 NaN / Inf
  3. 各维度 q01 / q99 极值, 以及相邻帧最大跳变（手柄漂移导致的瞬间跳变）
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

ARM_DIMS = 7          # 前 7 维为手臂关节（rad）
ARM_LIMIT_RAD = 3.2   # 手臂关节物理可达范围之外视为异常
GRIP_LIMIT_M = 0.1    # 夹爪关节（m）
JUMP_LIMIT_RAD = 1.0  # 单帧跳变上限（rad）


def load_dim_names(dataset: str, feature: str) -> list[str]:
    path = os.path.join(dataset, "meta", "info.json")
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return info.get("features", {}).get(feature, {}).get("names") or []


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot 数据集健康检查")
    ap.add_argument("--dataset", required=True, help="LeRobot 数据集根目录")
    ap.add_argument("--cols", default="observation.state,action",
                    help="要检查的列, 逗号分隔")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(
        args.dataset, "data", "chunk-*", "*.parquet")))
    if not files:
        raise SystemExit(f"未找到 parquet: {args.dataset}/data/chunk-*")

    cols = [c.strip() for c in args.cols.split(",")]
    total_frames = 0
    data = {c: [] for c in cols}
    episode_ids = []
    problems: list[str] = []

    for path in files:
        table = pq.read_table(path, columns=cols + ["episode_index"])
        episode_ids.append(int(table.column("episode_index")[0].as_py()))
        total_frames += table.num_rows
        for c in cols:
            arr = np.asarray(table.column(c).to_pylist(), dtype=np.float64)
            bad = ~np.isfinite(arr)
            if bad.any():
                ep = episode_ids[-1]
                dims = sorted(set(np.where(bad)[1].tolist()))
                problems.append(
                    f"{c}: episode {ep} 含 {int(bad.sum())} 个 NaN/Inf, 维度 {dims}")
            data[c].append(arr)

    # 与 meta 对账
    info_path = os.path.join(args.dataset, "meta", "info.json")
    meta_frames = None
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            meta_frames = json.load(f).get("total_frames")

    print("=" * 78)
    print(f"数据集      {args.dataset}")
    print(f"parquet     {len(files)} 个, episode 数 {len(episode_ids)}")
    print(f"总帧数      {total_frames}"
          + (f"（meta: {meta_frames}"
             + ("，一致）" if meta_frames == total_frames else "，不一致！）")
             if meta_frames is not None else ")"))
    print(f"缺失值      {'未发现 NaN/Inf' if not problems else '发现问题'}")
    print("=" * 78)

    for c in cols:
        stacked = np.concatenate(data[c], axis=0)
        names = load_dim_names(args.dataset, c)
        if len(names) != stacked.shape[1]:
            names = [f"dim{i}" for i in range(stacked.shape[1])]

        q01 = np.quantile(stacked, 0.01, axis=0)
        q99 = np.quantile(stacked, 0.99, axis=0)
        dmin = stacked.min(axis=0)
        dmax = stacked.max(axis=0)

        # 相邻帧最大跳变（按 episode 分段计算）
        jumps = np.zeros(stacked.shape[1])
        jump_ep = np.full(stacked.shape[1], -1)
        offset = 0
        for a, ep in zip(data[c], episode_ids):
            if len(a) > 1:
                d = np.abs(np.diff(a, axis=0)).max(axis=0)
                upd = d > jumps
                jumps[upd] = d[upd]
                jump_ep[upd] = ep
            offset += len(a)

        print(f"\n[{c}]  shape=({stacked.shape[0]}, {stacked.shape[1]})")
        print(f"  {'维度':<32}{'min':>9}{'q01':>9}{'q99':>9}"
              f"{'max':>9}{'max|Δ|':>9}  判定")
        for i, name in enumerate(names):
            limit = ARM_LIMIT_RAD if i < ARM_DIMS else GRIP_LIMIT_M
            verdict = "OK"
            if max(abs(q01[i]), abs(q99[i])) > limit or max(
                    abs(dmin[i]), abs(dmax[i])) > limit:
                verdict = "极值异常"
                problems.append(
                    f"{c}[{name}] 超出合理范围 "
                    f"(min={dmin[i]:.3f}, q01={q01[i]:.3f}, "
                    f"q99={q99[i]:.3f}, max={dmax[i]:.3f})")
            if jumps[i] > JUMP_LIMIT_RAD:
                verdict = (verdict + " / 跳变" if verdict != "OK" else "跳变")
                problems.append(
                    f"{c}[{name}] 相邻帧跳变 {jumps[i]:.3f} > "
                    f"{JUMP_LIMIT_RAD}（episode {jump_ep[i]}）")
            print(f"  {name:<32}{dmin[i]:>9.4f}{q01[i]:>9.4f}"
                  f"{q99[i]:>9.4f}{dmax[i]:>9.4f}{jumps[i]:>9.4f}  {verdict}")

    print("\n" + "=" * 78)
    if problems:
        print(f"结论: 发现 {len(problems)} 个问题")
        for p in problems:
            print("  - " + p)
    else:
        print("结论: 通过（无 NaN/Inf, 无异常极值, 无异常跳变）")
    print("=" * 78)


if __name__ == "__main__":
    main()
