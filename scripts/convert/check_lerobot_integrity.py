"""检查 LeRobot v2.1 数据集物理文件与元数据是否严格对齐。

检查项:
  1) info.json 的 total_episodes / total_frames 与 parquet、episodes.jsonl 是否一致
  2) episodes.jsonl / episodes_stats.jsonl 行数与 episode_index 是否连续
  3) data/ 与 videos/ 下文件是否严格 45 个、编号 0..44、与 jsonl 一一对应
  4) 每个 episode 的 parquet 行数是否等于两路 mp4 帧数（物理对齐）
  5) parquet 内部 index / timestamp / frame_index 是否连续单调

用法:
    python check_lerobot_integrity.py
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/user/pi/lerobot_g1_pick")
CAMS = ["base_0_rgb", "right_wrist_0_rgb"]
problems: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  [{'OK ' if ok else 'BAD'}] {msg}")
    if not ok:
        problems.append(msg)


def main():
    info = json.loads((ROOT / "meta/info.json").read_text())
    episodes = [json.loads(l) for l in (ROOT / "meta/episodes.jsonl").read_text().splitlines() if l]
    stats = [json.loads(l) for l in (ROOT / "meta/episodes_stats.jsonl").read_text().splitlines() if l]

    print("=" * 70)
    print("1. info.json 与 parquet 对账")
    parquet_files = sorted((ROOT / "data").rglob("*.parquet"))
    rows = {int(re.search(r"episode_(\d+)\.parquet$", p.name).group(1)): pq.read_table(p).num_rows
            for p in parquet_files}
    total_rows = sum(rows.values())
    check(info["total_episodes"] == 45, f"total_episodes = {info['total_episodes']}（应为 45）")
    check(info["total_frames"] == total_rows,
          f"total_frames = {info['total_frames']}，parquet 总行数 = {total_rows}")
    check(info["total_frames"] == sum(e["length"] for e in episodes),
          f"total_frames = {info['total_frames']}，episodes.jsonl 长度和 = {sum(e['length'] for e in episodes)}")
    check(info["total_episodes"] == len(episodes) == len(stats),
          f"info.episodes = {info['total_episodes']}，episodes.jsonl = {len(episodes)}，"
          f"episodes_stats.jsonl = {len(stats)}")
    check(info["splits"] == {"train": "0:45"}, f"splits = {info['splits']}")

    print("\n2. 元数据连续性")
    ep_idx = [e["episode_index"] for e in episodes]
    st_idx = [s["episode_index"] for s in stats]
    check(ep_idx == list(range(45)), f"episodes.jsonl episode_index = {ep_idx[:3]}...{ep_idx[-3:]}，"
                                     f"连续 0..44: {ep_idx == list(range(45))}")
    check(st_idx == list(range(45)), f"episodes_stats.jsonl episode_index 连续: {st_idx == list(range(45))}")
    check(all(e["length"] == rows[e["episode_index"]] for e in episodes),
          "每集 episodes.jsonl 的 length 与 parquet 行数一致")
    check(all(st["stats"]["action"]["count"][0] == rows[st["episode_index"]] for st in stats),
          "episodes_stats.jsonl 的 action.count 与 parquet 行数一致")

    print("\n3. 物理文件与编号")
    check(len(parquet_files) == 45, f"data/ parquet 文件数 = {len(parquet_files)}")
    check(sorted(rows) == list(range(45)), f"parquet 编号 = 0..44: {sorted(rows) == list(range(45))}")
    for cam in CAMS:
        vids = sorted((ROOT / "videos" / "chunk-000" / f"observation.images.{cam}").glob("*.mp4"))
        ids = sorted(int(re.search(r"episode_(\d+)\.mp4$", p.name).group(1)) for p in vids)
        check(len(vids) == 45 and ids == list(range(45)),
              f"{cam}: {len(vids)} 个 mp4，编号 0..44: {ids == list(range(45))}")
        check(ids == ep_idx, f"{cam}: 文件编号与 episodes.jsonl 一一对应")

    print("\n4. 每集 parquet 行数 vs 两路视频帧数")
    bad = []
    for e in episodes:
        i = e["episode_index"]
        for cam in CAMS:
            cap = cv2.VideoCapture(str(ROOT / "videos/chunk-000" / f"observation.images.{cam}" /
                                      f"episode_{i:06d}.mp4"))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n != rows[i]:
                bad.append((i, cam, n, rows[i]))
    check(not bad, f"45 集 × 2 路全部一致" if not bad else f"不一致: {bad[:5]}")

    print("\n5. parquet 内部时间轴")
    offs = np.cumsum([0] + [e["length"] for e in episodes])
    bad_ts, bad_gi = [], []
    for e in episodes:
        i = e["episode_index"]
        t = pq.read_table(ROOT / "data/chunk-000" / f"episode_{i:06d}.parquet")
        ts = np.asarray(t.column("timestamp").to_pylist(), dtype=np.float64)
        fi = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
        gi = np.asarray(t.column("index").to_pylist(), dtype=np.int64)
        ei = np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64)
        if not np.allclose(ts, np.arange(len(ts)) / info["fps"], atol=1e-5):
            bad_ts.append(i)
        if not (np.array_equal(fi, np.arange(len(fi))) and
                np.array_equal(gi, np.arange(offs[i], offs[i] + len(gi))) and
                np.array_equal(ei, np.full(len(gi), i))):
            bad_gi.append(i)
    check(not bad_ts, f"timestamp = frame/fps：{'全部一致' if not bad_ts else bad_ts}")
    check(not bad_gi, f"frame_index 0..M-1、全局 index 连续、episode_index 正确："
                      f"{'全部一致' if not bad_gi else bad_gi}")

    print("\n" + "=" * 70)
    if problems:
        print(f"发现 {len(problems)} 处问题：")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"全部通过：45 集，{total_rows} 帧，parquet {len(parquet_files)} 个，"
          f"mp4 {2 * len(parquet_files)} 个，元数据与物理文件严格对齐。")


if __name__ == "__main__":
    main()
