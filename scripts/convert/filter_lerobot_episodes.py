"""从 LeRobot v2.1 数据集中剔除指定 episode，并重编号为连续索引。

只重写 data/videos/meta，不动 norm_stats.json（剔除后需重新跑 compute_norm_stats.py）。
被剔除的集在源 HDF5 里仍然保留。

用法:
    python filter_lerobot_episodes.py --drop 1 12 26 36 41 --dry-run
    python filter_lerobot_episodes.py --drop 1 12 26 36 41
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ["base_0_rgb", "right_wrist_0_rgb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/pi/lerobot_g1_pick")
    ap.add_argument("--drop", nargs="+", type=int, required=True, help="要剔除的 LeRobot episode 编号")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    episodes = [json.loads(l) for l in (meta / "episodes.jsonl").read_text().splitlines() if l]
    stats_all = [json.loads(l) for l in (meta / "episodes_stats.jsonl").read_text().splitlines() if l]

    drop = set(args.drop)
    missing = drop - {e["episode_index"] for e in episodes}
    if missing:
        raise SystemExit(f"数据集里没有这些编号: {sorted(missing)}")

    keep = [e for e in episodes if e["episode_index"] not in drop]
    mapping = {e["episode_index"]: i for i, e in enumerate(keep)}
    print(f"原 {len(episodes)} 集 → 保留 {len(keep)} 集")
    print("重编号映射:", {old: new for old, new in mapping.items() if old != new})
    print("剔除:", sorted(drop))

    if args.dry_run:
        print("dry-run，未改动任何文件")
        return

    stage = root / "_stage"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "data" / "chunk-000").mkdir(parents=True)
    for cam in CAMS:
        (stage / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True)

    global_index = 0
    new_eps, new_stats = [], []
    stats_by_idx = {s["episode_index"]: s for s in stats_all}

    for old, new in mapping.items():
        src = root / "data" / "chunk-000" / f"episode_{old:06d}.parquet"
        t = pq.read_table(src)
        m = t.num_rows
        t = t.set_column(
            t.schema.get_field_index("episode_index"), "episode_index",
            pa.array(np.full(m, new, dtype=np.int64)),
        )
        t = t.set_column(
            t.schema.get_field_index("index"), "index",
            pa.array(np.arange(global_index, global_index + m, dtype=np.int64)),
        )
        pq.write_table(t, stage / "data" / "chunk-000" / f"episode_{new:06d}.parquet")

        for cam in CAMS:
            shutil.copy2(
                root / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{old:06d}.mp4",
                stage / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{new:06d}.mp4",
            )

        new_eps.append({"episode_index": new, "tasks": next(
            e for e in episodes if e["episode_index"] == old)["tasks"], "length": m})
        s = dict(stats_by_idx[old])
        s["episode_index"] = new
        new_stats.append(s)
        global_index += m

    shutil.rmtree(root / "data")
    shutil.rmtree(root / "videos")
    shutil.move(str(stage / "data"), str(root / "data"))
    shutil.move(str(stage / "videos"), str(root / "videos"))
    shutil.rmtree(stage)

    info.update(
        total_episodes=len(new_eps),
        total_frames=global_index,
        total_videos=len(new_eps) * len(CAMS),
        total_chunks=1,
        splits={"train": f"0:{len(new_eps)}"},
    )
    (meta / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in new_eps))
    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in new_stats))

    print(f"完成：{len(new_eps)} 集，{global_index} 帧，"
          f"parquet {len(list((root/'data').rglob('*.parquet')))} 个，"
          f"mp4 {len(list((root/'videos').rglob('*.mp4')))} 个")


if __name__ == "__main__":
    main()
