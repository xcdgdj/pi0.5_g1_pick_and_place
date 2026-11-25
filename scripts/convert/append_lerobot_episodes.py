"""把一个 LeRobot v2.1 数据集追加到另一个数据集，生成合并副本。

用法:
    python append_lerobot_episodes.py --base /home/user/pi/lerobot_g1_pick \
        --add /home/user/pi/lerobot_g1_pick_place5 --out /home/user/pi/lerobot_g1_pick_v2

行为:
  - base 的 data/videos 用硬链接复制（不占额外空间），meta 正常复制
  - add 的 episode 重编号为 base 的续号，global index 顺延，task_index 按任务文本映射
  - 重写 meta/info.json / episodes.jsonl / episodes_stats.jsonl / tasks.jsonl
  - 删除继承来的 norm_stats.json（合并后必须重算）
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ["observation.images.base_0_rgb", "observation.images.right_wrist_0_rgb"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.unlink(missing_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base, add, out = Path(args.base), Path(args.add), Path(args.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在: {out}")

    base_info = json.loads((base / "meta/info.json").read_text())
    add_info = json.loads((add / "meta/info.json").read_text())
    base_eps = read_jsonl(base / "meta/episodes.jsonl")
    add_eps = read_jsonl(add / "meta/episodes.jsonl")
    base_stats = read_jsonl(base / "meta/episodes_stats.jsonl")
    add_stats = read_jsonl(add / "meta/episodes_stats.jsonl")
    base_tasks = read_jsonl(base / "meta/tasks.jsonl")
    add_tasks = read_jsonl(add / "meta/tasks.jsonl")

    n0 = len(base_eps)
    total0 = int(base_info["total_frames"])
    task_by_text = {t["task"]: t["task_index"] for t in base_tasks}
    for t in add_tasks:
        if t["task"] not in task_by_text:
            task_by_text[t["task"]] = len(task_by_text)
    tasks_out = [{"task_index": i, "task": s} for s, i in task_by_text.items()]

    # 1) base → out（data/videos 硬链接，meta 普通复制）
    out.mkdir(parents=True)
    for sub in ["data", "videos"]:
        if (base / sub).is_dir():
            shutil.copytree(base / sub, out / sub, copy_function=os.link)
    shutil.copytree(base / "meta", out / "meta")
    (out / "norm_stats.json").unlink(missing_ok=True)

    # 2) 追加 add 的 episode
    new_eps, new_stats = [], []
    global_index = total0
    add_task_index = {}
    for i, ep in enumerate(add_eps):
        src = add / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        table = pq.read_table(src)
        m = table.num_rows
        new_idx = n0 + i
        # task_index 按任务文本映射到合并后的编号
        tsk = int(table["task_index"][0].as_py())
        text = next(t["task"] for t in add_tasks if t["task_index"] == tsk)
        mapped = task_by_text[text]
        add_task_index[tsk] = mapped
        table = table.set_column(
            table.schema.get_field_index("episode_index"), "episode_index",
            pa.array([new_idx] * m, type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("index"), "index",
            pa.array(range(global_index, global_index + m), type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("task_index"), "task_index",
            pa.array([mapped] * m, type=pa.int64()),
        )
        dst = out / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet"
        pq.write_table(table, dst)
        global_index += m

        for cam in CAMS:
            src_v = add / "videos" / "chunk-000" / cam / f"episode_{i:06d}.mp4"
            dst_v = out / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
            link_or_copy(src_v, dst_v)

        new_eps.append({"episode_index": new_idx, "tasks": ep["tasks"], "length": ep["length"]})
        st = dict(add_stats[i])
        st["episode_index"] = new_idx
        new_stats.append(st)
        print(f"  {i:03d} → {new_idx:03d}  {ep['length']} 帧")

    # 3) meta
    write_jsonl(out / "meta/episodes.jsonl", base_eps + new_eps)
    write_jsonl(out / "meta/episodes_stats.jsonl", base_stats + new_stats)
    write_jsonl(out / "meta/tasks.jsonl", tasks_out)
    info = dict(base_info)
    info["total_episodes"] = n0 + len(add_eps)
    info["total_frames"] = total0 + int(add_info["total_frames"])
    (out / "meta/info.json").unlink()
    (out / "meta/info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4))

    print(f"合并完成: {n0} + {len(add_eps)} = {info['total_episodes']} 集 / "
          f"{info['total_frames']} 帧 → {out}")


if __name__ == "__main__":
    main()
