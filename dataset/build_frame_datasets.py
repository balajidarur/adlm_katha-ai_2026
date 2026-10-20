#!/usr/bin/env python
"""Build the frame-interleaved CMD-AD datasets A / B / C (see DATASET_PLAN.md).

One 1 FPS frame pool is extracted per clip; the three variants reference subsets of it.

    # pilot: 5 random clips per split
    python build_frame_datasets.py --n-per-split 5
    # full run
    python build_frame_datasets.py --n-per-split 0 --workers 60

Frames: every variant has at most --cap frames (200) and fits --token-budget (60k). A span whose
1 FPS frames fit is used as is; otherwise the same span is resampled uniformly at a lower rate
(A/C: the whole clip; B: the dialogue-gap windows), so no clip is dropped or cut short.

Frames -> {frame_root}/{video_id}/f_{k:05d}.jpg   (f_0000k = the picture at t = k + 0.5 s)
          {frame_root}/{video_id}/u_{ms:07d}.jpg  (resampled frames, the picture at ms / 1000 s)
JSON   -> {out}/cmdad_{variant name}_{split}.json + _meta.json + dataset_info.json
          e.g. frame_variants_448x252_pilot5/cmdad_B_dense_dialogue_gapframes_eval.json
"""

import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import frames_lib as fl
from analysis_lib import ntok


def build_clip(job):
    """One clip. A failure is recorded in build_stats.json, never allowed to kill the run."""
    try:
        return _build_clip(job)
    except Exception as e:
        return {"video_id": job[0]["video_id"], "error": f"{type(e).__name__}: {e}"[:300]}


def _build_clip(job):
    row, ads, args = job
    vid, year = row["video_id"], row["year"]
    info = fl.video_info(fl.video_path(year, vid))
    if info is None:
        return {"video_id": vid, "error": "video unreadable"}
    dense, events, dense_end = fl.load_dense(vid)
    end = min(dense_end, info["dur"])
    dialogue = fl.parse_srt(fl.asr_path(year, vid))
    grid = fl.frame_times(end)
    windows = fl.windows_for(dialogue, end)

    t0 = time.time()
    clip_dir = f"{args.frame_root}/{vid}"
    got = sorted(f for f in os.listdir(clip_dir) if f.startswith("f_")) if os.path.isdir(clip_dir) else []
    if len(got) < len(grid):   # a re-run skips clips whose 1 FPS frames are already on disk
        got = fl.extract_frames(fl.video_path(year, vid), clip_dir, len(grid),
                                args.width, args.height, threads=args.ffmpeg_threads)
    # a video shorter than its dense timeline yields fewer frames: keep the ones that exist
    grid = grid[:min(len(got), len(grid))]

    target = fl.render_target(dialogue, ads)
    out = {"video_id": vid, "records": {}, "extract_s": round(time.time() - t0, 2),
           "n_extracted": len(got)}
    target_tok = ntok(target)
    per_frame = fl.frame_tokens(args.width, args.height)
    plans = {}
    for v in fl.VARIANTS:
        n_1fps = len(fl.select_frames(v, grid, windows))
        text_tok = ntok(fl.render_user(v, dense, events, dialogue, np.array([]))) + target_tok
        # at most `cap` frames, then only as many as fit the token budget next to the full text;
        # when that is fewer than the span's 1 FPS frames, the span is resampled uniformly
        n = fl.fit_frames(min(n_1fps, args.cap or n_1fps), text_tok, args.token_budget, per_frame)
        times, on_grid = fl.plan_frames(v, grid, windows, end, n)
        span = end if v in ("A", "C") else sum(e - s for s, e in windows)
        plans[v] = (times, on_grid, n_1fps, text_tok, span)
    off_grid = sorted({float(t) for times, on_grid, *_ in plans.values() if not on_grid for t in times
                       if not os.path.exists(fl.resampled_file(args.frame_root, vid, t))})
    if off_grid:
        fl.extract_at(fl.video_path(year, vid), args.frame_root, vid, np.array(off_grid),
                      args.width, args.height, threads=args.ffmpeg_threads)
    out["extract_s"] = round(time.time() - t0, 2)
    for v in fl.VARIANTS:
        times, on_grid, n_1fps, text_tok, span = plans[v]
        user = fl.render_user(v, dense, events, dialogue, times)
        images = [(fl.frame_file if on_grid else fl.resampled_file)(args.frame_root, vid, t)
                  for t in times]
        assert user.count("<image>") == len(images), (vid, v)
        assert all(os.path.exists(p) for p in images), (vid, v)
        hits = fl.frames_in_ads(ads, times)
        cov = fl.window_overlap(ads, windows) if v == "B" else np.ones(len(ads))
        out["records"][v] = {
            "record": {"messages": [{"role": "user", "content": user},
                                    {"role": "assistant", "content": target}],
                       "images": images},
            "meta": {"video_id": vid, "year": year, "split": row["split"], "variant": v,
                     "variant_name": fl.VARIANT_NAMES[v],
                     "video_dur": round(info["dur"], 2), "content_end": round(end, 2),
                     "n_grid": len(grid), "n_frames": len(images),
                     "n_frames_1fps": n_1fps,
                     "sampling": "1fps" if on_grid else "uniform",
                     "fps": round(len(images) / span, 3) if span else 0.0,
                     "seq_tokens": text_tok + len(images) * per_frame,
                     "n_events": len(events) if v != "A" else 0,
                     "n_dialogue": len(dialogue), "n_ad": len(ads),
                     "ad_window_covered": int((cov >= 0.5).sum()),
                     "ad_with_frame": int((hits > 0).sum())},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=5, help="0 = every clip")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=252)
    ap.add_argument("--cap", type=int, default=200,
                    help="max frames per clip, subsampled evenly across it; 0 = none")
    ap.add_argument("--token-budget", type=int, default=60000,
                    help="max prompt+target tokens per clip (4B on one 48 GB A6000 held 60,645); "
                         "frames are subsampled evenly to fit. 0 = off")
    ap.add_argument("--ids", nargs="*", default=None, help="build only these video ids")
    ap.add_argument("--frame-root", default=None)
    ap.add_argument("--out", default=None,
                    help="default: datasets/frame_variants_{W}x{H}_pilot{N} or _full")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ffmpeg-threads", type=int, default=0)
    args = ap.parse_args()
    args.frame_root = args.frame_root or f"{fl.ROOT}/frames/1fps_{args.width}x{args.height}"
    args.out = args.out or (f"{fl.ROOT}/datasets/frame_variants_{args.width}x{args.height}_"
                            + ("ids" if args.ids else
                               f"pilot{args.n_per_split}" if args.n_per_split else "full"))

    ads = fl.load_ads()
    index = [r for r in fl.load_index() if r["split"] in args.splits]
    if args.ids:
        index = [r for r in index if r["video_id"] in set(args.ids)]
    rng = random.Random(args.seed)
    jobs = []
    for split in args.splits:
        rows = [r for r in index if r["split"] == split]
        if args.n_per_split and not args.ids:
            rng.shuffle(rows)
            # skip the truncated downloads so the pilot always gets n usable clips
            rows = [r for r in rows if fl.video_info(fl.video_path(r["year"], r["video_id"]))]
            rows = rows[:args.n_per_split]
        jobs += [(r, ads.get(r["video_id"], []), args) for r in rows]
    print(f"clips: {len(jobs)}  frames -> {args.frame_root}  json -> {args.out}")

    # longest clips first, so a few 5-minute clips don't straggle at the end (this makes the early
    # rate slower than the average, so tqdm's first ETAs run long)
    shots = json.load(open(f"{fl.ROOT}/dense/shots_cache_{fl.DENSE_MODEL}.json"))
    jobs.sort(key=lambda j: -(shots.get(j[0]["video_id"]) or [{"end": 0}])[-1]["end"])
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(build_clip, jobs, chunksize=1), total=len(jobs),
                            desc="clips", unit="clip", smoothing=0.05, mininterval=2, dynamic_ncols=True))
    failed = [r for r in results if "error" in r]
    results = sorted((r for r in results if "error" not in r), key=lambda r: r["video_id"])
    print(f"built: {len(results)}  failed: {len(failed)} {[r['video_id'] for r in failed][:10]}")

    os.makedirs(args.out, exist_ok=True)
    info, stats = {}, {}
    for split in args.splits:
        for v in fl.VARIANTS:
            recs = [r["records"][v] for r in results if r["records"][v]["meta"]["split"] == split]
            name = f"cmdad_{fl.VARIANT_NAMES[v]}_{split}"
            with open(f"{args.out}/{name}.json", "w") as fh:
                json.dump([r["record"] for r in recs], fh, ensure_ascii=False, indent=1)
            meta = [r["meta"] for r in recs]
            with open(f"{args.out}/{name}_meta.json", "w") as fh:
                json.dump(meta, fh, indent=1)
            info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                          "columns": {"messages": "messages", "images": "images"},
                          "tags": {"role_tag": "role", "content_tag": "content",
                                   "user_tag": "user", "assistant_tag": "assistant",
                                   "system_tag": "system"}}
            n_ad = sum(m["n_ad"] for m in meta)
            stats[name] = {"clips": len(meta), "frames": sum(m["n_frames"] for m in meta),
                           "clips_resampled": sum(m["sampling"] == "uniform" for m in meta),
                           "ads": n_ad,
                           "ad_window_covered": sum(m["ad_window_covered"] for m in meta),
                           "ad_with_frame": sum(m["ad_with_frame"] for m in meta)}
            print(f"{name:42s} {stats[name]}")
    with open(f"{args.out}/dataset_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
    with open(f"{args.out}/build_stats.json", "w") as fh:
        json.dump({"args": {k: v for k, v in vars(args).items()}, "stats": stats,
                   "failed": [r["video_id"] for r in failed],
                   "extract_s": {r["video_id"]: r["extract_s"] for r in results}}, fh, indent=2)


if __name__ == "__main__":
    main()
