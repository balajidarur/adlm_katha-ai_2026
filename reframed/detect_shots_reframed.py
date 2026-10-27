"""Shot cache for REFRAMED made exactly like the CMD-AD one (batch_round_pipeline.py): detect_shots() from
test_adlm_descriptions.py (ffmpeg scene filter) with the pipeline params from config.yaml.

Out: {out} = {excerpt_id: [{"id", "start", "end"}, ...]}  (same format as shots_cache_gemini-3.5-flash-lite.json)
"""

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

REPO = "/home/dbalu/workspace/ad_gen/adlm_katha-ai_2026"
sys.path.insert(0, REPO)
from omegaconf import OmegaConf  # noqa: E402
from test_adlm_descriptions import detect_shots  # noqa: E402

P = OmegaConf.load(f"{REPO}/config.yaml").pipeline


def one(video):
    shots = detect_shots(video, threshold=P.scene_threshold, max_shot_duration=P.get("max_shot_duration", 10.0),
                         min_shot_duration=P.get("min_shot_duration", 3.0),
                         shot_duration_tolerance=P.get("shot_duration_tolerance", 0.3),
                         shot_absorb_threshold=P.get("shot_absorb_threshold", 1.0))
    return os.path.basename(video)[:-4], shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/dbalu/reframed/shots_cache_scenes.json")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    vids = sorted(glob.glob("/storage/datasets/Reframed/*/excerpt/video/*/*.mp4"))
    cache = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [v for v in vids if os.path.basename(v)[:-4] not in cache]
    print(f"{len(vids)} videos, {len(todo)} to do; params {dict(P)}", flush=True)
    with Pool(args.workers) as p:
        for i, (k, s) in enumerate(p.imap_unordered(one, todo), 1):
            cache[k] = s
            if i % 100 == 0:
                json.dump(cache, open(args.out, "w"))
                print(f"{i}/{len(todo)}", flush=True)
    json.dump(cache, open(args.out, "w"))
    L = [x["end"] - x["start"] for v in cache.values() for x in v]
    print(f"DONE {len(cache)} videos, {len(L)} shots, mean {sum(L) / len(L):.2f}s, "
          f"max {max(L):.2f}s, at 10s cap {sum(abs(x - 10) < 0.01 for x in L) / len(L):.0%}", flush=True)


if __name__ == "__main__":
    main()
