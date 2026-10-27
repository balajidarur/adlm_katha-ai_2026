"""Lay REFRAMED out the way prep_charbank.py expects CMD-AD (CMD_AD_ROOT={root}):

  {root}/videos/{movie}/{excerpt}.mp4                      symlinks into the extracted splits
  {root}/annotations/cmd_ad_anno_v1.csv                    cmd_filename=movie/excerpt, imdbid, split
  {root}/annotations/character_bank/metadata.json          REFRAMED character bank
  {root}/dense/shots_cache_gemini-3.5-flash-lite.json      fixed 10 s shots per excerpt
  {root}/clips_all.txt                                      one movie/excerpt per line

Shots: windows of SHOT_LEN s from 0 to the video's duration (ffprobe). The CMD-AD cache also
caps shots at 10 s. A last window shorter than ABSORB s (absorb_threshold in config.yaml) is
merged into the one before.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

SHOT_LEN = 10.0
ABSORB = 1.0


def duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def windows(dur):
    shots, t, i = [], 0.0, 1
    while t < dur - 1e-6:
        shots.append({"id": i, "start": round(t, 3), "end": round(min(t + SHOT_LEN, dur), 3)})
        t += SHOT_LEN
        i += 1
    if len(shots) > 1 and shots[-1]["end"] - shots[-1]["start"] < ABSORB:
        shots[-2]["end"] = shots.pop()["end"]
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/scratch/reframed")
    ap.add_argument("--root", default="/scratch/rf_data")
    ap.add_argument("--charbank", required=True, help="character_bank/metadata.json")
    ap.add_argument("--shots-cache", default=None, help="use these shots (e.g. detect_shots_reframed.py output) instead of fixed windows")
    args = ap.parse_args()
    for d in ("videos", "annotations/character_bank", "dense"):
        os.makedirs(f"{args.root}/{d}", exist_ok=True)

    rows = []
    for split in ("train", "validation", "test"):
        for v in sorted(glob.glob(f"{args.src}/{split}/excerpt/video/*/*.mp4")):
            movie, exc = v.split("/")[-2], os.path.basename(v)[:-4]
            os.makedirs(f"{args.root}/videos/{movie}", exist_ok=True)
            dst = f"{args.root}/videos/{movie}/{exc}.mp4"
            if not os.path.lexists(dst):
                os.symlink(v, dst)
            rows.append((f"{movie}/{exc}", movie, split, v))

    with ThreadPoolExecutor(48) as ex:
        durs = list(ex.map(duration, [r[3] for r in rows]))
    cache = {r[0].split("/")[1]: windows(d) for r, d in zip(rows, durs)}
    if args.shots_cache:
        given = json.load(open(args.shots_cache))
        cache = {k: given[k] for k in cache}
    json.dump(cache, open(f"{args.root}/dense/shots_cache_gemini-3.5-flash-lite.json", "w"))
    with open(f"{args.root}/annotations/cmd_ad_anno_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cmd_filename", "imdbid", "split", "duration"])
        for r, d in zip(rows, durs):
            w.writerow([r[0], r[1], r[2], round(d, 3)])
    open(f"{args.root}/clips_all.txt", "w").write("".join(r[0] + "\n" for r in rows))
    os.system(f"cp {args.charbank} {args.root}/annotations/character_bank/metadata.json")
    n = sum(len(s) for s in cache.values())
    print(f"{len(rows)} excerpts ({sum(r[2] == 'train' for r in rows)} train), {n} shots, "
          f"{sum(durs) / 3600:.1f} h, unique excerpt ids: {len(cache)}")


if __name__ == "__main__":
    main()
