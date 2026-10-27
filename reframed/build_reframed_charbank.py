"""Character bank for REFRAMED (train + validation + test), in the CMD-AD metadata.json format.

Movies already in the CMD-AD bank reuse that entry, and their portraits are symlinks to the
CMD-AD images (same cast list as the CMD-AD Qwen3.8 run). New movies are fetched from IMDb
with process_single_movie() from srt_for_ads/ift/data_creation/build_char_bank.py.

Out: {out}/metadata.json, {out}/images/{actor_imdb_id}.jpg, {out}/movies.json (split + source per movie)
"""

import argparse
import json
import os
import sys
import types
from multiprocessing import Pool

BUILDER_DIR = "/home/dbalu/workspace/srt_for_ads/ift/data_creation"
CMD_META = "/storage/datasets/CMD-AD/data/character_bank/metadata.json"
REFRAMED = "/storage/datasets/Reframed"

sys.modules["utils"] = types.SimpleNamespace(load_cmd_ad_annotations=None, load_config=None)  # skip its heavy deps
sys.path.insert(0, BUILDER_DIR)
from build_char_bank import process_single_movie  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REFRAMED}/character_bank")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    img_dir = f"{args.out}/images"
    os.makedirs(img_dir, exist_ok=True)

    split_of = {}
    for split in ("train", "validation", "test"):
        for m in json.load(open(f"{REFRAMED}/{split}/meta.json")):
            split_of[m] = split
    cmd = json.load(open(CMD_META))
    meta = json.load(open(f"{args.out}/metadata.json")) if os.path.exists(f"{args.out}/metadata.json") else {}

    # reused from CMD-AD: same entries, portraits symlinked
    n_link = 0
    for m in split_of:
        if m not in cmd:
            continue
        entry = {}
        for k, e in cmd[m].items():
            e = dict(e)
            src = e.get("picture_path")
            if src and os.path.exists(src):
                dst = f"{img_dir}/{os.path.basename(src)}"
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
                    n_link += 1
                e["picture_path"] = dst
            entry[k] = e
        meta[m] = entry

    # new movies: IMDb
    todo = [m for m in split_of if m not in meta or not meta[m]]
    have = {os.path.splitext(f)[0] for f in os.listdir(img_dir)}
    print(f"{len(split_of)} movies: {sum(m in cmd for m in split_of)} from CMD-AD ({n_link} new symlinks), "
          f"{len(todo)} to fetch from IMDb", flush=True)
    with Pool(args.workers) as p:
        for key, data, _ in p.imap_unordered(process_single_movie, [(m, img_dir, have) for m in todo]):
            if data:
                meta[key] = data
            json.dump(meta, open(f"{args.out}/metadata.json", "w"), indent=2, ensure_ascii=False)

    json.dump(meta, open(f"{args.out}/metadata.json", "w"), indent=2, ensure_ascii=False)
    json.dump({m: {"split": s, "source": "cmd_ad" if m in cmd else "imdb", "n_cast": len(meta.get(m, {}))}
               for m, s in split_of.items()}, open(f"{args.out}/movies.json", "w"), indent=1)
    missing = [m for m in split_of if not meta.get(m)]
    print(f"done: {len(meta)} movies in bank, missing: {missing}", flush=True)


if __name__ == "__main__":
    main()
