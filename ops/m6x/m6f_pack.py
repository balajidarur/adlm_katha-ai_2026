#!/usr/bin/env python3
"""Box A: package the 64 extra eval591 clips (rows 527-590 of dd4b_eval591.json) for M6 free-running elsewhere: frames copied
under OUT/frames/<vid>/<file>, record image paths rewritten to /scratch/m6f/frames/..., OUT/m6f64.json + OUT/ids.json (591 index).
  python m6f_pack.py /prep/m6x/out/dd4b_eval591.json /prep/m6x/m6f"""
import json, os, shutil, sys
src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src)); assert len(d) == 591
rows = d[527:]; os.makedirs(f"{out}/frames", exist_ok=True); n = 0
for r in rows:
    new = []
    for p in r["images"]:
        v, f = p.split("/")[-2], p.split("/")[-1]; dst = f"{out}/frames/{v}/{f}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst): shutil.copy(p, dst); n += 1
        new.append(f"/scratch/m6f/frames/{v}/{f}")
    r["images"] = new
json.dump(rows, open(f"{out}/m6f64.json", "w"), ensure_ascii=False); json.dump(list(range(527, 591)), open(f"{out}/ids591.json", "w"))
print("PACKED", len(rows), "clips", n, "frames")
