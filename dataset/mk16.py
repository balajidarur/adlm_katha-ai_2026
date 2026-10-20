"""Pick 16 clips spread across the frame-count range (not the 16 smallest)."""
import json
import numpy as np

src = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_eval.json"
d = json.load(open(src))
# require real bin signal: several AD lines, and bins that are not all budget_8
cand = []
for r in d:
    t = r["messages"][1]["content"]
    n_ad = t.count("[AD]")
    bins = set(int(b) for b in __import__("re").findall(r"<\|budget_(\d)\|>", t))
    if n_ad >= 4 and len(bins) >= 4 and len(r["images"]) <= 140:
        cand.append((len(r["images"]), len(bins), n_ad, r))
cand.sort(key=lambda x: x[0])
idx = [int(x) for x in np.linspace(0, len(cand) - 1, 16)]
pick = [cand[i] for i in idx]
out = [p[3] for p in pick]
json.dump(out, open("/data/cmd_ad/datasets/bins_frames/sixteen.json", "w"), ensure_ascii=False)
print(f"candidates: {len(cand)}  ->  picked 16")
print("frames per clip :", [p[0] for p in pick])
print("distinct bins   :", [p[1] for p in pick])
print("AD lines        :", [p[2] for p in pick])
print(f"total frames {sum(p[0] for p in pick)} | total AD lines {sum(p[2] for p in pick)}")
