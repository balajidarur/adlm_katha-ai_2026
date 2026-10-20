"""Same 150-frame cap, tighter 40k token budget (44k OOM'd twice at 43.76 GiB)."""
import json, re
import numpy as np
exec(open('/home/ubuntu/adlm/mk150.py').read().split('d = json.load')[0])  # reuse helpers
SRC = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train.json"
OUT = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train_f150_40k.json"
BUDGET = 40000
d = json.load(open(SRC))
for r in d:
    cap(r)
e = np.array([est(r) for r in d])
keep = [r for r, x in zip(d, e) if x <= BUDGET]
json.dump(keep, open(OUT, "w"), ensure_ascii=False)
ke = np.array([est(r) for r in keep])
f = np.array([len(r["images"]) for r in keep])
print(f"{len(keep):,} clips (dropped {len(d)-len(keep):,}) | frames max {f.max()} | tokens max {ke.max():,.0f}")
print(f"steps/epoch {len(keep)//32} | 3 epochs = {3*(len(keep)//32)}")
