"""Eval subsets from the held-out CMD-AD split, same caps as training."""
import json
import numpy as np
exec(open('/home/ubuntu/adlm/mk150.py').read().split('d = json.load')[0])  # cap/est helpers

SRC = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_eval.json"
d = json.load(open(SRC))
for r in d:
    cap(r)                       # frames -> max 150, thinned per dialogue gap
e = np.array([est(r) for r in d])
keep = [r for r, x in zip(d, e) if x <= 40000]
# spread across the length range rather than taking the easiest
ke = np.array([est(r) for r in keep])
order = np.argsort(ke)
pick50 = [keep[order[i]] for i in np.linspace(0, len(order) - 1, 50).astype(int)]
one = [pick50[len(pick50) // 2]]          # a mid-length clip

json.dump(one, open("/data/cmd_ad/datasets/bins_frames/eval_one.json", "w"), ensure_ascii=False)
json.dump(pick50, open("/data/cmd_ad/datasets/bins_frames/eval_50.json", "w"), ensure_ascii=False)
print(f"eval pool {len(keep)}/{len(d)} clips (dropped {len(d)-len(keep)} over 40k)")
f1 = one[0]
print(f"eval_one : {len(f1['images'])} frames, {est(f1):,.0f} tok, "
      f"{f1['messages'][1]['content'].count('[AD]')} AD lines")
f50 = np.array([est(r) for r in pick50])
print(f"eval_50  : tokens {f50.min():,.0f}-{f50.max():,.0f} (median {np.median(f50):,.0f}), "
      f"{sum(r['messages'][1]['content'].count('[AD]') for r in pick50)} AD lines total")
