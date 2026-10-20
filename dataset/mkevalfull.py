import json
import numpy as np
exec(open('/home/ubuntu/adlm/mk150.py').read().split('d = json.load')[0])
d = json.load(open("/data/cmd_ad/datasets/bins_frames/gapframes_bins_eval.json"))
for r in d:
    cap(r)
e = np.array([est(r) for r in d])
keep = [r for r, x in zip(d, e) if x <= 40000]
json.dump(keep, open("/data/cmd_ad/datasets/bins_frames/eval_full.json", "w"), ensure_ascii=False)
ke = np.array([est(r) for r in keep])
print(f"eval_full: {len(keep)} clips | tokens p50 {np.median(ke):,.0f} max {ke.max():,.0f} "
      f"| {sum(r['messages'][1]['content'].count('[AD]') for r in keep):,} AD lines")
