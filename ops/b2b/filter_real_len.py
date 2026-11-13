#!/usr/bin/env python3
"""Drop training records whose REAL tokenized length exceeds a cap (the 40k "est" cap in mk150 undercounts the repetitive
2B dense descriptions: real lengths reach 48.8k and OOM a 46 GB L40S at step 13). Lengths come from the LLaMA-Factory
tokenized cache of the full set (same order as the json). Writes <out>.json and adds it to dataset_info.json.
  python filter_real_len.py <dataset_dir> <name> <tokcache_dir> <max_tokens> <out_name>"""
import json, sys
import numpy as np
from datasets import load_from_disk
D, name, tc, cap, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
recs = json.load(open(f"{D}/{name}.json"))
t = load_from_disk(tc); t = t["train"] if hasattr(t, "keys") else t
L = np.array([len(x) for x in t["input_ids"]]); assert len(L) == len(recs), (len(L), len(recs))
keep = [r for r, l in zip(recs, L) if l <= cap]
json.dump(keep, open(f"{D}/{out}.json", "w"), ensure_ascii=False)
info = json.load(open(f"{D}/dataset_info.json")); info[out] = dict(info[name], file_name=f"{out}.json")
json.dump(info, open(f"{D}/dataset_info.json", "w"), indent=2)
print(f"FILTER {name}: {len(recs)} -> {len(keep)} records (real tokens <= {cap}; max kept {int(L[L <= cap].max())}, dropped {int((L > cap).sum())})")
