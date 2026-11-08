#!/usr/bin/env python3
"""E3 probe sets: exact Qwen3-VL-8B sequence lengths of every training record, then
probe_top10 = the longest 10% (sorted longest first) and probe_rand96 = 96 random records (seed 0).

Length = the qwen3_vl_nothink chat template over the user + assistant turns, the 9 budget tokens added as
special tokens (as in training), each `<image>` expanded to <|vision_start|> + N x <|image_pad|> + <|vision_end|>,
with N from the Qwen3-VL image processor on a real frame (448x252 -> 112).

  python e3_probe_ds.py --model <Qwen3-VL-8B dir> --train <q3vl8b_f150_40k.json> --out <dir>
"""
import argparse, hashlib, json, os, random

import numpy as np
from PIL import Image
from transformers import AutoProcessor

BIN_TOKENS = [f"<|budget_{i}|>" for i in range(9)]

ap = argparse.ArgumentParser()
for k in ("model", "train", "out"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
proc = AutoProcessor.from_pretrained(a.model)
tok = proc.tokenizer
tok.add_tokens(BIN_TOKENS, special_tokens=True)
d = json.load(open(a.train))
img = Image.open(next(r["images"][0] for r in d if r["images"])).convert("RGB")
grid = proc.image_processor(images=[img], return_tensors="np")["image_grid_thw"][0]
n_img = int(np.prod(grid)) // (proc.image_processor.merge_size ** 2)
print(f"frame {img.size} -> grid_thw {grid.tolist()} -> {n_img} image tokens per frame")
vis = "<|vision_start|>" + "<|image_pad|>" * n_img + "<|vision_end|>"


def length(r):
    u = r["messages"][0]["content"].replace("<image>", vis)
    t = r["messages"][1]["content"]
    s = (f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{t}<|im_end|>\n")
    return len(tok(s, add_special_tokens=False)["input_ids"])


L = np.array([length(r) for r in d])
est = np.array([112 * len(r["images"]) + (len(r["messages"][0]["content"]) + len(r["messages"][1]["content"])) / 3.6
                for r in d])
order = np.argsort(-L)
k = int(round(0.1 * len(d)))
top = [d[i] for i in order[:k]]
rnd = [d[i] for i in random.Random(0).sample(range(len(d)), 96)]
stats = dict(n=len(d), image_tokens_per_frame=n_img, max=int(L.max()), p50=int(np.median(L)), p90=int(np.percentile(L, 90)),
             mean=float(L.mean()), over_49152=int((L > 49152).sum()), over_40000=int((L > 40000).sum()),
             top10_n=k, top10_min=int(L[order[k - 1]]), top10_mean=float(L[order[:k]].mean()),
             rand96_mean=float(np.mean([L[i] for i in random.Random(0).sample(range(len(d)), 96)])),
             ratio_exact_over_est_mean=float((L / est).mean()))
print(json.dumps(stats, indent=1))
json.dump(stats, open(f"{a.out}/lengths_stats.json", "w"), indent=1)
json.dump([int(x) for x in L], open(f"{a.out}/lengths.json", "w"))
info = {}
tmpl = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                 "system_tag": "system"}}
for name, recs in (("probe_top10", top), ("probe_rand96", rnd)):
    json.dump(recs, open(f"{a.out}/{name}.json", "w"), ensure_ascii=False)
    info[name] = dict(tmpl, file_name=f"{name}.json")
json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
with open(f"{a.out}/SHA256SUMS", "w") as f:
    for fn in ("probe_top10.json", "probe_rand96.json", "dataset_info.json"):
        f.write(hashlib.sha256(open(f"{a.out}/{fn}", "rb").read()).hexdigest() + f"  {fn}\n")
