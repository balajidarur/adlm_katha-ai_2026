"""Merge ft_infer.sh shards (stream/*.jsonl idx -> ids.json) into one preds file {video_id, predict}.
  python merge_preds.py RAWDIR TAG SPLIT OUT.jsonl"""
import glob, json, sys
raw, tag, split, out = sys.argv[1:5]
rows = {}
for d in sorted(glob.glob(f"{raw}/{tag}_{split}_s*of*")):
    ids = json.load(open(f"{d}/ids.json"))
    for f in glob.glob(f"{d}/stream/*.jsonl"):
        for l in open(f):
            r = json.loads(l); rows[ids[r["idx"]]] = r["predict"]
want = json.load(open("/workspace/dbalu/tasks_nov8/T5/split.json"))[split]
miss = [v for v in want if v not in rows]
with open(out, "w") as fh:
    for v in want:
        if v in rows:
            fh.write(json.dumps({"video_id": v, "predict": rows[v]}) + "\n")
n_ad = sum(p.count("[AD]") for p in rows.values())
print(f"{tag} {split}: {len(rows)}/{len(want)} videos, {n_ad} ADs, missing {len(miss)} {miss[:5]}")
