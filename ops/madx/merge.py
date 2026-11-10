#!/usr/bin/env python3
"""MADX (garuda): merge the TF shards of one model into {wid: predict} for t3/score_mad_seg.py --preds-json.
Shard dirs: <root>/s<i>of<K>/{teacher_forced_predictions.jsonl, idx.json}; rows are in idx.json["wid"] order.
  python merge.py --shards DIR --out preds_by_wid.json [--expect 576]
"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True); ap.add_argument("--out", required=True); ap.add_argument("--expect", type=int, default=576)
a = ap.parse_args()
out, report = {}, {}
for d in sorted(glob.glob(f"{a.shards}/s*of*")):
    wids = json.load(open(f"{d}/idx.json"))["wid"]
    rows = [json.loads(l) for l in open(f"{d}/teacher_forced_predictions.jsonl")]
    report[os.path.basename(d)] = (len(rows), len(wids))
    assert len(rows) == len(wids), (d, len(rows), len(wids))
    for w, r in zip(wids, rows):
        assert w not in out, w
        out[w] = r["predict"]
print(report, "windows", len(out))
assert len(out) == a.expect, len(out)
json.dump(out, open(a.out, "w"), ensure_ascii=False)
