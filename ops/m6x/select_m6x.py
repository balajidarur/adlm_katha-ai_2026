#!/usr/bin/env python3
"""M6 (Qwen3.5-4B DDs) on the full CMD-AD eval591: rows 0-526 = the exact M6 eval527 records (s3 adlm/datasets/cmdad_qwen35_4b_dd/
dd4b_eval527.json, the set behind M6's 38.65), rows 527-590 = the 64 clips of cmd_ad_eval591 (missing64.txt order) rebuilt with 4B
DDs exactly as ops/b2b/select_2b.py does for 2B (special clips from the isolated build with official videos / ASR, frame paths
rewritten to /data/cmd_ad/frames/eval591_official; others from the main eval build, binned if add_bins kept them, else no-drop).
  python select_m6x.py --eval527 dd4b_eval527.json --main V --special S/build --missing64 m64.txt --special-frames F --out O"""
import argparse, hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); DS = os.path.join(HERE, "..", "..", "dataset"); sys.path.insert(0, DS)
import add_bins_to_gapframes as ab
mk = {}; exec(open(f"{DS}/mk150.py").read().split("d = json.load")[0], mk)
OFFICIAL = "/data/cmd_ad/frames/eval591_official"
vid_of = lambda r: r["images"][0].split("/")[-2] if r["images"] else r["messages"][-1]["content"]
def sha(p):
    h = hashlib.sha256(); f = open(p, "rb")
    for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def nodrop(recs, metas, budget):
    out = {}
    for r, m in zip(recs, metas):
        ab.convert(r, m["video_id"], budget); assert r["messages"][0]["content"].count("<image>") == len(r["images"]); out[m["video_id"]] = r
    return out
ap = argparse.ArgumentParser()
for k in ("eval527", "main", "special", "missing64", "special-frames", "out"): ap.add_argument(f"--{k}", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); budget = ab.load_budget()
out527 = json.load(open(a.eval527)); assert len(out527) == 527
ev_bins = {vid_of(r): r for r in json.load(open(f"{a.main}/gapframes_bins_eval.json"))} if os.path.exists(f"{a.main}/gapframes_bins_eval.json") else {}
ev_raw = nodrop(json.load(open(f"{a.main}/cmdad_dds_gapframes_eval.json")), json.load(open(f"{a.main}/cmdad_dds_gapframes_eval_meta.json")), budget)
sp = nodrop(json.load(open(f"{a.special}/cmdad_dds_gapframes_eval.json")), json.load(open(f"{a.special}/cmdad_dds_gapframes_eval_meta.json")), budget)
for v, r in sp.items(): r["images"] = [p.replace(a.special_frames.rstrip("/"), OFFICIAL) for p in r["images"]]
src, ext = {}, []
for line in open(a.missing64):
    c = line.strip()
    if not c: continue
    v = c.split("/")[-1]
    if v in sp: x, s = sp[v], "special"
    elif v in ev_bins: x, s = ev_bins[v], "main_binned"
    elif v in ev_raw: x, s = ev_raw[v], "main_nodrop"
    else: src[v] = "MISSING"; continue
    mk["cap"](x); ext.append(x); src[v] = s
out = out527 + ext
rep = dict(clips=len(out), sources={k: sum(1 for x in src.values() if x == k) for k in set(src.values())}, missing=[v for v, s in src.items() if s == "MISSING"])
json.dump(out, open(f"{a.out}/dd4b_eval591.json", "w"), ensure_ascii=False)
json.dump({"report": rep, "src64": src}, open(f"{a.out}/build_report.json", "w"), indent=1)
open(f"{a.out}/SHA256SUMS", "w").write("".join(f"{sha(f'{a.out}/{f}')}  {f}\n" for f in ("dd4b_eval591.json", "build_report.json")))
print(json.dumps(rep)); assert not rep["missing"] and len(out) == 591, "eval591 incomplete"
