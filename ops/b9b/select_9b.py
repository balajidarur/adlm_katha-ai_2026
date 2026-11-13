#!/usr/bin/env python3
"""Assemble the Qwen3.5-9B-DD variant of M3 (ops/b9b/b9b_build.sh step 3).

  train   : M3's train clips (bins_f150_40k) in M3's order, rebuilt with 9B DDs, 150-frame cap, dropped above 40k tokens
  eval527 : M3's eval527 clips in M3's order, 150-frame cap, none dropped (cutoff_len is 49,152; over-40k ones reported)
  eval591 : eval527 + the 64 clips of cmd_ad_eval591 (missing64.txt order) at rows 527-590, 150-frame cap, none dropped.
            The special clips come from the isolated build (official videos / ASR), budget bins added WITHOUT dropping clips
            whose ADs get 0 bins; their frame paths are rewritten to /data/cmd_ad/frames/eval591_official/{vid}/ (s3
            adlm/frames/eval591_official/). The rest come from the main build: binned record if add_bins kept it, else the
            unbinned record converted with no drop (same as T2).
Writes <name>.json files, dataset_info.json, SHA256SUMS and build_report.json.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "..", "dataset")
sys.path.insert(0, DS)
import add_bins_to_gapframes as ab
mk = {}
exec(open(f"{DS}/mk150.py").read().split("d = json.load")[0], mk)
BUDGET = 40000
OFFICIAL = "/data/cmd_ad/frames/eval591_official"


def vid_of(rec):
    return rec["images"][0].split("/")[-2] if rec["images"] else rec["messages"][-1]["content"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nodrop(recs, metas, budget):
    out = {}
    for r, m in zip(recs, metas):
        ab.convert(r, m["video_id"], budget)
        assert r["messages"][0]["content"].count("<image>") == len(r["images"])
        out[m["video_id"]] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("main", "special", "m3", "missing64", "special-frames", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep, budget = {}, ab.load_budget()

    tr = {vid_of(r): r for r in json.load(open(f"{a.main}/gapframes_bins_train.json"))}
    out_tr, miss, over = [], [], []
    for r in json.load(open(f"{a.m3}/gapframes_bins_train_f150_40k.json")):
        v = vid_of(r)
        if v not in tr:
            miss.append(v); continue
        x = tr[v]; mk["cap"](x)
        if mk["est"](x) > BUDGET:
            over.append(v); continue
        out_tr.append(x)
    rep["train"] = dict(clips=len(out_tr), m3=len(out_tr) + len(miss) + len(over), missing=miss, dropped_over_40k=over)

    ev_bins = {vid_of(r): r for r in json.load(open(f"{a.main}/gapframes_bins_eval.json"))}
    ev_raw = nodrop(json.load(open(f"{a.main}/cmdad_dds_gapframes_eval.json")),
                    json.load(open(f"{a.main}/cmdad_dds_gapframes_eval_meta.json")), budget)
    sp = nodrop(json.load(open(f"{a.special}/cmdad_dds_gapframes_eval.json")),
                json.load(open(f"{a.special}/cmdad_dds_gapframes_eval_meta.json")), budget)
    for v, r in sp.items():
        r["images"] = [p.replace(a.special_frames.rstrip("/"), OFFICIAL) for p in r["images"]]
    out527, over527, miss527 = [], [], []
    for r in json.load(open(f"{a.m3}/eval527.json")):
        v = vid_of(r)
        x = ev_bins.get(v)
        if x is None:
            miss527.append(v); continue
        mk["cap"](x); out527.append(x)
        if mk["est"](x) > BUDGET: over527.append(v)
    rep["eval527"] = dict(clips=len(out527), missing=miss527, over_40k_kept=over527)
    src64 = {}
    ext = []
    for line in open(a.missing64):
        c = line.strip()
        if not c: continue
        v = c.split("/")[-1]
        if v in sp: x, s = sp[v], "special"
        elif v in ev_bins: x, s = ev_bins[v], "main_binned"
        elif v in ev_raw: x, s = ev_raw[v], "main_nodrop"
        else: src64[v] = "MISSING"; continue
        mk["cap"](x); ext.append(x); src64[v] = s
    out591 = out527 + ext
    rep["eval591"] = dict(clips=len(out591), new64_sources={k: sum(1 for x in src64.values() if x == k) for k in set(src64.values())},
                          missing=[v for v, s in src64.items() if s == "MISSING"], over_40k_kept=[vid_of(x) for x in out591 if mk["est"](x) > BUDGET])
    files = {"dd9b_f150_40k": out_tr, "dd9b_eval527": out527, "dd9b_eval591": out591}
    tmpl = next(iter(json.load(open(f"{a.m3}/dataset_info.json")).values()))
    info = {}
    for k, recs in files.items():
        json.dump(recs, open(f"{a.out}/{k}.json", "w"), ensure_ascii=False)
        info[k] = dict(tmpl, file_name=f"{k}.json")
    json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    json.dump({"report": rep, "eval591_new64_source": src64}, open(f"{a.out}/build_report.json", "w"), indent=1)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in [f"{k}.json" for k in files] + ["dataset_info.json", "build_report.json"]:
            f.write(f"{sha(f'{a.out}/{fn}')}  {fn}\n")
    print(json.dumps(rep, indent=1)[:3000])
