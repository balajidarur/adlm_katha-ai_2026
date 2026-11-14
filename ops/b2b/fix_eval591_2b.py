#!/usr/bin/env python3
"""Re-assemble dd2b_eval591 with all 591 clips (select_2b.py lost 8 eval527 clips).

select_2b.py keys records by vid_of(), which falls back to the assistant text when a record has no images. With the long,
repetitive Qwen3.5-2B DDs some clips' text alone exceeds the 48k build budget, so the builder gives them 0 frames; for
eval527 those clips then could not be found in the binned set and were reported "missing" (the 64 extra clips were not
affected: they also fall back to the meta-keyed unbinned records). Here eval527 uses the same rule as the extra 64: binned
record if add_bins kept it (matched by assistant text when it has no images), else the unbinned record converted with no
drop. Everything else is select_2b.py unchanged; records of the 583 clips already in --old must come out identical.
  python fix_eval591_2b.py --main V --special S/build --m3 M3DIR --missing64 F --special-frames S/frames/1fps_448x252 \
      --old dd2b_eval591.json --out dd2b_eval591_full.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_2b as s

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("main", "special", "m3", "missing64", "special-frames", "old", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    budget = s.ab.load_budget()
    bins = json.load(open(f"{a.main}/gapframes_bins_eval.json"))
    ev_bins = {s.vid_of(r): r for r in bins if r["images"]}
    bins_noimg = {r["messages"][-1]["content"]: r for r in bins if not r["images"]}
    ev_raw = s.nodrop(json.load(open(f"{a.main}/cmdad_dds_gapframes_eval.json")),
                      json.load(open(f"{a.main}/cmdad_dds_gapframes_eval_meta.json")), budget)
    sp = s.nodrop(json.load(open(f"{a.special}/cmdad_dds_gapframes_eval.json")),
                  json.load(open(f"{a.special}/cmdad_dds_gapframes_eval_meta.json")), budget)
    for v, r in sp.items():
        r["images"] = [p.replace(a.special_frames.rstrip("/"), s.OFFICIAL) for p in r["images"]]

    def pick(v, special_ok):
        if special_ok and v in sp: return sp[v], "special"
        if v in ev_bins: return ev_bins[v], "main_binned"
        if v in ev_raw and ev_raw[v]["messages"][-1]["content"] in bins_noimg and not ev_raw[v]["images"]:
            return bins_noimg[ev_raw[v]["messages"][-1]["content"]], "main_binned_noimg"
        if v in ev_raw: return ev_raw[v], "main_nodrop"
        raise SystemExit(f"MISSING {v}")

    keys, out, src = [], [], {}
    for r in json.load(open(f"{a.m3}/eval527.json")):
        v = s.vid_of(r); x, src[v] = pick(v, False); keys.append(v); out.append(x)
    for line in open(a.missing64):
        if line.strip():
            v = line.strip().split("/")[-1]; x, src[v] = pick(v, True); keys.append(v); out.append(x)
    for x in out:
        s.mk["cap"](x)
    assert len(out) == 591 and len(set(keys)) == 591, (len(out), len(set(keys)))

    old = json.load(open(a.old))
    old_by = {}
    for r in old:
        v = s.vid_of(r) if r["images"] else next(k for k, x in zip(keys, out) if x["messages"] == r["messages"])
        old_by[v] = r
    same = sum(old_by[v] == x for v, x in zip(keys, out) if v in old_by)
    new = [v for v in keys if v not in old_by]
    assert same == len(old_by) == 583, (same, len(old_by))
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    json.dump(keys, open(a.out.replace(".json", "_ids.json"), "w"))
    print(json.dumps(dict(records=len(out), identical_to_old=same, added=new,
                          added_src={v: src[v] for v in new}, added_n_images={v: len(out[keys.index(v)]["images"]) for v in new},
                          added_est_tokens={v: int(s.mk["est"](out[keys.index(v)])) for v in new},
                          zero_image_clips=[v for v, x in zip(keys, out) if not x["images"]]), indent=1))
