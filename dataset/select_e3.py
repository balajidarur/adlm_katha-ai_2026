#!/usr/bin/env python3
"""E3 / M8: pick M3's clips from a rebuild with other dense descriptions (here Qwen3-VL-8B).

train  = M3's bins_f150_40k clips, in M3's order (that list is re-derived from M3's own build: gapframes_bins_train ->
         mk150 cap -> est <= 40k, and must come out at 7,763). Each E3 record gets the same cap; a clip whose E3
         record is over 40k (longer dense descriptions) or is missing from the rebuild is dropped and reported
         (same rule as select_like_m3.py).
eval   = all 591 CMD-AD eval clips in T2's M3-591 order (eval527 order, then the other 64 in eval_ids order), records
         uncapped exactly like M3's eval rows (eval527.json == the uncapped cap-200/48k gapframes_bins_eval records):
         every built eval clip goes through add_bins' convert() with no drop (T2's group C rule), and the clips the
         full build cannot make (T2 groups A/B: no or truncated video) come from --extra (built from T2's inputs).
         Nothing is dropped; clips over 40k est. tokens are reported (eval cutoff_len 65536 as for M3-591).

  python select_e3.py --name q3vl8b --v <rebuild dir> --m3 <M3 build dir> --eval527 <M3 eval527.json>
                      --eval-ids eval_ids.txt [--extra extra.json ...] --out <dir>
"""
import argparse, copy, hashlib, json, os

import add_bins_to_gapframes as ab

HERE = os.path.dirname(os.path.abspath(__file__))
mk = {}
exec(open(f"{HERE}/mk150.py").read().split("d = json.load")[0], mk)
BUDGET = 40000


def vid_of(rec):
    return rec["images"][0].split("/")[-2] if rec["images"] else rec["messages"][-1]["content"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def m3_train_list(m3):
    d = json.load(open(f"{m3}/gapframes_bins_train.json"))
    out = []
    for r in d:
        mk["cap"](r)
        if mk["est"](r) <= BUDGET:
            out.append(vid_of(r))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("name", "v", "m3", "eval527", "eval-ids", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = {}

    # ---- train
    ref = m3_train_list(a.m3)
    print(f"M3 bins_f150_40k re-derived: {len(ref)} clips (expected 7,763)")
    assert len(ref) == 7763, len(ref)
    by = {vid_of(r): r for r in json.load(open(f"{a.v}/gapframes_bins_train.json"))}
    tr, missing, over = [], [], []
    for v in ref:
        if v not in by:
            missing.append(v)
            continue
        x = by[v]
        mk["cap"](x)
        if mk["est"](x) > BUDGET:
            over.append(v)
            continue
        tr.append(x)
    rep["train"] = dict(m3=len(ref), kept=len(tr), missing_in_rebuild=missing, over_40k_dropped=over)
    print(f"train: {len(tr)} (M3 {len(ref)}) | missing {len(missing)} | over 40k dropped {len(over)}")

    # ---- eval
    budget = ab.load_budget()
    recs = json.load(open(f"{a.v}/cmdad_dds_gapframes_eval.json"))
    meta = json.load(open(f"{a.v}/cmdad_dds_gapframes_eval_meta.json"))
    ev = {}
    nbin = {}
    for rec, mt in zip(recs, meta):
        rec = copy.deepcopy(rec)
        n, _ = ab.convert(rec, mt["video_id"], budget)
        assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])
        ev[mt["video_id"]] = rec
        nbin[mt["video_id"]] = n
    src = {v: "full_build" for v in ev}
    for f in a.extra:
        for r in json.load(open(f)):
            ev[vid_of(r)] = r
            src[vid_of(r)] = os.path.basename(f)
    e527 = [vid_of(r) for r in json.load(open(a.eval527))]
    ids = [l.strip().split("/")[1] for l in open(a.eval_ids) if l.strip()]
    order = e527 + [v for v in ids if v not in set(e527)]
    assert len(order) == 591 and set(order) == set(ids), (len(order), len(set(ids)))
    miss_ev = [v for v in order if v not in ev]
    est = {v: mk["est"](ev[v]) for v in order if v in ev}
    rep["eval591"] = dict(clips=len(order) - len(miss_ev), missing=miss_ev,
                          over_40k_kept=[v for v in order if est.get(v, 0) > BUDGET],
                          zero_binned_kept=[v for v in order if nbin.get(v, 1) == 0],
                          from_extra=sorted(v for v, s in src.items() if s != "full_build"))
    print(f"eval591: {len(order) - len(miss_ev)} / 591 | missing {miss_ev} | over 40k (kept) "
          f"{len(rep['eval591']['over_40k_kept'])}")

    files = {f"{a.name}_f150_40k": tr}
    if not miss_ev:
        files[f"{a.name}_eval591"] = [ev[v] for v in order]
        files[f"{a.name}_eval527"] = [ev[v] for v in e527]
        json.dump(order, open(f"{a.out}/{a.name}_eval591_order.json", "w"))
    m3_info = json.load(open(f"{a.m3}/dataset_info.json"))
    tmpl = m3_info["cmdad_dds_gapframes_train"]
    info = {}
    for key, rs in files.items():
        json.dump(rs, open(f"{a.out}/{key}.json", "w"), ensure_ascii=False)
        info[key] = dict(tmpl, file_name=f"{key}.json")
    json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    json.dump(rep, open(f"{a.out}/select_report.json", "w"), indent=1)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in [f"{k}.json" for k in files] + ["dataset_info.json"]:
            f.write(f"{sha(f'{a.out}/{fn}')}  {fn}\n")
    print(open(f"{a.out}/SHA256SUMS").read())
