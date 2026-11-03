#!/usr/bin/env python3
"""Make a dense-description variant of M3 with exactly M3's clips.

Takes gapframes_bins_{train,eval}.json rebuilt with another dense-description source (same build, same frames,
same budget bins), applies M3's 150-frame cap (mk150.cap), and keeps the clips of M3's train set
(bins_f150_40k) and eval527, in M3's order, so only the dense descriptions differ. A clip that would exceed
M3's 40k token budget with the new descriptions is dropped and reported (train); eval keeps all 527 but
reports any above 40k (cutoff_len is 49,152).

  python select_like_m3.py --name dd4b --bins-dir <variant dir> --m3-dir <qwen38 dir> --out <dir>
Writes <name>_f150_40k.json, <name>_eval527.json, dataset_info.json and SHA256SUMS.
"""
import argparse, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
mk = {}
exec(open(f"{HERE}/mk150.py").read().split("d = json.load")[0], mk)
BUDGET = 40000


def vid_of(rec):
    return rec["images"][0].split("/")[-2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pick(src, ref, name, drop_over):
    by_vid = {}
    for r in json.load(open(src)):
        if r["images"]:
            by_vid[vid_of(r)] = r
    out, missing, over = [], [], []
    for r in json.load(open(ref)):
        v = vid_of(r)
        if v not in by_vid:
            missing.append(v)
            continue
        x = by_vid[v]
        mk["cap"](x)
        if mk["est"](x) > BUDGET:
            over.append(v)
            if drop_over:
                continue
        out.append(x)
    print(f"{name}: {len(out)} clips (M3 {len(json.load(open(ref)))}) | missing in variant {len(missing)} "
          f"| over {BUDGET:,} tokens {len(over)}{' (dropped)' if drop_over else ' (kept)'}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("name", "bins-dir", "m3-dir", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr = pick(f"{a.bins_dir}/gapframes_bins_train.json", f"{a.m3_dir}/gapframes_bins_train_f150_40k.json", "train", True)
    ev = pick(f"{a.bins_dir}/gapframes_bins_eval.json", f"{a.m3_dir}/eval527.json", "eval527", False)
    files = {f"{a.name}_f150_40k": tr, f"{a.name}_eval527": ev}
    m3_info = json.load(open(f"{a.m3_dir}/dataset_info.json"))
    tmpl = next(iter(m3_info.values()))
    info = {}
    for key, recs in files.items():
        json.dump(recs, open(f"{a.out}/{key}.json", "w"), ensure_ascii=False)
        info[key] = dict(tmpl, file_name=f"{key}.json")
    json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in [f"{k}.json" for k in files] + ["dataset_info.json"]:
            f.write(f"{sha(f'{a.out}/{fn}')}  {fn}\n")
    print(open(f"{a.out}/SHA256SUMS").read())
