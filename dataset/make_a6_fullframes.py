#!/usr/bin/env python3
"""A6: no dense descriptions, frames from the WHOLE clip at 1 fps (not only the dialogue gaps), at most 150.

Built from variant C (`cmdad_dds_frames`: DDs + every 1 fps frame of the clip, Qwen3.8 DDs, same build as M3) after
add_bins_to_gapframes.py (0.5 s bins, identical targets to M3). Per record:
  - dense-description lines and the DD-derived `## Locations` / `## Props` lists are removed exactly as in A3
    (make_ablations.no_dds); the `## Characters` list, frames and dialogue are kept; `## Visual events` -> `## Frames`;
  - frames are capped at 150 with mk150.cap() (thinned evenly per run of consecutive frames);
  - clips: exactly M3's (bins_f150_40k train / eval527), in M3's order; train clips over 40k tokens are dropped.
So A6 differs from A3 only in which frames are shown (whole clip vs dialogue gaps).

  python make_a6_fullframes.py --bins-dir <dir with cmdad_dds_frames_bins_{train,eval}.json> --m3-dir <qwen38 dir> --out <dir>
Writes a6_fullframes_{f150_40k,eval527}.json, dataset_info.json and SHA256SUMS.
"""
import argparse, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
mk, sel = {}, {}
exec(open(f"{HERE}/mk150.py").read().split("d = json.load")[0], mk)
exec(open(f"{HERE}/select_like_m3.py").read().split('if __name__ == "__main__":')[0], sel)
DD = re.compile(r"^\d+:\d\d\.\d  ")
INTRO_C = ("You are given the visual events of a film clip, each with its start time, frames sampled from it at one frame "
           "per second, and every line of dialogue in it.")
INTRO_A6 = "You are given frames sampled from a film clip at one frame per second, and every line of dialogue in it."
BUDGET = 40000


def no_dds(r):
    u = r["messages"][0]["content"]
    assert INTRO_C in u, "not a variant-C record"
    u = u.replace(INTRO_C, INTRO_A6)
    out, sec = [], None
    for l in u.split("\n"):
        if l.startswith("## "):
            sec = l
        if sec in ("## Locations", "## Props"):
            continue
        if sec == "## Visual events":
            if l == "## Visual events":
                l = "## Frames"
            elif DD.match(l):
                continue
        out.append(l)
    r["messages"][0]["content"] = re.sub(r"\n{3,}", "\n\n", "\n".join(out))
    return r


def pick(src, ref, drop_over):
    by_vid = {sel["vid_of"](r): r for r in json.load(open(src))}
    out, missing, over = [], 0, 0
    refs = json.load(open(ref))
    for r in refs:
        x = by_vid.get(sel["vid_of"](r))
        if x is None:
            missing += 1; continue
        assert x["messages"][1]["content"] == r["messages"][1]["content"], "target differs from M3"
        x = no_dds(x); mk["cap"](x)
        if mk["est"](x) > BUDGET:
            over += 1
            if drop_over:
                continue
        out.append(x)
    print(f"{os.path.basename(ref)}: {len(out)} clips (M3 {len(refs)}) | missing {missing} | over {BUDGET:,} {over}"
          f"{' dropped' if drop_over else ' kept'} | mean frames {sum(len(r['images']) for r in out) / max(1, len(out)):.1f}",
          flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("bins-dir", "m3-dir", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = {"a6_fullframes_f150_40k": pick(f"{a.bins_dir}/cmdad_dds_frames_bins_train.json",
                                            f"{a.m3_dir}/gapframes_bins_train_f150_40k.json", True),
             "a6_fullframes_eval527": pick(f"{a.bins_dir}/cmdad_dds_frames_bins_eval.json", f"{a.m3_dir}/eval527.json", False)}
    tmpl = next(iter(json.load(open(f"{a.m3_dir}/dataset_info.json")).values()))
    info = {}
    for k, recs in files.items():
        json.dump(recs, open(f"{a.out}/{k}.json", "w"), ensure_ascii=False)
        info[k] = dict(tmpl, file_name=f"{k}.json")
    json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in [f"{k}.json" for k in files] + ["dataset_info.json"]:
            f.write(f"{sel['sha'](f'{a.out}/{fn}')}  {fn}\n")
