"""[MADX: T3 copy + --dd-tag (DD source model dir); otherwise unchanged]
[T3 MAD-ZS copy of rc_build.py: GT rows carry a 4th field (anno_idx); dataset names mad_*]
Writer datasets for the challenge windows, built exactly like reframed_{bins,nobudget}_f150_999k_test:

  1. build_frame_datasets.py (CMD-AD builder, unchanged) via frames_lib monkey-patched as rf_build.py does:
     clips -> challenge windows, video -> window mp4, dialogue -> window SRT, dense -> Qwen3.8 desc_action_v2 DDs,
     GT ADs -> US AD sentences of the window (only used as the teacher-forcing target / label).
  2. rf_bins.py steps for a val/test split: no word times, so ADs stay plain text; the budget-bin legend is prepended
     to every prompt; frames capped at 150 (mk150.cap, even_cap fallback); no clip dropped (999k token budget).
  3. rf_nobudget.py: remove the legend (and any budget token) -> nobudget copy.

  python rc_build.py --code-a /scratch/codeA --root /scratch/rc_data --win /scratch/rc_win --videos /scratch/rcw \
      --dd /scratch/rc_dd_all --out /scratch/rc_ds --workers 44 [build_frame_datasets args...]
"""
import argparse
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
for k in ("code-a", "root", "win", "videos", "dd", "out"):
    ap.add_argument(f"--{k}", required=True)
ap.add_argument("--workers", type=int, default=44)
ap.add_argument("--dd-tag", default="Qwen3.8-27B-FP8", help="DD model dir under {dd}/{vid}/ (MADX: Qwen3.5-4B, Qwen3-VL-8B-Instruct)")
ap.add_argument("--stage", choices=["build", "post", "all"], default="all")
a, rest = ap.parse_known_args()

os.environ["DENSE_PATH_TMPL"] = f"{a.dd}/{{vid}}/{a.dd_tag}/adlm_descriptions.json"
sys.path.insert(0, f"{a.code_a}/dataset_prep")
os.chdir(f"{a.code_a}/dataset_prep")
import frames_lib as fl  # noqa: E402

W = json.load(open(f"{a.win}/windows.json"))
MOVIE = {w["wid"]: m for m, ws in W.items() for w in ws}
fl.ROOT = a.root
fl.load_index = lambda: [{"video_id": v, "year": m, "split": "challenge"} for v, m in sorted(MOVIE.items())]
fl.load_ads = lambda: {v: sorted((float(s), float(e), t) for s, e, t in (x[:3] for x in json.load(open(f"{a.win}/gt/{v}.json")))) for v in MOVIE}
fl.video_path = lambda year, vid: f"{a.videos}/{year}/{vid}.mp4"
fl.asr_path = lambda year, vid: f"{a.win}/srt/dialogue/{year}/{vid}.srt"
FRAMES = f"{a.out}/frames/1fps_448x252"

if a.stage in ("build", "all"):
    import build_frame_datasets as bfd  # noqa: E402
    sys.argv = [sys.argv[0], "--splits", "challenge", "--n-per-split", "0", "--out", a.out, "--frame-root", FRAMES,
                "--workers", str(a.workers)] + rest
    bfd.main()

if a.stage in ("post", "all"):
    import numpy as np
    ab, mk = {}, {}
    exec(open(f"{a.code_a}/add_bins_to_gapframes.py").read().split('if __name__ == "__main__":')[0], ab)
    exec(open(f"{a.code_a}/mk150.py").read().split("d = json.load")[0], mk)
    mk["MAX_F"] = 150

    def even_cap(rec, n):   # rf_bins.even_cap
        lines = rec["messages"][0]["content"].split("\n")
        idx = [i for i, l in enumerate(lines) if mk["FRAME_LINE"].match(l.strip())]
        keep = set(np.unique(np.linspace(0, len(idx) - 1, n).round().astype(int)).tolist())
        drop = {idx[k] for k in range(len(idx)) if k not in keep}
        rec["messages"][0]["content"] = "\n".join(l for i, l in enumerate(lines) if i not in drop)
        rec["images"] = [im for k, im in enumerate(rec["images"]) if k in keep]

    src = f"{a.out}/cmdad_dds_gapframes_challenge.json"
    data = json.load(open(src))
    info = json.load(open(f"{a.out}/dataset_info.json"))
    bins, nob = [], []
    for rec in data:
        # rf_bins.convert_clip_end on a split without word times: every AD line stays as it is, legend prepended
        rec["messages"][0]["content"] = ab["BIN_LEGEND"] + "\n\n" + rec["messages"][0]["content"]
        mk["cap"](rec)
        if len(rec["images"]) > 150:
            even_cap(rec, 150)
        assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])
        bins.append(rec)
        u, t = rec["messages"][0]["content"], rec["messages"][1]["content"]
        i = u.find("You are an audio description writer.")
        assert u[:i].startswith("## Time budget tokens") and "<|budget_" not in u[i:]
        nob.append({**rec, "messages": [{**rec["messages"][0], "content": u[i:]},
                                        {**rec["messages"][1], "content": re.sub(r"<\|budget_\d+\|>", "", t)}]})
    for name, recs in (("mad_bins_f150_999k_challenge", bins), ("mad_nobudget_f150_999k_challenge", nob)):
        json.dump(recs, open(f"{a.out}/{name}.json", "w"), ensure_ascii=False)
        info[name] = dict(info["cmdad_dds_gapframes_challenge"], file_name=f"{name}.json")
    json.dump(info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    meta = json.load(open(src.replace(".json", "_meta.json")))
    json.dump([m["video_id"] for m in meta], open(f"{a.out}/order.json", "w"))
    print("post:", len(bins), "clips; frames max", max(len(r["images"]) for r in bins))
