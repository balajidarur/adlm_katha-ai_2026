#!/usr/bin/env python3
"""Input ablations A2 / A3 / A4 (EXPERIMENTS.md) built from the M3 records (Qwen3.8 DDs, 0.5 s budget bins).

Only the JSON records change - no frame on disk is touched, clips and their order are identical to M3:
  A2  without gap frames : `[frame M:SS.s] <image>` lines and their images removed; the intro no longer mentions frames.
  A3  without DDs        : dense-description lines removed, plus the DD-derived `## Locations` / `## Props` lists;
                           cast list, gap frames and dialogue kept; `## Visual events` renamed `## Frames`.
  A4  dialogue placeholders: every dialogue line keeps its start-end time, its words become `<dialogue>` in the input
                           and in the target (`M:SS.s-M:SS.s [DIALOGUE] <dialogue>`); ADs and budget tokens unchanged.
Targets are unchanged for A2 and A3.

  python make_ablations.py --q <dir with gapframes_bins_train_f150_40k.json, eval527.json, dataset_info.json> --out <dir>
Writes {a2_noframes,a3_nodds,a4_dlgph}_{f150_40k,eval527}.json, dataset_info.json and SHA256SUMS (the lock).
"""
import argparse, hashlib, json, os, re

FRAME = re.compile(r"^\[frame (\d+:\d\d\.\d)\] <image>$")
DD = re.compile(r"^\d+:\d\d\.\d  ")
DLG_IN = re.compile(r"^(\d+:\d\d\.\d-\d+:\d\d\.\d)  .*$")
DLG_OUT = re.compile(r"^(\d+:\d\d\.\d-\d+:\d\d\.\d) \[DIALOGUE\] .*$")
INTRO = ("You are given the visual events of a film clip, each with its start time, frames sampled at one frame per second "
         "from the stretches between dialogue lines, and every line of dialogue in it. Each frame is labelled `[frame M:SS.s]` "
         "with the time it was taken, and each dialogue line with its start and end time, `M:SS.s-M:SS.s`.")
PH = "<dialogue>"
RULE = "- Reproduce every dialogue line exactly, with its start and end time."


def no_frames(u, a, imgs):      # A2
    assert INTRO in u
    u = u.replace(INTRO, "You are given the visual events of a film clip, each with its start time, and every line of "
                         "dialogue in it, each with its start and end time, `M:SS.s-M:SS.s`.")
    return "\n".join(l for l in u.split("\n") if not FRAME.match(l)), a, []


def no_dds(u, a, imgs):         # A3
    assert INTRO in u
    u = u.replace(INTRO, "You are given frames sampled at one frame per second from the stretches between dialogue "
                         "lines of a film clip, and every line of dialogue in it. Each frame is labelled `[frame M:SS.s]` "
                         "with the time it was taken, and each dialogue line with its start and end time, `M:SS.s-M:SS.s`.")
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)), a, imgs


def dlg_placeholder(u, a, imgs):  # A4
    assert INTRO in u and RULE in u
    u = u.replace(INTRO, "You are given the visual events of a film clip, each with its start time, frames sampled at one "
                         "frame per second from the stretches between dialogue lines, and the time slot of every line of "
                         "dialogue in it. Each frame is labelled `[frame M:SS.s]` with the time it was taken, and each "
                         f"dialogue slot with its start and end time, `M:SS.s-M:SS.s`; the words are withheld and shown as `{PH}`.")
    u = u.replace(RULE, f"- Reproduce every dialogue slot as `M:SS.s-M:SS.s [DIALOGUE] {PH}`, with its start and end time.")
    out, sec = [], None
    for l in u.split("\n"):
        if l.startswith("## "):
            sec = l
        m = DLG_IN.match(l) if sec == "## Dialogue" else None
        out.append(f"{m.group(1)}  {PH}" if m else l)
    a = "\n".join(f"{m.group(1)} [DIALOGUE] {PH}" if (m := DLG_OUT.match(l)) else l for l in a.split("\n"))
    return "\n".join(out), a, imgs


ABL = {"a2_noframes": no_frames, "a3_nodds": no_dds, "a4_dlgph": dlg_placeholder}


def apply(fn, recs):
    out = []
    for r in recs:
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        u2, a2, im2 = fn(u, a, r["images"])
        assert u2.count("<image>") == len(im2)
        out.append(dict(r, messages=[dict(r["messages"][0], content=u2), dict(r["messages"][1], content=a2)], images=im2))
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(f"{a.q}/dataset_info.json"))
    src = {"f150_40k": ("gapframes_bins_train_f150_40k.json", "bins_f150_40k"), "eval527": ("eval527.json", "eval527")}
    new_info, files = {}, []
    for split, (fn_src, key) in src.items():
        recs = json.load(open(f"{a.q}/{fn_src}"))
        for name, fn in ABL.items():
            out = apply(fn, recs)
            f = f"{name}_{split}.json"
            json.dump(out, open(f"{a.out}/{f}", "w"), ensure_ascii=False)
            new_info[f"{name}_{split}"] = dict(info[key], file_name=f)
            files.append(f)
            n_img = sum(len(r["images"]) for r in out)
            n_dd = sum(sum(bool(DD.match(l)) for l in r["messages"][0]["content"].split("\n")) for r in out)
            n_ph = sum(r["messages"][1]["content"].count(PH) for r in out)
            print(f"{f}: {len(out)} clips | frames {n_img:,} | DD lines {n_dd:,} | target placeholders {n_ph:,}", flush=True)
    json.dump(new_info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    files.append("dataset_info.json")
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in files:
            f.write(f"{sha(f'{a.out}/{fn}')}  {fn}\n")
    print(open(f"{a.out}/SHA256SUMS").read())
