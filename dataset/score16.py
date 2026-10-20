"""Score the 16-clip frames run: reproduction, bin behaviour, and dialogue leakage."""
import json, re, sys
from collections import Counter

AD = re.compile(r"^\s*([\d:.]+)\s*\[AD\]\s*(.*)$")
TOK = re.compile(r"<\|budget_(\d)\|>")
TOKA = re.compile(r"<\|budget_\d\|>")
PAD = re.compile(r"(<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>)+")


def clean(t):
    c = t.find("<|im_end|>")
    return t[:c] if c >= 0 else t


def ads(t):
    return [(m.group(1), PAD.sub("", m.group(2)).strip())
            for m in (AD.match(l) for l in t.split("\n")) if m]


def run(tag, path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n_ad = w_ok = b_ok = f_ok = 0
    ref_ad = 0
    dlg = leak = 0
    mae_s = mae_n = 0
    mono = emit = 0
    dist = Counter()
    for r in rows:
        pred, ref = clean(r["predict"]), r["label"]
        P, R = ads(pred), ads(ref)
        ref_ad += len(R)
        pm = {t: b for t, b in P}
        for t, rb in R:
            if t not in pm:
                continue
            pb = pm[t]
            n_ad += 1
            if TOKA.sub("", pb).split() == TOKA.sub("", rb).split():
                w_ok += 1
            pbin = [int(x) for x in TOK.findall(pb)]
            rbin = [int(x) for x in TOK.findall(rb)]
            if pbin == rbin:
                b_ok += 1
            if pb == rb:
                f_ok += 1
            if pbin:
                emit += 1
                dist.update(pbin)
                if all(y <= x for x, y in zip(pbin, pbin[1:])):
                    mono += 1
            for x, y in zip(pbin, rbin):
                mae_s += abs(x - y); mae_n += 1
        for l in pred.split("\n"):
            if "[DIALOGUE]" in l:
                dlg += 1
                if "<|budget_" in l:
                    leak += 1
    p = lambda a, b: f"{a/b:6.1%}" if b else "   n/a"
    print(f"\n=== {tag} ===")
    print(f"  AD lines matched by timestamp  {n_ad}/{ref_ad}  ({n_ad/max(ref_ad,1):.0%} recovered)")
    print(f"  exact words                    {p(w_ok, n_ad)}")
    print(f"  exact bins                     {p(b_ok, n_ad)}")
    print(f"  exact full line                {p(f_ok, n_ad)}")
    print(f"  bin MAE vs reference           {mae_s/max(mae_n,1):.3f} over {mae_n} words")
    print(f"  monotone non-increasing        {p(mono, emit)}")
    print(f"  DIALOGUE lines with budget tok {leak}/{dlg}  {p(leak, dlg)}  <- should be 0%")
    if dist:
        t = sum(dist.values())
        print("  predicted bin mix:             " + "  ".join(f"{k}:{dist[k]/t:.0%}" for k in sorted(dist)))


for tag, pth in [("TEACHER-FORCED", "/data/cmd_ad/runs/bins_frames_16/pred_tf/teacher_forced_predictions.jsonl"),
                 ("FREE-RUNNING", "/data/cmd_ad/runs/bins_frames_16/pred_free/generated_predictions.jsonl")]:
    try:
        run(tag, pth)
    except FileNotFoundError:
        print(f"\n=== {tag} === MISSING")
