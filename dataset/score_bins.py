#!/usr/bin/env python3
"""Score budget-bin predictions: are the tokens emitted, and do they behave?

Works on both LLaMA-Factory's generated_predictions.jsonl (free-running) and
teacher_forced_predictions.jsonl. Reports, per AD line:

  emitted      - did the line carry any budget token at all (the failure we keep hitting)
  coverage     - share of words that got one
  monotone     - bins never increase as the sentence runs on (time only moves forward)
  lands        - final word is in bin 0, i.e. the description uses up its gap
  bin MAE      - |predicted bin - reference bin| where both exist, in bin units
"""
import json, re, sys, argparse
from collections import Counter

TOK = re.compile(r"<\|budget_(\d)\|>")
AD = re.compile(r"^\s*[\d:.]+\s*\[AD\]\s*(.*)$")


def ad_bodies(text):
    return [m.group(1) for m in (AD.match(l) for l in text.split("\n")) if m]


def bins_and_words(body):
    """Bins in order, plus the word count (tokens + untagged words)."""
    bins = [int(b) for b in TOK.findall(body)]
    # strip the tokens, then count what is left: every word should carry one
    words = len([w for w in TOK.sub(" ", body).split() if w])
    return bins, words


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        pred = d.get("predict") or d.get("prediction") or d.get("output") or ""
        ref = d.get("label") or d.get("reference") or d.get("target") or ""
        rows.append((pred, ref))
    return rows


def analyse(rows, name):
    n_ad = n_emit = n_mono = n_land = 0
    cov_n = cov_d = 0
    dist = Counter()
    mae_s = mae_n = 0
    uniq = set()

    for pred, ref in rows:
        pb = ad_bodies(pred)
        rb = ad_bodies(ref)
        for i, body in enumerate(pb):
            n_ad += 1
            uniq.add(body.strip())
            bins, nw = bins_and_words(body)
            if bins:
                n_emit += 1
                dist.update(bins)
                cov_n += len(bins); cov_d += max(nw, 1)
                if all(b <= a for a, b in zip(bins, bins[1:])):
                    n_mono += 1
                if bins[-1] == 0:
                    n_land += 1
            if i < len(rb):
                gb, _ = bins_and_words(rb[i])
                for a, b in zip(bins, gb):
                    mae_s += abs(a - b); mae_n += 1

    p = lambda a, b: f"{a/b:6.1%}" if b else "   n/a"
    print(f"\n=== {name} ===")
    print(f"  AD lines                {n_ad}")
    print(f"  emitted a budget token  {p(n_emit, n_ad)}   <- the one that matters")
    print(f"  word coverage           {p(cov_n, cov_d)}")
    print(f"  monotone non-increasing {p(n_mono, n_emit)}")
    print(f"  ends in bin 0 (lands)   {p(n_land, n_emit)}   (reference: 25.9%)")
    print(f"  unique AD bodies        {p(len(uniq), n_ad)}   <- guards degenerate repetition")
    if mae_n:
        print(f"  bin MAE vs reference    {mae_s/mae_n:.3f} bins over {mae_n} aligned words")
    if dist:
        t = sum(dist.values())
        print("  predicted bin mix:      " +
              "  ".join(f"{k}:{dist[k]/t:.0%}" for k in sorted(dist)))
    return dict(n_ad=n_ad, emitted=n_emit, mono=n_mono, land=n_land)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    for pth in a.paths:
        try:
            analyse(load(pth), pth.split("/")[-2] + "/" + pth.split("/")[-1])
        except FileNotFoundError:
            print(f"\n=== {pth} === MISSING")
