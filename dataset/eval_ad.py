#!/usr/bin/env python3
"""CIDEr for audio descriptions, two ways.

SEGMENT-LEVEL (AutoAD convention)
    One (reference, generated) pair per AD, matched by the ground-truth
    timestamp. This is what teacher-forced decoding produces: the model is
    handed each AD's start time and only writes the text, so the score measures
    description quality in isolation from timing.

DENSE-CAPTIONING
    The model picks its own timestamps, so predictions must first be matched to
    references in time. An AD is a point event, not a span, so matching uses a
    time window rather than tIoU. Unmatched references count as empty
    predictions and unmatched predictions are counted as false positives -
    otherwise a model that stays silent, or one that floods the timeline,
    both score well by accident.

Two traps worth knowing, both learned the hard way on this dataset:
  * The IDF vocabulary must come from the WHOLE CMD-AD corpus (101,267 ADs).
    Building it from the few hundred references under test leaves common AD
    phrasing over-weighted.
  * CIDEr rewards degenerate output here. A single generic sentence repeated
    for every AD once scored 42.0 - above a real model's 28.4 - because it
    partially matches many references. Always read `unique_frac` alongside the
    score; a low value invalidates the number.
"""
import argparse, csv, json, re, sys
from collections import defaultdict

sys.path.insert(0, "/storage/users/dbalu/adqa")
sys.path.insert(0, "/storage/users/dbalu/pylibs")
from cider.cider import Cider

ANNO_CSV = "/storage/datasets/CMD-AD/data/cmd_ad_anno_v1.csv"
AD_RE = re.compile(r"^\s*(\d+:\d\d\.\d)\s*\[AD\]\s*(.*)$")
BUDGET_RE = re.compile(r"<\|budget_\d\|>")
PAD_RE = re.compile(r"(<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>)+")


def norm(s):
    """Strip budget tokens and padding, lowercase, collapse whitespace."""
    s = BUDGET_RE.sub(" ", PAD_RE.sub(" ", s))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def parse_ts(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def ads_of(text):
    """[(seconds, normalised text)] for every [AD] line."""
    cut = text.find("<|im_end|>")
    if cut >= 0:
        text = text[:cut]
    out = []
    for line in text.split("\n"):
        m = AD_RE.match(line)
        if m:
            body = norm(m.group(2))
            if body:
                out.append((parse_ts(m.group(1)), body))
    return out


def corpus_idf(split=None):
    """Every AD in CMD-AD, for document frequencies."""
    out = []
    with open(ANNO_CSV) as f:
        for r in csv.DictReader(f):
            if split and r.get("split") != split:
                continue
            t = norm(r["text"])
            if t:
                out.append(t)
    return out


def cider_score(pairs, idf_corpus):
    """pairs: [(reference, candidate)]. Returns (mean CIDEr, per-pair scores)."""
    if not pairs:
        return 0.0, []
    gts = {str(i): [r] for i, (r, _) in enumerate(pairs)}
    res = {str(i): [c if c else " "] for i, (_, c) in enumerate(pairs)}
    scorer = Cider()
    if hasattr(scorer, "init_vocabulary") and idf_corpus:
        scorer.init_vocabulary([[t] for t in idf_corpus])
    score, scores = scorer.compute_score(gts, res)
    return float(score), list(scores)


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            d = json.loads(line)
            rows.append((d.get("predict", ""), d.get("label", "")))
    return rows


def segment_level(rows, idf):
    """Match by reference order - teacher forcing preserves it one-for-one."""
    pairs, uniq = [], set()
    missing = 0
    for pred, ref in rows:
        P, R = ads_of(pred), ads_of(ref)
        pmap = {round(t, 1): b for t, b in P}
        for t, rb in R:
            cb = pmap.get(round(t, 1), "")
            if not cb:
                missing += 1
            else:
                uniq.add(cb)
            pairs.append((rb, cb))
    s, _ = cider_score(pairs, idf)
    n = len(pairs)
    return {"metric": "segment-level CIDEr", "cider": round(s, 2), "pairs": n,
            "missing_pred": missing, "missing_frac": round(missing / max(n, 1), 3),
            "unique_frac": round(len(uniq) / max(n - missing, 1), 3)}


def dense(rows, idf, window):
    """Greedy nearest-in-time matching within `window` seconds."""
    pairs, uniq = [], set()
    tp = fp = fn = 0
    for pred, ref in rows:
        P, R = ads_of(pred), ads_of(ref)
        used = set()
        for t, rb in R:
            best, bd = None, window
            for j, (pt, pb) in enumerate(P):
                if j in used:
                    continue
                d = abs(pt - t)
                if d <= bd:
                    best, bd = j, d
            if best is None:
                pairs.append((rb, ""))
                fn += 1
            else:
                used.add(best)
                pairs.append((rb, P[best][1]))
                uniq.add(P[best][1])
                tp += 1
        fp += len(P) - len(used)
    s, _ = cider_score(pairs, idf)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"metric": f"dense CIDEr @{window}s", "cider": round(s, 2),
            "matched": tp, "false_pos": fp, "missed": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "unique_frac": round(len(uniq) / max(tp, 1), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="generated_predictions.jsonl or teacher_forced_predictions.jsonl")
    ap.add_argument("--mode", choices=["segment", "dense", "both"], default="both")
    ap.add_argument("--windows", default="1,2,5", help="dense matching windows, seconds")
    ap.add_argument("--idf", choices=["cmdad", "train", "self"], default="cmdad")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    rows = load(a.preds)
    idf = [] if a.idf == "self" else corpus_idf(None if a.idf == "cmdad" else "train")
    print(f"clips {len(rows)} | IDF corpus {len(idf):,} ADs ({a.idf})\n")

    out = []
    if a.mode in ("segment", "both"):
        out.append(segment_level(rows, idf))
    if a.mode in ("dense", "both"):
        for w in [float(x) for x in a.windows.split(",")]:
            out.append(dense(rows, idf, w))

    for r in out:
        head = f"{r.pop('metric'):<24} CIDEr {r.pop('cider'):>7.2f}"
        print(head + "   " + "  ".join(f"{k}={v}" for k, v in r.items()))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2)
