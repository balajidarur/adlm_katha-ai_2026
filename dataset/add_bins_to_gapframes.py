#!/usr/bin/env python3
"""Add budget-bin tokens to the EXISTING gap-frames dataset.

Reuses cmdad_dds_gapframes_*.json exactly as exp-B produced it - same frames,
same `<image>` placement, same prompt - and changes only two things:

  * the assistant turn: every AD word gains a `<|budget_N|>` token saying how
    much silence remains before the next line of dialogue
  * the user turn: a legend block explaining the nine tokens

Nothing is re-extracted and no image is touched.
"""
import argparse, json, os, re

BUDGET_PARQUET = "/data/cmd_ad/annotations/ad_prefix_durations.parquet"
SPEED_LO, SPEED_HI = 0.67, 1.5
BIN_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
BIN_TOKENS = [f"<|budget_{i}|>" for i in range(9)]
BIN_LABELS = ["under 0.5", "0.5 to 1", "1 to 1.5", "1.5 to 2", "2 to 2.5",
              "2.5 to 3", "3 to 3.5", "3.5 to 4", "over 4"]
BIN_LEGEND = (
    "## Time budget tokens\n"
    "Each word of an audio description is preceded by one token saying how many "
    "seconds of silence remain before the next line of dialogue begins:\n"
    + "\n".join(f"- `{t}` : {l} seconds left" for t, l in zip(BIN_TOKENS, BIN_LABELS))
    + "\n`<|budget_0|>` means there is almost no time left - finish the sentence now.\n"
      "Emit the token that matches the time remaining, then the word."
)

AD_RE = re.compile(r"^(\d+:\d\d\.\d) \[AD\] (.*)$")
DLG_RE = re.compile(r"^(\d+:\d\d\.\d)-(\d+:\d\d\.\d) \[DIALOGUE\]")


def bin_of(sec):
    for i, e in enumerate(BIN_EDGES):
        if sec <= e:
            return i
    return len(BIN_EDGES)


def parse_ts(s):
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def load_budget():
    import pandas as pd
    df = pd.read_parquet(BUDGET_PARQUET)
    df = df[df.speed_factor.between(SPEED_LO, SPEED_HI)]
    out = {}
    for r in df.itertuples(index=False):
        vid = r.cmd_filename.split("/")[-1]
        pref = [float(x) for x in r.prefix_duration_s]
        out.setdefault(vid, []).append((float(r.scaled_start), list(r.words), pref[1:]))
    for v in out:
        out[v].sort()
    return out


def lookup(budget, vid, t, tol=0.07):
    """Nearest AD start for this clip within tol seconds.

    The dataset renders timestamps at 0.1 s, so an exact key match against the
    parquet's full-precision scaled_start silently misses almost everything.
    """
    rows = budget.get(vid)
    if not rows:
        return None
    best, bd = None, tol
    for st, words, spoken in rows:
        d = abs(st - t)
        if d <= bd:
            best, bd = (words, spoken), d
    return best


def convert(rec, vid, budget):
    lines = rec["messages"][1]["content"].split("\n")
    dlg_starts = sorted(parse_ts(m.group(1)) for m in
                        (DLG_RE.match(l) for l in lines) if m)
    out, n_ok, n_words = [], 0, 0
    for line in lines:
        m = AD_RE.match(line)
        if not m:
            out.append(line)
            continue
        t, text = parse_ts(m.group(1)), m.group(2)
        ent = lookup(budget, vid, t)
        if ent is None:
            out.append(line)
            continue
        words, spoken = ent
        if " ".join(words).strip() != text.strip():
            words = text.split()
            if len(words) != len(spoken):
                out.append(line)
                continue
        nxt = [a for a in dlg_starts if a >= t]
        gap = (nxt[0] - t) if nxt else None
        if gap is None:
            out.append(line)
            continue
        body = "".join(f"{BIN_TOKENS[bin_of(gap - p)]}{w} " for w, p in zip(words, spoken)).strip()
        out.append(f"{m.group(1)} [AD] {body}")
        n_ok += 1
        n_words += len(words)
    rec["messages"][1]["content"] = "\n".join(out)
    rec["messages"][0]["content"] = BIN_LEGEND + "\n\n" + rec["messages"][0]["content"]
    return n_ok, n_words


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    budget = load_budget()
    print(f"budget: {sum(len(v) for v in budget.values()):,} rows over {len(budget):,} clips")
    data = json.load(open(a.src))
    meta = json.load(open(a.meta))
    assert len(data) == len(meta), f"{len(data)} records vs {len(meta)} meta"

    keep, tot_ad, tot_w = [], 0, 0
    for rec, mt in zip(data, meta):
        vid = mt.get("video_id") or mt.get("vid")
        n, w = convert(rec, vid, budget)
        if n:
            # the image contract must survive the edit
            assert rec["messages"][0]["content"].count("<image>") == len(rec["images"]), vid
            keep.append(rec)
            tot_ad += n
            tot_w += w
        if a.limit and len(keep) >= a.limit:
            break
    json.dump(keep, open(a.out, "w"), ensure_ascii=False)
    print(f"wrote {len(keep)} clips | {tot_ad} AD lines binned | {tot_w} budget tokens "
          f"| {sum(len(r['images']) for r in keep)} frames -> {a.out}")
