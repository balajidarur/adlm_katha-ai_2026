#!/usr/bin/env python3
"""Assemble causal free-running tracks from a (possibly unfinished) cf_box.sh run.

  python cf_assemble.py <run dir: meta.json + g<k>/stream/generated.rank0.jsonl> <eval records json> <out.jsonl>

Only clips whose every gap record has a streamed row are assembled. Per gap, the kept ADs are those whose start t satisfies
gap start <= t < gap end (the last gap: t >= gap start); everything else the run wrote is ignored. One output row per
completed clip: {clip, video_id, n_gaps, gaps: [{gap, start, end, length, is_last, n_ads_out, kept: [{t, text, b0}]}],
track: "<script>"} where track is the kept ADs interleaved with the real dialogue lines (dialogue first at equal times), in
the model's line format with budget tokens kept, so it can be scored like a normal free-running predict. Also
"last_gap_raw" (the full-prompt run's output, for comparing with the normal free-running file; None if the last gap was not run).
Gaps without a record (skipped as short / overlapping) contribute no ADs.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_causal import DLG_P, sec, split_prompt  # noqa: E402

AD = re.compile(r"^(\d+):(\d\d\.\d) \[AD\] ?(.*)$")
BT = re.compile(r"<\|budget_(\d)\|>")


def clean(p):
    return p.split("<|im_end|>")[0].replace("<|endoftext|>", "").rstrip()


def main():
    rd, src, out = sys.argv[1:4]
    M = json.load(open(f"{rd}/meta.json"))
    E = json.load(open(src))
    S = {}
    for k in range(M["shards"]):
        f = f"{rd}/g{k}/stream/generated.rank0.jsonl"
        S[k] = {}
        if os.path.exists(f):
            for l in open(f):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:                      # a partially synced last line
                    continue
                S[k][r["idx"]] = r["predict"]
    by = {}
    for r in M["rows"]:
        if r["srow"] is not None:
            by.setdefault(r["clip"], []).append(r)
    n = 0
    with open(out, "w") as fo:
        for c in sorted(by):
            rows = sorted(by[c], key=lambda r: r["gap"])
            if not all(r["srow"] in S[r["shard"]] for r in rows):
                continue
            rec = E[c]
            assert rows[0]["video_id"] in rec["images"][0], (c, rows[0]["video_id"])
            _, _, _, dlg = split_prompt(rec["messages"][0]["content"])
            gaps, lines, last_raw = [], [], None
            for r in rows:
                p = clean(S[r["shard"]][r["srow"]])
                A = []
                for l in p.split("\n"):
                    m = AD.match(l.strip())
                    if m:
                        A.append((round(int(m.group(1)) * 60 + float(m.group(2)), 1), l.strip(), m.group(3)))
                k = [a for a in A if r["start"] <= a[0] < r["end"] or (r["is_last"] and a[0] >= r["start"])]
                gaps.append({"gap": r["gap"], "start": r["start"], "end": r["end"], "length": r["length"], "is_last": r["is_last"],
                             "n_ads_out": len(A), "kept": [{"t": t, "text": re.sub(r"\s+", " ", BT.sub("", x)).strip(),
                                                            "b0": (BT.findall(x) or [None])[0]} for t, _, x in k]})
                lines += [(t, 1, l) for t, l, _ in k]
                if r["is_last"]:
                    last_raw = p
            for x in dlg:
                m = DLG_P.match(x)
                lines.append((sec(m, 1), 0, f"{m.group(1)}:{m.group(2)}-{m.group(3)}:{m.group(4)} [DIALOGUE] {m.group(5)}"))
            track = "\n".join(l for _, _, l in sorted(lines, key=lambda z: (z[0], z[1])))
            fo.write(json.dumps({"clip": c, "video_id": rows[0]["video_id"], "n_gaps": len(rows), "gaps": gaps, "track": track,
                                 "last_gap_raw": last_raw}, ensure_ascii=False) + "\n")
            n += 1
    print("completed clips", n, "of", len(by), "streamed rows", sum(len(v) for v in S.values()), "->", out)


if __name__ == "__main__":
    main()
