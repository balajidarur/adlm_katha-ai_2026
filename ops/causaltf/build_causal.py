#!/usr/bin/env python3
"""Causal (no-lookahead) teacher-forced records: one record per ground-truth AD.

For a clip with K ADs, record k is the clip's record truncated so the model cannot see past the end of the
gap AD k sits in. With t_k = AD k's start and G_k = start of the next dialogue line after AD k in the target
(the gap end; clip end -> no truncation when there is none):
  prompt  - header sections (budget legend, instructions, characters, locations, props) unchanged;
            "## Visual events": event lines with time <= G_k, frame lines with time <= G_k (images trimmed to
            match, same order); "## Dialogue": the lines that precede AD k in the target, plus exactly one
            more - the real next dialogue line (the one starting at G_k), so the model knows where the gap ends.
  target  - every target line up to and including AD k (standard TF: dialogue forced, earlier ADs prefix
            forced / body generated). The last AD of the TF output is the prediction for AD k.
Dialogue lines are taken in target order, so a dialogue line that starts at the same time as AD k is counted
as "before" or "after" exactly as the target orders it; ADs that start inside a dialogue line (target puts
the dialogue first) keep that line. Also writes the 5 untruncated records (kind "std") for a same-box standard
TF rerun.

  python build_causal.py <test.json> <out_dir> [n_records=5] [--per-gap]
  -> <out_dir>/causal.json, std.json, meta.json (one meta row per GT AD)

--per-gap: ADs in the same gap get the identical prompt, and TF is deterministic (an earlier same-gap AD inside a
later record reproduces its own record's prediction exactly: 140/140 checked on 5 CMD-AD clips), so only the last
AD of each gap is kept as a record; every AD of that gap is read from it (meta "row" = record index, "pos" = the
AD's position among the record output's ADs). ADs after the last dialogue line have the full prompt and a prefix
of the full target, i.e. they ARE standard TF (16/16 + 3/3 identical); they get no record ("row": null) and are
taken from the same-box standard-TF run.
"""
import json, re, sys

EV = re.compile(r"^(\d+):(\d\d\.\d)  \S")
FR = re.compile(r"^\[frame (\d+):(\d\d\.\d)\] <image>$")
DLG_P = re.compile(r"^(\d+):(\d\d\.\d)-(\d+):(\d\d\.\d)  (.*)$")
DLG_T = re.compile(r"^(\d+):(\d\d\.\d)-(\d+):(\d\d\.\d) \[DIALOGUE\] (.*)$")
AD_T = re.compile(r"^(\d+):(\d\d\.\d) \[AD\] ")


def sec(m, i=1):
    return round(int(m.group(i)) * 60 + float(m.group(i + 1)), 1)


def split_prompt(u):
    """header lines, visual-events lines, blank separator(s), dialogue lines; asserts the layout."""
    L = u.split("\n")
    iv, idl = L.index("## Visual events"), L.index("## Dialogue")
    assert iv < idl and L.count("## Visual events") == 1 and L.count("## Dialogue") == 1
    vis, tail = L[iv + 1:idl], L[idl + 1:]
    while vis and vis[-1] == "":
        vis = vis[:-1]
    sep = L[iv + 1 + len(vis):idl]
    for l in vis:
        assert EV.match(l) or FR.match(l), f"unexpected visual line: {l[:80]!r}"
    for l in tail:
        assert DLG_P.match(l), f"unexpected dialogue line: {l[:80]!r}"
    assert "<image>" not in "\n".join(L[:iv + 1]), "images outside the visual-events section"
    return L[:iv + 1], vis, sep, tail


def build(rec, ridx):
    u, a = rec["messages"][0]["content"], rec["messages"][1]["content"]
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    head, vis, sep, dlg = split_prompt(u)
    assert u.count("<image>") == len(rec["images"])
    ftimes = [sec(FR.match(l)) for l in vis if FR.match(l)]
    assert ftimes == sorted(ftimes), "frame lines not in time order"
    tl = a.split("\n")
    tdlg = [l for l in tl if DLG_T.match(l)]
    assert len(tdlg) == len(dlg), (len(tdlg), len(dlg))
    for p, t in zip(dlg, tdlg):                          # prompt dialogue == target dialogue, same order
        mp, mt = DLG_P.match(p), DLG_T.match(t)
        assert (sec(mp, 1), sec(mp, 3), mp.group(5)) == (sec(mt, 1), sec(mt, 3), mt.group(5)), (p, t)
    for l in tl:
        assert DLG_T.match(l) or AD_T.match(l), f"unexpected target line: {l[:80]!r}"
    out, meta = [], []
    k = 0
    for li, l in enumerate(tl):
        m = AD_T.match(l)
        if not m:
            continue
        t_k = sec(m)
        nd_before = sum(bool(DLG_T.match(x)) for x in tl[:li])        # dialogue lines preceding AD k
        nxt = nd_before if nd_before < len(dlg) else None              # index of the next dialogue line
        if nxt is not None:
            G = sec(DLG_P.match(dlg[nxt]))
            assert G >= t_k, (ridx, k, t_k, G)
            keep_d = dlg[:nxt + 1]
            v = [x for x in vis if sec(EV.match(x) or FR.match(x)) <= G]
        else:
            G, keep_d, v = None, dlg, vis                              # no later dialogue: whole clip
        # kept lines are a prefix in time only if events are sorted; filter keeps order regardless
        nfr = sum(bool(FR.match(x)) for x in v)
        imgs = rec["images"][:nfr]                                     # frames are time-sorted -> prefix
        assert [sec(FR.match(x)) for x in v if FR.match(x)] == ftimes[:nfr]
        prompt = "\n".join(head + v + sep + ["## Dialogue"] + keep_d)
        assert prompt.count("<image>") == len(imgs)
        target = "\n".join(tl[:li + 1])
        out.append({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}],
                    "images": imgs})
        prev_end = max([sec(DLG_P.match(x), 3) for x in dlg[:nd_before]] or [0.0])
        meta.append({"rec": ridx, "k": k, "t": t_k, "gap_end": G, "gt": l[m.end():],
                     "n_events": sum(bool(EV.match(x)) for x in v), "n_events_full": sum(bool(EV.match(x)) for x in vis),
                     "n_frames": nfr, "n_frames_full": len(ftimes), "n_dlg": len(keep_d), "n_dlg_full": len(dlg),
                     "inside_dialogue": t_k < prev_end, "label": target})
        k += 1
    return out, meta


def main():
    src, od = sys.argv[1], sys.argv[2]
    args = [a for a in sys.argv[3:] if a != "--per-gap"]; per_gap = "--per-gap" in sys.argv
    n = int(args[0]) if args else 5
    d = json.load(open(src))[:n]
    recs, meta = [], []
    for i, r in enumerate(d):
        o, m = build(r, i)
        if per_gap:
            last = {}
            for j, x in enumerate(m):
                last[x["gap_end"]] = j                                   # gap_end identifies the gap within a clip
            for j, x in enumerate(m):
                x["pos"] = x["k"]
                if x["gap_end"] is None:
                    x["row"] = None
                else:
                    x["row"] = len(recs) + sorted(v for g, v in last.items() if g is not None).index(last[x["gap_end"]])
                    x["label"] = m[last[x["gap_end"]]]["label"]
            o = [o[j] for j in sorted(v for g, v in last.items() if g is not None)]
        recs += o; meta += m
    json.dump(recs, open(f"{od}/causal.json", "w"), ensure_ascii=False)
    json.dump(d, open(f"{od}/std.json", "w"), ensure_ascii=False)
    json.dump(meta, open(f"{od}/meta.json", "w"), ensure_ascii=False, indent=1)
    print("records", len(d), "causal", len(recs), "inside_dialogue", sum(x["inside_dialogue"] for x in meta),
          "no_next_dialogue", sum(x["gap_end"] is None for x in meta))


if __name__ == "__main__":
    main()
