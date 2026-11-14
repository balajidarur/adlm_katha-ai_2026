#!/usr/bin/env python3
"""Causal (no-lookahead) FREE-RUNNING records: one record per dialogue gap of one clip.

Dialogue lines d_1..d_N come from the prompt's "## Dialogue" section (asserted sorted by start). Gaps:
  gap 0 = [clip start (0.0), d_1^s),  gap j = [d_j^e, d_{j+1}^s),  gap N = [d_N^e, clip end].
A gap is skipped only when d_{j+1} starts before the latest end of d_1..d_j (overlapping lines); touching lines give a
zero-length gap, which is kept (and reported) like any other.
Truncation is that of build_causal.py: for the gap ending at G = d_{j+1}^s the header sections stay unchanged;
"## Visual events" keeps event and [frame] lines with time <= G (images = matching prefix, <image> count asserted);
"## Dialogue" keeps d_1..d_j plus exactly one more, the real next line d_{j+1}. The last gap gets the full prompt.
The assistant message is the clip's unchanged GT label: free-running predict generates from the prompt alone (no
assistant prefix), the label is only carried along.

  python build_causal_free.py <records.json> <record_index> <out_dir> [--video-id ID]
  -> <out_dir>/causal_free.json (records, gap order), gaps.json (one row per record: gap span, length, what was visible)
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_causal import EV, FR, DLG_P, sec, split_prompt  # noqa: E402


def build(rec):
    u = rec["messages"][0]["content"]
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    head, vis, sep, dlg = split_prompt(u)
    assert u.count("<image>") == len(rec["images"])
    vt = [sec(EV.match(x) or FR.match(x)) for x in vis]
    ftimes = [sec(FR.match(x)) for x in vis if FR.match(x)]
    assert ftimes == sorted(ftimes), "frame lines not in time order"
    D = [(sec(DLG_P.match(x), 1), sec(DLG_P.match(x), 3), DLG_P.match(x).group(5)) for x in dlg]
    assert [d[0] for d in D] == sorted(d[0] for d in D), "dialogue not sorted by start"
    clip_end = max(vt + [d[1] for d in D])
    out, meta = [], []
    for g in range(len(D) + 1):
        s = 0.0 if g == 0 else max(d[1] for d in D[:g])                 # latest end of d_1..d_g
        if g < len(D):
            G = D[g][0]
            if G < s:                                                     # overlapping dialogue: no gap
                meta.append({"gap": g, "skipped": "overlap", "start": s, "end": G})
                continue
            keep_d = dlg[:g + 1]
            v = [x for x, t in zip(vis, vt) if t <= G]
        else:
            G, keep_d, v = None, dlg, vis
        nfr = sum(bool(FR.match(x)) for x in v)
        imgs = rec["images"][:nfr]
        assert [sec(FR.match(x)) for x in v if FR.match(x)] == ftimes[:nfr]
        prompt = "\n".join(head + v + sep + ["## Dialogue"] + keep_d)
        assert prompt.count("<image>") == len(imgs)
        if G is None:
            assert prompt == u and imgs == rec["images"], "last gap must be the untouched prompt"
        out.append({"messages": [{"role": "user", "content": prompt}, rec["messages"][1]], "images": imgs})
        ev = [sec(EV.match(x)) for x in v if EV.match(x)]
        meta.append({"gap": g, "row": len(out) - 1, "start": s, "end": G if G is not None else clip_end,
                     "is_last": G is None, "length": round((G if G is not None else clip_end) - s, 1),
                     "n_events": len(ev), "n_events_full": sum(bool(EV.match(x)) for x in vis),
                     "n_frames": nfr, "n_frames_full": len(ftimes), "last_event": ev[-1] if ev else None,
                     "last_frame": ftimes[nfr - 1] if nfr else None, "n_dlg": len(keep_d), "n_dlg_full": len(dlg),
                     "next_dialogue": D[g][2] if G is not None else None})
    return out, meta, clip_end


def main():
    a = sys.argv[1:]
    vid = a[a.index("--video-id") + 1] if "--video-id" in a else None
    src, idx, od = a[0], int(a[1]), a[2]
    rec = json.load(open(src))[idx]
    if vid:
        assert vid in rec["images"][0], (vid, rec["images"][0])
    out, meta, clip_end = build(rec)
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/causal_free.json", "w"), ensure_ascii=False)
    json.dump({"src": src, "index": idx, "video_id": vid, "clip_end": clip_end, "gaps": meta},
              open(f"{od}/gaps.json", "w"), ensure_ascii=False, indent=1)
    run = [m for m in meta if "row" in m]
    print("records", len(out), "gaps", len(run), "skipped_overlap", len(meta) - len(run),
          "zero_length", sum(m["length"] == 0 for m in run), "clip_end", clip_end)


if __name__ == "__main__":
    main()
