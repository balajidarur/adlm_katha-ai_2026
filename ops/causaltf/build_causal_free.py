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
  python build_causal_free.py <records.json> <first>-<last> <out_dir> --shards K
  -> <out_dir>/shard_<k>.json (whole clips per shard, clip order kept, gaps in order), meta.json (shard, srow -> clip, gap)
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
    src, idx, od = a[0], a[1], a[2]
    os.makedirs(od, exist_ok=True)
    if "-" not in idx:                                                   # one clip -> causal_free.json + gaps.json
        idx = int(idx)
        rec = json.load(open(src))[idx]
        if vid:
            assert vid in rec["images"][0], (vid, rec["images"][0])
        out, meta, clip_end = build(rec)
        json.dump(out, open(f"{od}/causal_free.json", "w"), ensure_ascii=False)
        json.dump({"src": src, "index": idx, "video_id": vid, "clip_end": clip_end, "gaps": meta},
                  open(f"{od}/gaps.json", "w"), ensure_ascii=False, indent=1)
        run = [m for m in meta if "row" in m]
        print("records", len(out), "gaps", len(run), "skipped_overlap", len(meta) - len(run),
              "zero_length", sum(m["length"] == 0 for m in run), "clip_end", clip_end)
        return
    # clip range "i-j" (inclusive) -> shard_<k>.json for --shards K GPUs: whole clips per shard, clips balanced by record
    # count (largest first onto the least-loaded shard), each shard's clips in eval order; meta.json maps (shard, row) -> clip/gap
    lo, hi = map(int, idx.split("-"))
    K = int(a[a.index("--shards") + 1]) if "--shards" in a else 1
    D = json.load(open(src))
    clips = []
    for i in range(lo, hi + 1):
        out, meta, clip_end = build(D[i])
        v = D[i]["images"][0].split("/")[-2]
        clips.append((i, v, out, [dict(m, clip=i, video_id=v, clip_end=clip_end) for m in meta]))
    load, assign = [0] * K, {}
    for i, v, out, meta in sorted(clips, key=lambda c: -len(c[2])):
        k = min(range(K), key=lambda s: (load[s], s)); assign[i] = k; load[k] += len(out)
    M = {"src": src, "range": [lo, hi], "shards": K, "records_per_shard": load, "rows": []}
    for k in range(K):
        recs = []
        for i, v, out, meta in clips:
            if assign[i] != k:
                continue
            for m in meta:
                if "row" in m:
                    M["rows"].append(dict(m, shard=k, srow=len(recs) + m["row"]))
                else:
                    M["rows"].append(dict(m, shard=k, srow=None))
            recs += out
        json.dump(recs, open(f"{od}/shard_{k}.json", "w"), ensure_ascii=False)
    json.dump(M, open(f"{od}/meta.json", "w"), ensure_ascii=False)
    run = [m for m in M["rows"] if m["srow"] is not None]
    print("clips", len(clips), "records", len(run), "nonzero", sum(m["length"] > 0 for m in run),
          "zero_length", sum(m["length"] == 0 for m in run), "skipped_overlap", len(M["rows"]) - len(run),
          "last_gaps", sum(m["is_last"] for m in run), "per_shard", load)


if __name__ == "__main__":
    main()
