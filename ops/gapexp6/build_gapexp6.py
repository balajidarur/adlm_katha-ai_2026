# build_gapexp6.py - REFRAMED test, global dialogue-gap scaling for teacher-forced inference.
# For every clip, three versions: original (s = 1.0), all gaps x0.75 (dialogue lines extended into the gaps) and all gaps x1.25
# (dialogue lines compressed). Each gap between consecutive dialogue lines (gap > 0) changes by d = (s - 1) * gap, split half and
# half: the previous line's end moves by -d/2 and the next line's start by +d/2. Applied identically to the prompt's "## Dialogue"
# section and the forced target's [DIALOGUE] lines. A compressed line keeps at least MIN_LINE s (the gap change is clipped there).
# AD start times and frames are unchanged, so only the dialogue timing the model reads differs.
#   python3 build_gapexp6.py SRC_JSON OUT_JSON MANIFEST_JSON N_CLIPS
import json, re, sys
SRC, OUT, MAN, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SCALES = (1.0, 0.75, 1.25); MIN_LINE = 0.2
TS = r"(\d+):(\d+(?:\.\d+)?)"
def sec(m, s): return int(m) * 60 + float(s)
def fmt(t): return f"{int(t // 60)}:{t % 60:04.1f}"
def scale_lines(lines, s):
    """lines: [(start, end)] in order -> new [(start, end)], achieved gap ratio list."""
    new = [list(x) for x in lines]
    for i in range(len(lines) - 1):
        a, b = lines[i][1], lines[i + 1][0]; g = b - a
        if g <= 0: continue
        d = (s - 1) * g
        if d < 0:     # shrink: extend both lines into the gap
            new[i][1] = a - d / 2; new[i + 1][0] = b + d / 2
        else:         # stretch: compress both lines, never below MIN_LINE
            new[i][1] = max(new[i][0] + MIN_LINE, a - d / 2)
            new[i + 1][0] = min(lines[i + 1][1] - MIN_LINE, b + d / 2)
    for x in new:   # a line compressed from both sides
        if x[1] - x[0] < MIN_LINE: m = (x[0] + x[1]) / 2; x[0], x[1] = m - MIN_LINE / 2, m + MIN_LINE / 2
    ratios = [(new[i + 1][0] - new[i][1]) / (lines[i + 1][0] - lines[i][1]) for i in range(len(lines) - 1) if lines[i + 1][0] - lines[i][1] > 0.3]
    return new, ratios
def variant(rec, s):
    u, t = rec["messages"][0]["content"], rec["messages"][1]["content"]
    head, dl = u.split("## Dialogue", 1)
    plines = [(m.start(), m.end(), sec(*m.groups()[:2]), sec(*m.groups()[2:4])) for m in re.finditer(rf"^{TS}-{TS}", dl, re.M)]
    old = [(a, b) for _, _, a, b in plines]
    new, ratios = scale_lines(old, s)
    nd, last = [], 0
    for (p0, p1, _, _), (a, b) in zip(plines, new):
        nd.append(dl[last:p0]); nd.append(f"{fmt(a)}-{fmt(b)}"); last = p1
    nd.append(dl[last:])
    # target [DIALOGUE] lines are the same lines in the same order
    k = iter(new); tl = []
    for l in t.split("\n"):
        m = re.match(rf"^{TS}-{TS}(?= \[DIALOGUE\])", l)
        if m:
            a, b = next(k); l = f"{fmt(a)}-{fmt(b)}" + l[m.end():]
        tl.append(l)
    assert next(k, None) is None, "prompt / target dialogue mismatch"
    return {"messages": [{"role": "user", "content": head + "## Dialogue" + "".join(nd)}, {"role": "assistant", "content": "\n".join(tl)}],
            "images": rec["images"]}, ratios, len(old)
src = json.load(open(SRC))[:N]; out, man = [], []
for ci, rec in enumerate(src):
    for s in SCALES:
        r, ratios, nl = variant(rec, s)
        man.append({"row": len(out), "clip": ci, "clip_key": rec["images"][0].split("/")[-2] if rec["images"] else ci, "scale": s,
                    "n_lines": nl, "n_gaps": len(ratios), "mean_gap_ratio": round(sum(ratios) / max(len(ratios), 1), 4)})
        out.append(r)
json.dump(out, open(OUT, "w"), ensure_ascii=False); json.dump(man, open(MAN, "w"), indent=1)
print(len(src), "clips x", len(SCALES), "=", len(out), "rows; mean achieved gap ratio by scale:",
      {s: round(sum(m["mean_gap_ratio"] for m in man if m["scale"] == s) / len(src), 3) for s in SCALES})
