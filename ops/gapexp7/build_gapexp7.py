# build_gapexp7.py - REFRAMED test, AD-length-dependent budget change for teacher-forced inference.
# Threshold = the model's own median predicted AD TTS in the normal (x1.0) teacher-forced run. For every dialogue gap that holds
# GT ADs (AD start between the previous line's end and the next line's start; a gap with no next line is left alone): if most of
# its ADs were predicted longer than the median, the time from the gap's LAST AD start to the next line is halved; otherwise it is
# doubled. Done by moving only the next line's start (the line keeps >= MIN_LINE s, so some doublings are capped), identically in
# the prompt's "## Dialogue" section and the forced [DIALOGUE] lines. AD starts and frames unchanged.
#   python3 build_gapexp7.py SRC_JSON PRED_TTS_JSON MODEL_KEY OUT_JSON MANIFEST_JSON
import json, re, sys
SRC, PRED, KEY, OUT, MAN = sys.argv[1:6]
MIN_LINE = 0.2
TS = r"(\d+):(\d+(?:\.\d+)?)"
def sec(m, s): return int(m) * 60 + float(s)
def fmt(t): return f"{int(t // 60)}:{t % 60:04.1f}"
P = json.load(open(PRED))[KEY]; MED = P["median"]
src = json.load(open(SRC)); out, man = [], []
for ci, rec in enumerate(src):
    pt = P["per_clip"][str(ci)]
    u, t = rec["messages"][0]["content"], rec["messages"][1]["content"]
    head, dl = u.split("## Dialogue", 1)
    plines = [(m.start(), m.end(), sec(*m.groups()[:2]), sec(*m.groups()[2:4])) for m in re.finditer(rf"^{TS}-{TS}", dl, re.M)]
    lines = [[a, b] for _, _, a, b in plines]
    ads = [sec(*m.groups()) for m in (re.match(rf"^{TS} \[AD\]", l) for l in t.split("\n")) if m]
    assert len(ads) == len(pt), (ci, len(ads), len(pt))
    gaps = []
    for j in range(len(lines)):          # gap before line j: (end of line j-1 or 0, start of line j)
        lo = lines[j - 1][1] if j else 0.0; hi = lines[j][0]
        idx = [k for k, s in enumerate(ads) if lo <= s < hi]
        if not idx: continue
        longer = sum(pt[k] > MED for k in idx); f = 0.5 if longer > len(idx) / 2 else 2.0
        s_last = ads[idx[-1]]; R0 = hi - s_last
        new_hi = s_last + f * R0
        new_hi = min(new_hi, lines[j][1] - MIN_LINE)          # doubling capped by the line's own length
        lines[j][0] = new_hi
        gaps.append(dict(line=j, n_ads=len(idx), n_longer=longer, action="halve" if f == 0.5 else "double", ad_idx=idx,
                         remaining_before=round(R0, 3), remaining_after=round(new_hi - s_last, 3),
                         achieved_factor=round((new_hi - s_last) / R0, 3) if R0 > 0 else None))
    nd, last = [], 0
    for (p0, p1, _, _), (a, b) in zip(plines, lines):
        nd.append(dl[last:p0]); nd.append(f"{fmt(a)}-{fmt(b)}"); last = p1
    nd.append(dl[last:])
    k = iter(lines); tl = []
    for l in t.split("\n"):
        m = re.match(rf"^{TS}-{TS}(?= \[DIALOGUE\])", l)
        if m:
            a, b = next(k); l = f"{fmt(a)}-{fmt(b)}" + l[m.end():]
        tl.append(l)
    assert next(k, None) is None, "prompt / target dialogue mismatch"
    out.append({"messages": [{"role": "user", "content": head + "## Dialogue" + "".join(nd)}, {"role": "assistant", "content": "\n".join(tl)}],
                "images": rec["images"]})
    man.append({"clip": ci, "median": MED, "gaps": gaps})
json.dump(out, open(OUT, "w"), ensure_ascii=False); json.dump(man, open(MAN, "w"), indent=1)
G = [g for m in man for g in m["gaps"]]
for act in ("halve", "double"):
    x = [g for g in G if g["action"] == act]
    fa = sorted(g["achieved_factor"] for g in x if g["achieved_factor"] is not None)
    print(f"{KEY}: {act} {len(x)} gaps / {sum(g['n_ads'] for g in x)} ADs; achieved factor median {fa[len(fa) // 2] if fa else None}")
