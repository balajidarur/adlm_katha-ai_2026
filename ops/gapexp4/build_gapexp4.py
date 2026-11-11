# build_gapexp4.py - continuous dialogue-gap perturbation for budget vs no-budget (free-running). Per clip pick up to 4
# non-adjacent dialogue gaps of 3-5 s with visual scope; per variant every picked gap gets an independent shift
# d ~ U(-2, +2) s: previous line moves by -d/2, next line by +d/2 (new gap 1-7 s). Same clips/shifts for both datasets.
# Gap frames re-derived in the changed region (1 fps pool f_*.jpg) as in build_gapexp3.py.
import json, re, random
D = "/storage/datasets/CMD-AD/data/additional_preprocessed_data/gapframe_datasets_qwen38/frame_variants_448x252_full_qwen38/"
SRC = {"budget": D + "gapframes_bins_eval.json", "nobudget": D + "cmdad_dds_gapframes_eval.json"}
NCLIP, NVAR, SEED = 24, 2, 4
TS = r"(\d+):(\d+(?:\.\d+)?)"
def sec(m, s): return int(m) * 60 + float(s)
def fmt(t): return f"{int(t // 60)}:{t % 60:04.1f}"
def parse(u):
    head, rest = u.split("## Visual events", 1); ev, dl = rest.split("## Dialogue", 1)
    dlg = [(sec(*m.groups()[:2]), sec(*m.groups()[2:4]), m.group(5)) for m in re.finditer(rf"^{TS}-{TS}\s+(.*)$", dl, re.M)]
    return head, ev, dlg
recs = {k: {r["images"][0].split("/")[-2]: r for r in json.load(open(p)) if r["images"]} for k, p in SRC.items()}
rng = random.Random(SEED); cands = []
for v, r in recs["budget"].items():
    if v not in recs["nobudget"]: continue
    head, ev, dlg = parse(r["messages"][0]["content"])
    evt = [sec(*m.groups()) for m in re.finditer(rf"^{TS}\s", ev, re.M)]
    used, gaps = set(), []
    for i in range(1, len(dlg) - 2):          # gap between line i and i+1; need room around for widening by up to 1 s each side
        g = dlg[i + 1][0] - dlg[i][1]
        if not (3.0 <= g <= 5.0): continue
        if dlg[i][0] - dlg[i - 1][1] < 1.3 or dlg[i + 2][0] - dlg[i + 1][1] < 1.3: continue
        if {i, i + 1} & used: continue
        if sum(1 for t in evt if dlg[i][1] <= t <= dlg[i + 1][0]) < 1: continue
        gaps.append(i); used |= {i - 1, i, i + 1, i + 2}
    if gaps: cands.append((len(gaps), v, gaps[:4]))
rng.shuffle(cands); cands.sort(key=lambda x: -x[0]); pick = cands[:NCLIP]
def variant(r, i_list, shifts):
    head, ev, dlg = parse(r["messages"][0]["content"]); nd = list(dlg); regions = []
    for i, d in zip(i_list, shifts):
        for j, dd in ((i, -d / 2), (i + 1, d / 2)):
            s0, e0, t = dlg[j]; nd[j] = (s0 + dd, e0 + dd, t); regions.append((min(s0, s0 + dd) - 0.6, max(e0, e0 + dd) + 0.6))
    frames, evl, it = {}, [], iter(r["images"])
    for l in ev.split("\n"):
        m = re.match(rf"^\[frame {TS}\] <image>$", l.strip())
        if m: frames[round(sec(*m.groups()), 1)] = next(it); continue
        m = re.match(rf"^{TS}\s", l)
        if m: evl.append((sec(*m.groups()), l))
    inside = lambda t: any(a < t < b for a, b, _ in nd)
    base = r["images"][0].rsplit("/", 1)[0]
    for lo, hi in regions:
        for k in list(frames):
            if lo <= k <= hi and inside(k): frames.pop(k)
        s = max(int(lo), 0)
        while s + 0.5 <= hi:
            t = s + 0.5
            if not inside(t) and t not in frames: frames[t] = f"{base}/f_{s:05d}.jpg"
            s += 1
    items = sorted([(t, 0, f"[frame {fmt(t)}] <image>", p) for t, p in frames.items()] + [(t, 1, l, None) for t, l in evl], key=lambda x: (x[0], x[1]))
    nu = head + "## Visual events\n" + "\n".join(x[2] for x in items) + "\n\n## Dialogue\n" + "\n".join(f"{fmt(a)}-{fmt(b)}  {t}" for a, b, t in nd) + "\n"
    imgs = [x[3] for x in items if x[1] == 0]; assert nu.count("<image>") == len(imgs)
    return {"messages": [{"role": "user", "content": nu}, r["messages"][1]], "images": imgs}, nd
out = {k: [] for k in SRC}; man = []
for _, v, gl in pick:
    for k in SRC: out[k].append(recs[k][v])
    _, _, dlg = parse(recs["budget"][v]["messages"][0]["content"])
    man.append({"row": len(man), "vid": v, "variant": 0, "gaps": [{"i": i, "orig": [dlg[i][1], dlg[i + 1][0]], "shift": 0.0} for i in gl]})
    for n in range(1, NVAR + 1):
        sh = [round(rng.uniform(-2, 2), 2) for _ in gl]
        for k in SRC:
            rec, nd = variant(recs[k][v], gl, sh); out[k].append(rec)
        man.append({"row": len(man), "vid": v, "variant": n, "gaps": [{"i": i, "orig": [dlg[i][1], dlg[i + 1][0]], "new": [nd[i][1], nd[i + 1][0]], "shift": s} for i, s in zip(gl, sh)]})
for k in SRC: json.dump(out[k], open(f"gapexp4_{k}.json", "w"), ensure_ascii=False)
json.dump(man, open("manifest4.json", "w"), indent=1)
vids = sorted({m["vid"] for m in man}); open("vids.txt", "w").write("\n".join(vids))
print(len(cands), "cand clips;", len(pick), "picked;", len(man), "records/model;", sum(len(m["gaps"]) for m in man if m["variant"]), "shifted-gap points/model")
