# build_gapexp5.py - TEACHER-FORCED dialogue-gap perturbation, budget vs no-budget (CMD-AD eval). Per clip one dialogue gap of
# 3-5 s that holds exactly one GT AD (start >= 0.5 s before the next line). Variants: d = 0, two narrowings and two widenings of
# 0.5-3 s (gap-aware: narrowing leaves >= RMIN = 1.0 s between AD start and next line, widening <= the free room). A shift moves the previous
# line by -d/2 and the next line by +d/2 in BOTH the prompt (dialogue section + gap frames re-derived as in gapexp4) and the
# forced target ([DIALOGUE] lines), and moves the AD start with the previous line (-d/2), so the time from the AD start to the
# next line changes by exactly d. Teacher forcing then forces every dialogue line and the shifted "M:SS.s [AD] " prefix and the
# model writes only the AD text. Same clips / gaps / shifts for both models.
#   python3 build_gapexp5.py OUT_DIR [N_GAPS]   (one gap per clip first, then second gaps of clips to reach N_GAPS)
import json, re, random, sys, os
D = "/storage/datasets/CMD-AD/data/additional_preprocessed_data/gapframe_datasets_qwen38/frame_variants_448x252_full_qwen38/"
SRC = {"budget": D + "gapframes_bins_eval.json", "nobudget": D + "cmdad_dds_gapframes_eval.json"}
OUT = sys.argv[1]; NCLIP = int(sys.argv[2]) if len(sys.argv) > 2 else 105; SEED = 5; RMIN = 1.0; DMIN, DMAX = 0.5, 3.0
TS = r"(\d+):(\d+(?:\.\d+)?)"
def sec(m, s): return int(m) * 60 + float(s)
def fmt(t): return f"{int(t // 60)}:{t % 60:04.1f}"
def parse(u):
    head, rest = u.split("## Visual events", 1); ev, dl = rest.split("## Dialogue", 1)
    dlg = [(sec(*m.groups()[:2]), sec(*m.groups()[2:4]), m.group(5)) for m in re.finditer(rf"^{TS}-{TS}\s+(.*)$", dl, re.M)]
    return head, ev, dlg
def tparse(t):   # target lines -> [(kind, start, end, line)]
    out = []
    for l in t.split("\n"):
        m = re.match(rf"^{TS}-{TS} \[DIALOGUE\]", l)
        if m: out.append(("D", sec(*m.groups()[:2]), sec(*m.groups()[2:4]), l)); continue
        m = re.match(rf"^{TS} \[AD\]", l)
        if m: out.append(("A", sec(*m.groups()), None, l)); continue
        out.append(("X", None, None, l))
    return out
recs = {k: {r["images"][0].split("/")[-2]: r for r in json.load(open(p)) if r["images"]} for k, p in SRC.items()}
rng = random.Random(SEED); cands = []
for v, r in recs["budget"].items():
    if v not in recs["nobudget"]: continue
    head, ev, dlg = parse(r["messages"][0]["content"])
    evt = [sec(*m.groups()) for m in re.finditer(rf"^{TS}\s", ev, re.M)]
    T = tparse(r["messages"][1]["content"]); ads = [x[1] for x in T if x[0] == "A"]
    tdl = [(round(x[1], 1), round(x[2], 1)) for x in T if x[0] == "D"]
    if tdl != [(round(a, 1), round(b, 1)) for a, b, _ in dlg]: continue          # prompt and target dialogue must agree
    ok = []
    for i in range(len(dlg) - 1):
        a, b = dlg[i][1], dlg[i + 1][0]
        if not (3.0 <= b - a <= 5.0): continue
        inn = [s for s in ads if a <= s <= b - 0.5]
        if len(inn) != 1 or any(b - 0.5 < s < b for s in ads): continue
        if any(dlg[i][0] <= s < a or dlg[i + 1][0] <= s < dlg[i + 1][1] for s in ads): continue   # no AD inside the moving lines
        if not any(a <= t <= b for t in evt): continue
        # widening (d > 0) moves line i earlier and line i+1 later by d/2: cap d by the free room (dialogue or AD) on each side
        before = max([dlg[i - 1][1] if i else 0.0] + [s for s in ads if s < dlg[i][0]])
        after = min([dlg[i + 2][0] if i + 2 < len(dlg) else 1e9] + [s for s in ads if s >= dlg[i + 1][1]])
        dmax = min(DMAX, 2 * (dlg[i][0] - before - 0.1), 2 * (after - dlg[i + 1][1] - 0.1))
        nmax = min(DMAX, (b - inn[0]) - RMIN)                    # largest narrowing that leaves RMIN s for the AD
        if dmax < DMIN + 0.2 or nmax < DMIN + 0.2: continue
        ok.append((i, inn[0], round(dmax, 2), round(nmax, 2)))
    if ok: rng.shuffle(ok); cands.append((v, ok))
rng.shuffle(cands)
pick = [(v, ok[0]) for v, ok in cands]                                 # one gap per clip first,
pick += [(v, g) for v, ok in cands for g in ok[1:]][:max(0, NCLIP - len(pick))]   # then second gaps of clips to reach N
pick = sorted(pick[:NCLIP])
def variant(r, i, d, s_ad):
    head, ev, dlg = parse(r["messages"][0]["content"]); nd = list(dlg); regions = []
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
    # forced target: shift dialogue lines i, i+1 and the AD in the gap
    lines, di, new_ad = [], -1, round(s_ad - d / 2, 1)
    for kind, s, e, l in tparse(r["messages"][1]["content"]):
        if kind == "D":
            di += 1
            if di in (i, i + 1):
                a, b, _ = nd[di]; l = re.sub(rf"^{TS}-{TS}", f"{fmt(a)}-{fmt(b)}", l)
        key0 = s
        if kind == "A" and abs(s - s_ad) < 0.05:
            l = re.sub(rf"^{TS}", fmt(new_ad), l); s = new_ad
        elif kind == "D" and di in (i, i + 1):
            s = nd[di][0]
        lines.append((s, l, key0))
    tt = [x for x in lines if x[0] is not None]   # shifting must not reorder any target line relative to the original order
    assert sorted(range(len(tt)), key=lambda j: (tt[j][2], j)) == sorted(range(len(tt)), key=lambda j: (tt[j][0], tt[j][2], j)), "target order broken"
    return {"messages": [{"role": "user", "content": nu}, {"role": "assistant", "content": "\n".join(x[1] for x in lines)}], "images": imgs}, nd, new_ad
out = {k: [] for k in SRC}; man = []
for v, (i, s_ad, dmax, nmax) in pick:
    _, _, dlg = parse(recs["budget"][v]["messages"][0]["content"]); a, b = dlg[i][1], dlg[i + 1][0]
    # gap-aware shifts of 0.5-3 s: two narrowings drawn from the lower / upper half of [DMIN, nmax] (nmax keeps >= RMIN s between
    # the AD start and the next line) and two widenings from the halves of [DMIN, dmax] (dmax = free room, <= 3 s), plus d = 0
    hn, hw = (nmax - DMIN) / 2, (dmax - DMIN) / 2
    ds = [0.0] + [round(-rng.uniform(DMIN + k * hn, DMIN + (k + 1) * hn), 2) for k in (1, 0)] + [round(rng.uniform(DMIN + k * hw, DMIN + (k + 1) * hw), 2) for k in (0, 1)]
    for n, d in enumerate(ds):
        for k in SRC:
            rec, nd, new_ad = variant(recs[k][v], i, d, s_ad); out[k].append(rec)
        man.append({"row": len(man), "vid": v, "variant": n, "gap_i": i, "shift": d, "orig_gap": [a, b], "new_gap": [nd[i][1], nd[i + 1][0]],
                    "ad_start": s_ad, "dmax": dmax, "nmax": nmax, "new_ad_start": new_ad, "orig_remaining": round(b - s_ad, 2), "new_remaining": round(nd[i + 1][0] - new_ad, 2)})
os.makedirs(OUT, exist_ok=True)
for k in SRC: json.dump(out[k], open(f"{OUT}/gapexp5_{k}.json", "w"), ensure_ascii=False)
json.dump(man, open(f"{OUT}/manifest5.json", "w"), indent=1)
open(f"{OUT}/vids.txt", "w").write("\n".join(sorted({m["vid"] for m in man})))
print(len(cands), "candidate clips;", sum(len(o) for _, o in cands), "candidate gaps;", len(pick), "gaps picked from", len({v for v, _ in pick}), "clips;", len(man), "records/model")
