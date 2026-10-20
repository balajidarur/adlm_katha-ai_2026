"""Cap frames at 150 (JSON only) then drop clips still over the 44k budget.

Nothing on disk is touched - only the `images` list and the matching
`[frame M:SS.s] <image>` lines in the prompt. Frames are thinned per dialogue
gap (proportional budget, linspace within each run) so coverage stays uniform.
"""
import json, re
import numpy as np

SRC = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train.json"
OUT = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train_f150_44k.json"
MAX_F, BUDGET = 150, 44000
FRAME_LINE = re.compile(r"^\[frame (\d+):(\d\d\.\d)\] <image>$")


def frame_lines(u):
    return [(i, int(m.group(1)) * 60 + float(m.group(2)))
            for i, l in enumerate(u.split("\n"))
            for m in [FRAME_LINE.match(l.strip())] if m]


def runs_of(times, max_gap=1.5):
    runs, cur = [], [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap:
            runs.append(cur); cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def allocate(sizes, budget):
    n = len(sizes)
    if budget <= n:
        return [1] * n
    tot = sum(sizes)
    raw = [max(1.0, s * budget / tot) for s in sizes]
    base = [int(x) for x in raw]
    order = sorted(range(n), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(budget - sum(base)):
        base[order[i % n]] += 1
    return [min(b, s) for b, s in zip(base, sizes)]


def cap(rec):
    u = rec["messages"][0]["content"]
    fl = frame_lines(u)
    if len(fl) <= MAX_F:
        return 0
    times = [t for _, t in fl]
    rs = runs_of(times)
    quota = allocate([len(r) for r in rs], MAX_F)
    keep = set()
    for run, k in zip(rs, quota):
        k = max(1, min(k, len(run)))
        for j in np.unique(np.linspace(0, len(run) - 1, k).round().astype(int)):
            keep.add(run[j])
    keep = sorted(keep)
    drop = {fl[i][0] for i in range(len(fl)) if i not in set(keep)}
    lines = u.split("\n")
    rec["messages"][0]["content"] = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    rec["images"] = [rec["images"][i] for i in keep]
    assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])
    return len(fl) - len(keep)


def est(r):
    return 112 * len(r["images"]) + (len(r["messages"][0]["content"]) +
                                     len(r["messages"][1]["content"])) / 3.6


d = json.load(open(SRC))
before = sum(len(r["images"]) for r in d)
capped = cut = 0
for r in d:
    n = cap(r)
    if n:
        capped += 1; cut += n

e = np.array([est(r) for r in d])
keep = [r for r, x in zip(d, e) if x <= BUDGET]
json.dump(keep, open(OUT, "w"), ensure_ascii=False)

f = np.array([len(r["images"]) for r in keep])
ke = np.array([est(r) for r in keep])
print(f"frame cap {MAX_F}: thinned {capped:,} clips, removed {cut:,} frames "
      f"({cut/before:.1%} of {before:,})")
print(f"budget {BUDGET:,}: dropped {len(d)-len(keep):,} clips")
print(f"-> {len(keep):,} clips | frames p50 {np.percentile(f,50):.0f} max {f.max()} "
      f"| tokens p50 {np.percentile(ke,50):,.0f} max {ke.max():,.0f}")

# top-20 heaviest, for the memory probe
top = [keep[i] for i in np.argsort(-ke)[:20]]
json.dump(top, open("/data/cmd_ad/datasets/bins_frames/top20_f150_44k.json", "w"),
          ensure_ascii=False)
te = np.array([est(r) for r in top])
print(f"top20 probe: tokens {te.min():,.0f}-{te.max():,.0f} | frames "
      f"{min(len(r['images']) for r in top)}-{max(len(r['images']) for r in top)}")
