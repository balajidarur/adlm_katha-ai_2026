"""Cap frames per clip at MAX_F, thinning each dialogue gap evenly.

Frames arrive as 1-FPS runs inside speech-free gaps. Thinning globally would let
one long gap eat the budget, so the budget is split across runs in proportion to
their length (largest-remainder, floor of 1), and each run is then sampled with
linspace so its first and last frame survive.
"""
import json, re, sys
import numpy as np

SRC = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train.json"
OUT = "/data/cmd_ad/datasets/bins_frames/gapframes_bins_train_cap160.json"
MAX_F = 160
FRAME_LINE = re.compile(r"^\[frame (\d+):(\d\d\.\d)\] <image>$")


def frame_times(user_msg):
    out = []
    for i, l in enumerate(user_msg.split("\n")):
        m = FRAME_LINE.match(l.strip())
        if m:
            out.append((i, int(m.group(1)) * 60 + float(m.group(2))))
    return out


def runs_of(times, max_gap=1.5):
    """Split the frame sequence wherever the time jump implies intervening speech."""
    runs, cur = [], [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > max_gap:
            runs.append(cur); cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    return runs


def allocate(sizes, budget):
    """Largest-remainder split of budget across runs, at least 1 each."""
    n = len(sizes)
    if budget <= n:
        return [1] * n
    total = sum(sizes)
    raw = [max(1.0, s * budget / total) for s in sizes]
    base = [int(x) for x in raw]
    rem = budget - sum(base)
    order = sorted(range(n), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(rem):
        base[order[i % n]] += 1
    return [min(b, s) for b, s in zip(base, sizes)]


def cap(rec):
    u = rec["messages"][0]["content"]
    fl = frame_times(u)
    if len(fl) <= MAX_F:
        return rec, 0
    times = [t for _, t in fl]
    rs = runs_of(times)
    quota = allocate([len(r) for r in rs], MAX_F)
    keep = set()
    for run, k in zip(rs, quota):
        k = max(1, min(k, len(run)))
        idx = np.unique(np.linspace(0, len(run) - 1, k).round().astype(int))
        keep.update(run[j] for j in idx)
    keep = sorted(keep)

    lines = u.split("\n")
    drop_lines = {fl[i][0] for i in range(len(fl)) if i not in set(keep)}
    new_u = "\n".join(l for i, l in enumerate(lines) if i not in drop_lines)
    rec["messages"][0]["content"] = new_u
    rec["images"] = [rec["images"][i] for i in keep]
    assert new_u.count("<image>") == len(rec["images"]), "marker/image mismatch"
    return rec, len(fl) - len(keep)


if __name__ == "__main__":
    d = json.load(open(SRC))
    touched = removed = 0
    for r in d:
        _, n = cap(r)
        if n:
            touched += 1; removed += n
    json.dump(d, open(OUT, "w"), ensure_ascii=False)
    f = np.array([len(r["images"]) for r in d])
    print(f"capped {touched:,} clips, removed {removed:,} frames")
    print(f"frames now: p50 {np.percentile(f,50):.0f} p95 {np.percentile(f,95):.0f} max {f.max()}")
    est = np.array([112 * len(r["images"]) +
                    (len(r["messages"][0]["content"]) + len(r["messages"][1]["content"])) / 3.6
                    for r in d])
    for lim in (40000, 42000, 44000):
        print(f"  still over {lim:,}: {int((est>lim).sum()):,} clips ({(est>lim).mean():.1%})")
