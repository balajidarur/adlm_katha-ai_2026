"""Table of MTG metrics (official aggregate.py loader + rules: per-video, capped values dropped, mean +- 1.96 SEM).
  python mtg_table.py <split> sys1 sys2 ...     (dirs /storage/users/dbalu/mtg_eval/sys/<sys>_<split>/eval/metrics)"""
import glob, os, sys, types
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
sys.path.insert(0, "/workspace/dbalu/tasks_nov8/T4/mtg-ad")
import matplotlib; matplotlib.use("Agg")
import numpy as np
from analysis import aggregate as A
E = "/storage/users/dbalu/mtg_eval/sys"
split, systems = sys.argv[1], sys.argv[2:]
cols = [k for k in A.METRICS if not k.startswith("video_asr")]
rows = {s: {os.path.basename(p): A.load_video_metrics(p) for p in glob.glob(f"{E}/{s}_{split}/eval/metrics/output_*.json")} for s in systems}
def cell(s, k):
    x = np.array([r[k] for r in rows[s].values() if r and k in r], float)
    return f"{x.mean():.2f} ± {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.2f} (n={len(x)})" if len(x) > 1 else "–"
print("| System | videos with ADs | " + " | ".join(A.METRICS[k][0] for k in cols) + " |")
print("|---|---|" + "---|" * len(cols))
for s in systems:
    n = sum(1 for r in rows[s].values() if r and "total_overlap" in r)
    print(f"| {s} | {n}/{len(rows[s])} | " + " | ".join(cell(s, k) for k in cols) + " |")
