# usage: qprog.py OUT_ROOT CLIPS_FILE PREP_ROOT — every 5 min append a snapshot to OUT_ROOT/progress.{jsonl,txt}
import glob, json, os, sys, time
out, clips, prep = sys.argv[1:4]
L = [c.strip() for c in open(clips) if c.strip()]
tot_shots = {c: len(json.load(open(f"{prep}/{c}/meta.json"))["shots"]) for c in L if os.path.exists(f"{prep}/{c}/meta.json")}
T = sum(tot_shots.values()); t0 = time.time(); s0 = None
while True:
    done = [c for c in tot_shots if os.path.exists(f"{out}/{c.split('/')[-1]}/Qwen3.8-27B-FP8/adlm_descriptions.json")]
    shots = sum(tot_shots[c] for c in done)
    for lp in glob.glob(f"{out}/_logs/*.log"):
        c = os.path.basename(lp)[:-4].replace("_", "/", 1)
        if c in tot_shots and c not in done:
            shots += sum(1 for l in open(lp, errors="ignore") if l.startswith("shot "))
    s0 = shots if s0 is None else s0
    el = time.time() - t0
    rate = (shots - s0) / el * 60 if el > 60 else 0
    eta_h = (T - shots) / rate / 60 if rate else None
    snap = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "clips_done": len(done), "clips_total": len(tot_shots),
            "shots_done": shots, "shots_total": T, "shots_per_min": round(rate, 1), "eta_h": round(eta_h, 1) if eta_h else None}
    open(f"{out}/progress.jsonl", "a").write(json.dumps(snap) + "\n")
    open(f"{out}/progress.txt", "w").write(json.dumps(snap, indent=1) + "\n")
    time.sleep(300)
