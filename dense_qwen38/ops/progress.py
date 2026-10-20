"""Every 5 min: append a progress snapshot of the full prep run to /prep/progress.jsonl and
rewrite /prep/progress.txt (human-readable). Exits once the run's summary line appears."""
import json, re, shutil, subprocess, time
LOG, TOTAL = "/prep/prep_full.log", 8599
t0, prev = None, None
while True:
    lines = open(LOG).read().splitlines()
    ok = sum(" ok=True " in l for l in lines)
    fail = [l.split()[1] for l in lines if " ok=False " in l]
    frames = sum(int(m.group(1)) for l in lines if (m := re.search(r" ok=True frames=(\d+)", l)))
    now = time.time()
    if t0 is None:
        import os; t0 = os.path.getmtime("/prep/prep_360p/_portraits") if os.path.exists("/prep/prep_360p/_portraits") else now
    rate5 = (frames - prev[1]) / (now - prev[0]) if prev else frames / max(now - t0, 1)
    done = ok + len(fail)
    remaining = TOTAL - done
    eta_min = remaining / ((done - prev[2]) / (now - prev[0])) / 60 if prev and done > prev[2] else None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    du = shutil.disk_usage("/prep")
    snap = {"utc": time.strftime("%H:%M", time.gmtime(now)), "done": done, "ok": ok, "failed": len(fail),
            "remaining": remaining, "pct": round(100 * done / TOTAL, 1), "frames": frames,
            "frames_per_s_last5min": round(rate5), "eta_min": round(eta_min) if eta_min else None,
            "gpu_util_now": gpu, "disk_used_gb": round(du.used / 1e9, 1), "failed_clips": fail}
    open("/prep/progress.jsonl", "a").write(json.dumps(snap) + "\n")
    open("/prep/progress.txt", "w").write(
        f"{snap['utc']} UTC  done {done}/{TOTAL} ({snap['pct']}%)  ok {ok}  failed {len(fail)}  remaining {remaining}\n"
        f"frames {frames:,}  last-5-min {snap['frames_per_s_last5min']} frames/s  ETA {snap['eta_min']} min\n"
        f"GPU util {gpu}  /prep used {snap['disk_used_gb']} GB\nfailed: {' '.join(fail)}\n")
    prev = (now, frames, done)
    if any(l.startswith("summary:") for l in lines):
        break
    time.sleep(300)
