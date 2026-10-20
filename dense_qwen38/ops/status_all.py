import glob, json, subprocess, time
M = "A:local C:172.31.33.99".split()
done = len(glob.glob("/prep/qwen_dense_final/*/Qwen3.8-27B-FP8/adlm_descriptions.json"))
print(time.strftime("== %Y-%m-%d %H:%M UTC", time.gmtime()), f"| synced on box A: {done}/8553 clips ({100 * done / 8553:.1f}%)")
tot_rate = 0
for e in M:
    n, ip = e.split(":")
    cmd = ("cat /prep/qwen_dense_final/progress.txt" if ip == "local" else
           f"ssh -o ConnectTimeout=8 -i /home/ubuntu/.ssh/c.pem ubuntu@{ip} cat qwen_dense_final/progress.txt")
    try:
        p = json.loads(subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=25).stdout)
        tot_rate += p["shots_per_min"]
        print(f"{n:4s} {p['clips_done']:>5}/{p['clips_total']:<5} clips  {p['shots_done']:>6}/{p['shots_total']:<6} shots  {p['shots_per_min']:>5} shots/min  eta {p['eta_h']} h")
    except Exception as ex:
        print(f"{n:4s} status unavailable ({type(ex).__name__})")
print(f"total {tot_rate:.0f} shots/min")
