#!/bin/bash
# on a box after FULL_DONE: dedupe per shot, final S3 sync, verify sizes vs S3, stop servers/loops, remove creds,
# re-enable idle shutdown.  Prints VERIFY_OK / VERIFY_FAIL.  usage: finalize_box.sh [VARIANT]
V=${1:-nocast_mtg_b}; OUT=/scratch/mtg/dd_full_$V; S3=s3://balu-transfer-933516006154-apne1/mtg/dd_full_$V/
. /scratch/cbenv/bin/activate
touch /scratch/mtg/sync.stop
python /scratch/code/mtg/dedupe_dd.py --root $OUT --stats $OUT/_dedup_stats_$(hostname).json | tail -20
. $HOME/.cr.env; export AWS_DEFAULT_REGION=ap-northeast-1
mkdir -p $OUT/_runlogs_$(hostname); cp /scratch/mtg/run_full.log $OUT/run_log.jsonl $OUT/_runlogs_$(hostname)/ 2>/dev/null
aws s3 sync --only-show-errors $OUT $S3 --exclude "_logs/*" --exclude "*/_logs/*"
# verify: every local file (except _logs) present in S3 with the same size
python - "$OUT" "$S3" <<'PY'
import os, subprocess, sys
out, s3 = sys.argv[1], sys.argv[2]
loc = {}
for r, _, fs in os.walk(out):
    if "/_logs" in r or r.endswith("_logs"): continue
    for f in fs:
        p = os.path.join(r, f); loc[os.path.relpath(p, out)] = os.path.getsize(p)
lst = subprocess.run(["aws", "s3", "ls", "--recursive", s3], capture_output=True, text=True).stdout.splitlines()
pre = s3.split("/", 3)[3]
rem = {l.split(None, 3)[3][len(pre):]: int(l.split(None, 3)[2]) for l in lst if len(l.split(None, 3)) == 4}
bad = [k for k, v in loc.items() if rem.get(k) != v]
n = sum(1 for k in loc if k.endswith("/adlm_descriptions.json"))
print(f"local files {len(loc)}, videos {n}, mismatched {len(bad)}", bad[:5])
print("VERIFY_OK" if not bad else "VERIFY_FAIL")
PY
pkill -f "vllm serve" ; pkill -f sync_loop.sh
sudo rm -f /etc/gpu-idle-shutdown.disable
rm -f $HOME/.cr.env
echo FINALIZED
