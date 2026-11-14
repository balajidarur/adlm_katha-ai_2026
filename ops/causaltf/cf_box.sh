#!/bin/bash
# Causal free-running per-gap run on one multi-GPU box (2026-11-14), cancel-safe and incremental.
#   bash cf_box.sh <commit> <tag> <ckpt_s3_prefix> <s3 records dir under adlm/datasets>
# The records dir holds shard_<k>.json (one per GPU, whole clips in clip order, built by build_causal_free.py <a>-<b> --shards K)
# and meta.json. GPU k runs shard k with ops/xeval/xe_infer.sh free:2048 (greedy, cutoff 49152, bins); the trainer streams one
# row per record to <out>/stream/generated.rank0.jsonl as it finishes. Every 5 min the stream files and progress.json
# (clips/gaps done) go to s3 adlm/xeval/preds/<tag>/g<k>/ and adlm/xeval/preds/<tag>/progress.json.
# Cancel: touch ~/<tag>.cancel on the box -> GPU jobs are killed, a final sync is made, progress says CANCELLED.
WANT=$1; TAG=$2; CK=$3; RD=$4; B=s3://balu-transfer-933516006154-apne1; P=$B/adlm/xeval/preds/$TAG
export AWS_DEFAULT_REGION=ap-northeast-1 PATH=$HOME/.local/bin:/usr/local/bin:$PATH
exec >> $HOME/$TAG.log 2>&1; echo "START $(date -u) $*"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p /scratch && sudo mount /dev/$DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
cd /scratch; [ "$(git -C adlm_repo rev-parse --short HEAD 2>/dev/null)" = "$WANT" ] || { aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && rm -rf adlm_repo && git clone -q -b qwen38-dense-dataset-training adlm_$WANT.bundle adlm_repo; }
R=/scratch/adlm_repo; HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
MD5=$(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null || [ ! -d $HOME/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B ]; then
  mkdir -p pack; aws s3 cp --only-show-errors --recursive $B/adlm/tf_rerun/ pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc pack/$f.tgz | sudo tar -x -C / ; done; rm -rf pack; sudo chown -R ubuntu:ubuntu $HOME/adlm
fi
NG=$(nvidia-smi -L | wc -l); echo "GPUS $NG"
W=/scratch/cf/$TAG; mkdir -p $W; cd $W
aws s3 cp --only-show-errors --recursive $B/adlm/datasets/$RD/ . || { echo DATA_FAIL; exit 1; }
K=$(python3 -c "import json;print(json.load(open('meta.json'))['shards'])"); [ "$K" = "$NG" ] || { echo "SHARDS $K != GPUS $NG"; exit 1; }
python3 -c "import json;print('\n'.join(sorted({r['video_id'] for r in json.load(open('meta.json'))['rows']})))" > vids.txt
cat vids.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /scratch/frames/1fps_448x252/{}/
aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/frames_src64.json .
python3 -c "import json;w=set(open('vids.txt').read().split());[print(v, p) for v, p in json.load(open('frames_src64.json')).items() if v in w and p != 'adlm/frames/1fps_448x252']" | \
  while read v p; do aws s3 sync --only-show-errors $B/$p/$v/ /scratch/frames/1fps_448x252/$v/; done
sed -i 's#/data/cmd_ad/frames/1fps_448x252/#/scratch/frames/1fps_448x252/#g' shard_*.json
python3 -c "import json,glob,os;m=sum(not os.path.exists(p) for f in glob.glob('shard_*.json') for r in json.load(open(f)) for p in r['images']);print('MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }
CKD=/scratch/ckpt/$(echo $CK | md5sum | cut -c1-8)
[ -f $CKD/adapter_model.safetensors ] || aws s3 cp --only-show-errors --recursive $CK/ $CKD/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
[ -f $CKD/adapter_model.safetensors ] || { echo "CKPT_MISSING $CK"; exit 1; }
aws s3 cp --only-show-errors $W/meta.json $P/meta.json
progress(){ python3 - "$1" <<PY
import json, os, sys, datetime
M = json.load(open("$W/meta.json")); done = {}
for k in range(M["shards"]):
    f = "/scratch/inf/${TAG}_g%d_s0of1/stream/generated.rank0.jsonl" % k
    done[k] = {json.loads(l)["idx"] for l in open(f) if l.strip()} if os.path.exists(f) else set()
rows = [r for r in M["rows"] if r["srow"] is not None]; by = {}
for r in rows: by.setdefault(r["clip"], []).append(r["srow"] in done[r["shard"]])
cd = sorted(c for c, v in by.items() if all(v))
json.dump({"status": sys.argv[1], "utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "clips_total": len(by),
           "clips_done": len(cd), "gaps_total": len(rows), "gaps_done": sum(len(v) for v in done.values()),
           "per_gpu_done": [len(done[k]) for k in range(M["shards"])], "per_gpu_total": M["records_per_shard"], "clips_done_list": cd},
          open("$W/progress.json", "w"))
print("PROGRESS", sys.argv[1], len(cd), "clips", sum(len(v) for v in done.values()), "gaps")
PY
}
dosync(){ . $HOME/.cr.env; progress "$1"
  for k in $(seq 0 $((NG - 1))); do O=/scratch/inf/${TAG}_g${k}_s0of1; [ -d $O ] || continue
    aws s3 sync --only-show-errors $O $P/g$k/ --exclude "*" --include "stream/*" --include "*.jsonl" --include "*.json" --include "*.yaml"
    [ -f $O.log ] && aws s3 cp --only-show-errors $O.log $P/g$k/run.log; done
  aws s3 cp --only-show-errors $W/progress.json $P/progress.json; aws s3 cp --only-show-errors $HOME/$TAG.log $P/box.log; }
echo "JOB_START $TAG $(date -u +%T)"; PIDS=""
for g in $(seq 0 $((NG - 1))); do
  setsid bash $R/ops/xeval/xe_infer.sh $WANT $CKD ${TAG}_g$g $W/shard_$g.json $g 0 1 bins free:2048 > /scratch/${TAG}_g$g.drv.log 2>&1 &
  PIDS="$PIDS $!"; done
echo "PIDS$PIDS"
while :; do
  alive=0; for p in $PIDS; do kill -0 $p 2>/dev/null && alive=1; done
  if [ -f $HOME/$TAG.cancel ]; then for p in $PIDS; do kill -TERM -- -$p 2>/dev/null; done; sleep 20
    for p in $PIDS; do kill -KILL -- -$p 2>/dev/null; done; dosync CANCELLED; echo "CANCELLED $(date -u)"; exit 0; fi
  [ $alive = 0 ] && break
  dosync RUNNING; for i in $(seq 30); do [ -f $HOME/$TAG.cancel ] && break; sleep 10; done
done
grep -h -E "^END|SYNCED|MISMATCH" /scratch/${TAG}_g*.drv.log; dosync DONE; echo "ALL_DONE $(date -u)"
