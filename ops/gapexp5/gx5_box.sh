#!/bin/bash
# gapexp5 box (TEACHER-FORCED gap shift): repo at <commit>, env/model if missing, checkpoint, records, frames, then 4 GPU shards.
# Frames: used in place if every record path exists (never writes to an existing /data); otherwise the clips are synced from
# s3 adlm/frames into /scratch/frames and the record paths are rewritten there.
#   bash gx5_box.sh <commit> <s3 ckpt prefix> <tag> <json name> <bins|nobins> ["<gpu ids>", default "0 1 2 3"]
WANT=$1; CKS3=$2; TAG=$3; JS=$4; TOK=$5; GPUS=${6:-0 1 2 3}; NG=$(echo $GPUS | wc -w); B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
exec > $HOME/gx5_$TAG.log 2>&1; echo "START $(date -u)"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p /scratch && sudo mount /dev/$DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
until mkdir /scratch/.setup_lock 2>/dev/null; do sleep 5; done   # one setup at a time when two runs share a box
cd /scratch; [ "$(git -C adlm_repo rev-parse --short HEAD 2>/dev/null)" = "$WANT" ] || { aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && rm -rf adlm_repo && git clone -q -b qwen38-dense-dataset-training adlm_$WANT.bundle adlm_repo; }
HAVE=$(git -C adlm_repo rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; rmdir /scratch/.setup_lock; exit 1; }
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null || [ ! -d $HOME/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B ]; then
  mkdir -p pack; aws s3 cp --only-show-errors --recursive $B/adlm/tf_rerun/ pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc pack/$f.tgz | sudo tar -x -C / ; done; rm -rf pack; sudo chown -R ubuntu:ubuntu $HOME/adlm
fi
mkdir -p gx5
aws s3 cp --only-show-errors $B/gapexp5/in/$JS gx5/ && aws s3 cp --only-show-errors $B/gapexp5/in/vids.txt gx5/
aws s3 cp --only-show-errors --recursive $CKS3/ /scratch/ckpt/$TAG/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt" &
python3 - <<PY > gx5/frames_ok.txt
import json, os; d = json.load(open("/scratch/gx5/$JS")); print(sum(not os.path.exists(p) for r in d for p in r["images"]))
PY
if [ "$(cat gx5/frames_ok.txt)" != 0 ]; then
  mkdir -p /scratch/frames/1fps_448x252
  cat gx5/vids.txt | xargs -P 24 -I{} sh -c "[ \$(ls /scratch/frames/1fps_448x252/{} 2>/dev/null | wc -l) -gt 10 ] || aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /scratch/frames/1fps_448x252/{}/"
  sed -i 's#/data/cmd_ad/frames/1fps_448x252/#/scratch/frames/1fps_448x252/#g' gx5/$JS
fi
wait
python3 - <<PY
import json, os
d = json.load(open("/scratch/gx5/$JS")); keep = [i for i, r in enumerate(d) if all(os.path.exists(p) for p in r["images"])]
json.dump([d[i] for i in keep], open("/scratch/gx5/f_$JS", "w"), ensure_ascii=False); json.dump(keep, open("/scratch/gx5/kept_rows_$TAG.json", "w"))
print("FRAMES kept", len(keep), "of", len(d))
PY
rmdir /scratch/.setup_lock; aws s3 cp --only-show-errors /scratch/gx5/kept_rows_$TAG.json $B/gapexp5/preds/kept_rows_$TAG.json
ls /scratch/ckpt/$TAG/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
k=0; for g in $GPUS; do bash /scratch/adlm_repo/ops/gapexp5/gx5_infer.sh $WANT /scratch/ckpt/$TAG $TAG /scratch/gx5/f_$JS $g $k $NG $TOK > $HOME/gx5_${TAG}_g$g.log 2>&1 & k=$((k + 1)); done
wait; echo "ALL_DONE $(date -u)"
