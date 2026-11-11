#!/bin/bash
# gapexp4 box: repo at <commit>, checkpoint, gap-shift records, frames for the 24 clips into /data/cmd_ad/frames (record paths),
# then 4 GPU shards.   bash gx_box.sh <commit> <s3 ckpt prefix> <tag> <json name>
WANT=$1; CKS3=$2; TAG=$3; JS=$4; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
exec > $HOME/gx_$TAG.log 2>&1; echo "START $(date -u)"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dpbno NAME,SIZE,TYPE | awk '$3=="disk"' | sort -k2 -n -r | while read d sz t; do
          [ -z "$(lsblk -no MOUNTPOINT $d | tr -d '[:space:]')" ] && [ $(lsblk -no NAME $d | wc -l) = 1 ] && { echo $d; break; }; done)
  sudo mkfs.ext4 -q -F $DEV && sudo mkdir -p /scratch && sudo mount $DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
cd /scratch; aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && { [ -d adlm_repo/.git ] || git clone -q adlm_$WANT.bundle adlm_repo; } \
  && git -C adlm_repo fetch -q /scratch/adlm_$WANT.bundle qwen38-dense-dataset-training && git -C adlm_repo checkout -q $WANT; echo "COMMIT $(git -C adlm_repo rev-parse --short HEAD)"
mkdir -p gx; aws s3 cp --only-show-errors $B/gapexp4/in/$JS gx/ && aws s3 cp --only-show-errors $B/gapexp4/in/vids.txt gx/
aws s3 cp --only-show-errors --recursive $CKS3/ /scratch/ckpt/$TAG/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt" &
[ -d /data/cmd_ad ] || { sudo mkdir -p /data/cmd_ad && sudo chown ubuntu:ubuntu /data/cmd_ad; }; mkdir -p /data/cmd_ad/frames/1fps_448x252
cat gx/vids.txt | xargs -P 24 -I{} sh -c "[ \$(ls /data/cmd_ad/frames/1fps_448x252/{} 2>/dev/null | wc -l) -gt 10 ] || aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /data/cmd_ad/frames/1fps_448x252/{}/"
wait; python3 -c "import json,os; d=json.load(open('/scratch/gx/$JS')); print('missing frames', sum(not os.path.exists(p) for r in d for p in r['images']))"
for g in 0 1 2 3; do bash /scratch/adlm_repo/ops/gapexp4/gx_infer.sh $WANT /scratch/ckpt/$TAG $TAG /scratch/gx/$JS $g $g 4 > $HOME/gx_${TAG}_g$g.log 2>&1 & done; wait; echo "ALL_DONE $(date -u)"
