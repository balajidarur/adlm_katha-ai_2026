#!/bin/bash
# causaltf full CMD-AD eval591 run on one 4xL40S box: causal TF (one record per gap, build_causal.py --per-gap) on 4 GPUs,
# then standard TF on all 591 clips on the same 4 GPUs (same code/settings, ct_infer.sh). Shards sync to s3 causaltf/preds.
#   bash run_full.sh <commit> <s3 ckpt prefix> <tag>
WANT=$1; CKS3=$2; TAG=$3; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1 PATH=$HOME/.local/bin:/usr/local/bin:$PATH
echo "START $(date -u)"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable; sudo chown ubuntu:ubuntu /opt/dlami/nvme
cd /scratch; [ "$(git -C adlm_repo rev-parse --short HEAD 2>/dev/null)" = "$WANT" ] || { aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && rm -rf adlm_repo && git clone -q -b qwen38-dense-dataset-training adlm_$WANT.bundle adlm_repo; }
R=/scratch/adlm_repo; HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
MD5=$(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
W=/scratch/causaltf/$TAG; mkdir -p $W && cd $W
aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/SHA256SUMS . && aws s3 cp --only-show-errors $B/adlm/datasets/e1_eval591/eval591.json eval591_9b.json
grep eval591_9b.json SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('eval591_9b.json')) for p in r['images']})))" > vids.txt
cat vids.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /scratch/frames/1fps_448x252/{}/
# 64 of the 591 clips list their frame source in frames_src64.json (some live under cmd_ad_eval591/ instead of adlm/frames)
aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/frames_src64.json .
python3 -c "import json;[print(v, p) for v, p in json.load(open('frames_src64.json')).items() if p != 'adlm/frames/1fps_448x252']" | \
  while read v p; do aws s3 sync --only-show-errors $B/$p/$v/ /scratch/frames/1fps_448x252/$v/; done
sudo mkdir -p /data/cmd_ad/frames && sudo ln -sfn /scratch/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252
python3 -c "import json,os;d=json.load(open('eval591_9b.json'));m=sum(not os.path.exists(p) for r in d for p in r['images']);print('CLIPS',len(d),'MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }
aws s3 cp --only-show-errors --recursive $CKS3/ /scratch/ckpt/$TAG/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
ls /scratch/ckpt/$TAG/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
python3 -I $R/ops/causaltf/build_causal.py $W/eval591_9b.json $W 591 --per-gap || exit 1
aws s3 cp --only-show-errors $W/meta.json $B/causaltf/preds/meta_$TAG.json
echo "CAUSAL_START $(date -u)"
for g in 0 1 2 3; do bash $R/ops/causaltf/ct_infer.sh $WANT /scratch/ckpt/$TAG causal_$TAG $W/causal.json $g $g 4 bins > $W/causal_g$g.log 2>&1 & done; wait
echo "CAUSAL_DONE $(date -u)"; grep -h -E "^END|SYNCED" $W/causal_g*.log
for g in 0 1 2 3; do bash $R/ops/causaltf/ct_infer.sh $WANT /scratch/ckpt/$TAG std_$TAG $W/std.json $g $g 4 bins > $W/std_g$g.log 2>&1 & done; wait
echo "STD_DONE $(date -u)"; grep -h -E "^END|SYNCED" $W/std_g*.log; echo ALL_DONE
