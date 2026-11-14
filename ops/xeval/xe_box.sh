#!/bin/bash
# xeval box driver (2026-11-13): cross-dataset / combined-model evals on one GPU box. Sets up /scratch (repo at <commit>, env,
# base model), fetches each needed eval set, then runs the jobs one after another, each sharded over all GPUs.
#   bash xe_box.sh <commit> JOB [JOB ...]      JOB = tag,ckpt_s3_prefix,data,bins|nobins,tf|free:N[,k/m]  (k/m: this box takes part k of m, so m boxes share a job)
#   data: cmd591 (CMD-AD eval591, e1_eval591/eval591.json) | rftest (REFRAMED test 91) | rch (REFRAMED challenge, 620 windows)
#         | s3:<key under adlm/datasets> (any CMD-AD record json, frames from s3 adlm/frames/1fps_448x252, or per cmdad_eval591_9b/frames_src64.json)
WANT=$1; shift; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1 PATH=$HOME/.local/bin:/usr/local/bin:$PATH
exec >> $HOME/xe.log 2>&1; echo "START $(date -u) jobs: $*"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
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
get_cmd591(){ [ -f /scratch/xe/cmd591.ok ] && return; mkdir -p /scratch/xe && cd /scratch/xe
  aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/SHA256SUMS . && aws s3 cp --only-show-errors $B/adlm/datasets/e1_eval591/eval591.json eval591_9b.json
  grep eval591_9b.json SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('eval591_9b.json')) for p in r['images']})))" > vids.txt
  cat vids.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /scratch/frames/1fps_448x252/{}/
  aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/frames_src64.json .
  python3 -c "import json;[print(v, p) for v, p in json.load(open('frames_src64.json')).items() if p != 'adlm/frames/1fps_448x252']" | \
    while read v p; do aws s3 sync --only-show-errors $B/$p/$v/ /scratch/frames/1fps_448x252/$v/; done
  # never touch the box's /data (box C keeps its own data there): point the records at /scratch instead
  sed -i 's#/data/cmd_ad/frames/1fps_448x252/#/scratch/frames/1fps_448x252/#g' eval591_9b.json
  python3 -c "import json,os;d=json.load(open('eval591_9b.json'));m=sum(not os.path.exists(p) for r in d for p in r['images']);print('CMD591',len(d),'MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }
  touch /scratch/xe/cmd591.ok; }
get_rftest(){ [ -f /scratch/rf_datasets/frame_variants_448x252/reframed_bins_f150_999k_test.json ] || \
  aws s3 cp --only-show-errors $B/reframed/train_data/rf_frames_datasets.tar.gz - | pigz -dc | tar -x -C /scratch
  python3 -c "import json,os;d=json.load(open('/scratch/rf_datasets/frame_variants_448x252/reframed_bins_f150_999k_test.json'));m=sum(not os.path.exists(p) for r in d for p in r['images']);print('RFTEST',len(d),'MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }; }
get_rch(){ [ -f /scratch/xe/rch.ok ] && return; mkdir -p /scratch/xe; C=$B/reframed/challenge_eval/r2/ds
  aws s3 cp --only-show-errors $C/ds_json.tgz - | tar xz -C /
  for b in g2 q1 q2; do aws s3 cp --only-show-errors $C/frames_$b.tgz - | pigz -dc | tar x -C / ; done
  python3 -c "import json,os;d=json.load(open('/scratch/rc_ds/reframed_bins_f150_999k_challenge.json'));m=sum(not os.path.exists(p) for r in d for p in r['images']);print('RCH',len(d),'MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }
  touch /scratch/xe/rch.ok; }
get_s3json(){ # $1 = s3 key under adlm/datasets (CMD-AD records with /data/cmd_ad/frames/1fps_448x252 paths) -> /scratch/xe/<name>
  mkdir -p /scratch/xe && cd /scratch/xe; local n=$(basename $1); [ -f $n.ok ] && { JS=/scratch/xe/$n; return; }
  aws s3 cp --only-show-errors $B/adlm/datasets/$1 $n
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('$n')) for p in r['images']})))" > vids_$n.txt
  cat vids_$n.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ /scratch/frames/1fps_448x252/{}/
  # clips whose frames live elsewhere (the 64 extra eval591 clips): same source map as get_cmd591
  aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/frames_src64.json .
  python3 -c "import json;w=set(open('vids_$n.txt').read().split());[print(v, p) for v, p in json.load(open('frames_src64.json')).items() if v in w and p != 'adlm/frames/1fps_448x252']" | \
    while read v p; do aws s3 sync --only-show-errors $B/$p/$v/ /scratch/frames/1fps_448x252/$v/; done
  # DD-variant builds (dd2b/dd4b/dd9b eval591): the 23 special clips point at /data/cmd_ad/frames/eval591_official (s3 adlm/frames/eval591_official)
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('$n')) for p in r['images'] if '/eval591_official/' in p})))" | \
    while read v; do [ -n "$v" ] && aws s3 sync --only-show-errors $B/adlm/frames/eval591_official/$v/ /scratch/frames/eval591_official/$v/; done
  sed -i -e 's#/data/cmd_ad/frames/1fps_448x252/#/scratch/frames/1fps_448x252/#g' -e 's#/data/cmd_ad/frames/eval591_official/#/scratch/frames/eval591_official/#g' $n
  python3 -c "import json,os;d=json.load(open('$n'));m=sum(not os.path.exists(p) for r in d for p in r['images']);print('S3JSON',len(d),'MISSING',m);assert m==0" || { echo FRAMES_MISSING; exit 1; }
  touch $n.ok; JS=/scratch/xe/$n; }
for J in "$@"; do IFS=, read TAG CK DATA TOK MODE PART <<< "$J"; PART=${PART:-0/1}; K=${PART%/*}; M=${PART#*/}
  case $DATA in cmd591) get_cmd591; JS=/scratch/xe/eval591_9b.json;;
    rftest) get_rftest; JS=/scratch/rf_datasets/frame_variants_448x252/reframed_$([ $TOK = bins ] && echo bins || echo nobudget)_f150_999k_test.json;;
    s3:*) get_s3json ${DATA#s3:};;
    rch) get_rch; JS=/scratch/rc_ds/reframed_$([ $TOK = bins ] && echo bins || echo nobudget)_f150_999k_challenge.json;; esac
  . $HOME/.cr.env; CKD=/scratch/ckpt/$(echo $CK | md5sum | cut -c1-8)
  [ -f $CKD/adapter_model.safetensors ] || aws s3 cp --only-show-errors --recursive $CK/ $CKD/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
  [ -f $CKD/adapter_model.safetensors ] || { echo "CKPT_MISSING $CK"; continue; }
  echo "JOB_START $TAG $(date -u +%T) $JS"
  for g in $(seq 0 $((NG - 1))); do bash $R/ops/xeval/xe_infer.sh $WANT $CKD $TAG $JS $g $((K * NG + g)) $((M * NG)) $TOK $MODE > /scratch/xe_${TAG}_g$g.log 2>&1 & done; wait
  echo "JOB_DONE $TAG $(date -u +%T)"; grep -h -E "^END|SYNCED|MISMATCH" /scratch/xe_${TAG}_g*.log
done; echo "ALL_DONE $(date -u)"
