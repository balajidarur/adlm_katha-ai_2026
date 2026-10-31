# Input ablations (A2 / A3 / A4) on top of M3: same config as qwen38_bins_frames_full.yaml, only the dataset differs.
#   bash abl_train.sh <name: a2_noframes|a3_nodds|a4_dlgph> <commit> <n_gpus> <grad_accum>
# Effective batch is kept at 32 = 1 x grad_accum x n_gpus (4 GPUs -> 8, 1 GPU -> 32). 3-epoch schedule, stopped once the
# epoch-2 checkpoints (480 = 1.98 ep, 500 = 2.06 ep) are in S3. Code = commit-checked clone (git bundle from S3);
# dataset = the locked JSONs in s3 adlm/datasets/cmdad_qwen38_ablations (SHA256SUMS verified). Never writes to /data.
NAME=$1; WANT=$2; NG=$3; GA=$4
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
if [ -d /opt/dlami/nvme ]; then W=/opt/dlami/nvme; sudo chown ubuntu:ubuntu $W
else W=/scratch
  if ! mountpoint -q $W; then
    DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
    sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p $W && sudo mount /dev/$DEV $W && sudo chown ubuntu:ubuntu $W
  fi
fi
echo "WORKDIR $W"
mv /tmp/cr.env $W/creds.env 2>/dev/null; chmod 600 $W/creds.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1/adlm
R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; D=$W/abl; OUT=$W/runs/qwen38_$NAME; M=$B/models/cmdad/qwen38-dd_$NAME
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }

# environment: box A keeps its own venv + model; any other box gets the packed env from tf_rerun
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi

# dataset: locked JSONs, checksums verified
mkdir -p $D && cd $D
aws s3 cp --only-show-errors $B/datasets/cmdad_qwen38_ablations/SHA256SUMS .
aws s3 cp --only-show-errors $B/datasets/cmdad_qwen38_ablations/dataset_info.json .
for s in f150_40k eval527; do aws s3 cp --only-show-errors $B/datasets/cmdad_qwen38_ablations/${NAME}_$s.json .; done
grep -E "${NAME}_|dataset_info" SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
cd $HOME

# frames: needed unless the run has none (A2); use /data if this box has them, else pull the S3 copy made by box A
NEEDF=$(python3 -c "import json;print(int(any(r['images'] for r in json.load(open('$D/${NAME}_f150_40k.json'))[:50])))")
if [ "$NEEDF" = 1 ] && [ ! -d /data/cmd_ad/frames/1fps_448x252 ]; then
  until aws s3 ls $B/frames/_DONE >/dev/null 2>&1; do sleep 120; . $W/creds.env; done
  aws s3 sync --only-show-errors $B/frames/1fps_448x252/ $W/frames/1fps_448x252/
  sudo mkdir -p /data/cmd_ad/frames && sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252
fi
$HOME/adlm/venv/bin/python - <<EOF
import json, os
d = json.load(open("$D/${NAME}_f150_40k.json")) + json.load(open("$D/${NAME}_eval527.json"))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"DATASET $NAME {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing {miss}")
assert miss == 0
EOF
[ $? = 0 ] || { echo DATASET_CHECK_FAILED; exit 1; }

export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$W/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=$W/wandb WANDB_PROJECT=adlm-ablations
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p $W/wandb $W/tokcache $W/runs
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo "GPUS $(nvidia-smi -L | wc -l) (want $NG) | effective batch = 1 x $GA x $NG = $((GA * NG))"

( while true; do . $W/creds.env
    for c in $OUT/checkpoint-*; do [ -f $c/trainer_state.json ] || continue; n=$(basename $c)
      [ -f $W/synced_$n ] && continue
      sleep 20; aws s3 sync --only-show-errors $c $M/$n/ && touch $W/synced_$n && echo "$(date +%T) SYNCED $n" >> $W/ckpt_sync.log
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    sleep 60; done ) &
( while [ ! -f $W/synced_checkpoint-500 ]; do sleep 60; done
  echo "$(date +%T) EPOCH2_STOP" >> $W/ckpt_sync.log; pkill -f "qwen38_bins_frames_full.yaml" ) &

cd $LF && echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins_frames_full.yaml \
  dataset_dir=$D dataset=${NAME}_f150_40k tokenized_path=$W/tokcache/$NAME output_dir=$OUT run_name=qwen38-$NAME \
  gradient_accumulation_steps=$GA preprocessing_num_workers=$(( $(nproc) > 48 ? 48 : $(nproc) )) \
  > $W/runs/train_$NAME.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
. $W/creds.env; aws s3 cp --only-show-errors $W/runs/train_$NAME.log $M/train.log
