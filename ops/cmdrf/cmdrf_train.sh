# CMD-AD + REFRAMED combined budget model on box A (4x L40S): M3 recipe (qwen38_bins_frames_full.yaml: Qwen3.5-4B + LoRA,
# 0.5 s bins, Qwen3.8 dense descriptions, effective batch 32, LR 1e-4, cosine), dataset = build_cmdrf.py output, 2 epochs.
#   bash cmdrf_train.sh <commit>
# Workdir = instance NVMe /scratch (wiped on stop): every complete checkpoint is synced to S3 and size-verified as it lands.
WANT=$1
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
W=/scratch
mountpoint -q $W || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p $W && sudo mount /dev/nvme3n1 $W && sudo chown ubuntu:ubuntu $W; }
[ -f /tmp/cr.env ] && mv /tmp/cr.env $W/creds.env; chmod 600 $W/creds.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1
R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; M=$B/adlm/models/cmdrf/qwen38-dd_budget_cmdad_reframed
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle $W/adlm.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }

# data: REFRAMED frames + budget dataset (same tarball the REFRAMED runs used) -> /scratch/rf_frames, /scratch/rf_datasets
[ -f $W/rf_datasets/frame_variants_448x252/reframed_bins_f150_40k_train.json ] || \
  aws s3 cp --only-show-errors $B/reframed/train_data/rf_frames_datasets.tar.gz - | pigz -dc | tar -x -C $W
D=$W/cmdrf_ds; P=$HOME/adlm/venv/bin/python
$P -u $R/ops/cmdrf/build_cmdrf.py /data/cmd_ad/datasets/frame_variants_448x252_full_qwen38/gapframes_bins_train_f150_40k.json \
  $W/rf_datasets/frame_variants_448x252/reframed_bins_f150_40k_train.json $D || { echo BUILD_FAILED; exit 1; }
$P - <<EOF || { echo DATASET_CHECK_FAILED; exit 1; }
import json, os
d = json.load(open("$D/cmdrf_bins_f150_40k.json"))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"DATASET {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing {miss}")
assert miss == 0
EOF
sha256sum $D/*.json > $D/SHA256SUMS; aws s3 sync --only-show-errors $D $B/adlm/datasets/cmdrf_bins_f150_40k/

export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$W/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=$W/wandb WANDB_PROJECT=adlm-cmdrf
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p $W/wandb $W/tokcache $W/runs
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo "GPUS $(nvidia-smi -L | wc -l) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

OUT=$W/runs/qwen38-dd_budget_cmdad_reframed; SV=${SAVE_STEPS:-50}; NG=4
complete(){ local c=$1; for f in trainer_state.json optimizer.pt scheduler.pt adapter_model.safetensors; do [ -s $c/$f ] || return 1; done
  [ "$(ls $c/rng_state*.pth 2>/dev/null | wc -l)" = "$NG" ] || return 1; return 0; }
( while true; do . $W/creds.env
    for c in $OUT/checkpoint-*; do n=$(basename $c); [ -f $W/synced_$n ] && continue
      complete $c || continue; sleep 30
      aws s3 sync --only-show-errors $c $M/$n/ || continue
      L=$(find $c -type f -printf "%s\n" | awk '{s+=$1} END {print s}')
      Rz=$(aws s3 ls --recursive $M/$n/ | awk '{s+=$3} END {print s}')
      if [ "$L" = "$Rz" ]; then touch $W/synced_$n; echo "$(date +%T) SYNCED+VERIFIED $n $L bytes" >> $W/ckpt_sync.log
      else echo "$(date +%T) SIZE_MISMATCH $n local $L s3 $Rz (retry)" >> $W/ckpt_sync.log; fi
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    aws s3 cp --only-show-errors $W/runs/train.log $M/train.log 2>/dev/null
    sleep 60; done ) &
cd $LF
echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins_frames_full.yaml \
  dataset_dir=$D dataset=cmdrf_bins_f150_40k tokenized_path=$W/tokcache/cmdrf_bins_f150_40k \
  output_dir=$OUT run_name=qwen38-dd_budget_cmdad_reframed num_train_epochs=2.0 \
  save_steps=$SV save_total_limit=null save_only_model=false > $W/runs/train.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
. $W/creds.env; aws s3 cp --only-show-errors $W/runs/train.log $M/train.log
