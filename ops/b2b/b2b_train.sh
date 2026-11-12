# Box C: M3 recipe trained on the Qwen3.5-2B-DD variant (s3 adlm/datasets/cmdad_qwen35_2b_dd, dd2b_f150_40k).
# qwen38_bins_frames_full.yaml unchanged except data/output paths (Qwen3.5-4B + LoRA, 0.5 s bins, eff. batch 32, LR 1e-4,
# 3-epoch cosine schedule, as the 4B-DD run); checkpoints every 20 steps, each synced to S3 and size-verified as it lands.
# Frames: the clips' 1 FPS pool from s3 adlm/frames/1fps_448x252 into /scratch/frames (never C's /data), JSON paths rewritten.
#   bash b2b_train.sh <commit>
WANT=$1
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
exec > $HOME/b2b_train.log 2>&1
sudo touch /etc/gpu-idle-shutdown.disable
W=/scratch; mountpoint -q $W || { echo NO_SCRATCH; exit 1; }
. $HOME/.cr.env; aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1; M=$B/adlm/models/cmdad/qwen35_2b-dd_bins
R=$W/b2b_repo; LF=$R/training/LLaMA-Factory
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle $W/b2b.bundle && rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/b2b.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
D=$W/b2b_ds; mkdir -p $D && cd $D
aws s3 sync --only-show-errors $B/adlm/datasets/cmdad_qwen35_2b_dd/ . --exclude "*" --include SHA256SUMS --include dataset_info.json --include build_report.json --include "dd2b_*.json"
grep -E "dd2b_f150_40k.json|dataset_info" SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('dd2b_f150_40k.json')) for p in r['images']})))" > $W/b2b_vids.txt
mkdir -p $W/frames/1fps_448x252
cat $W/b2b_vids.txt | xargs -P 48 -I{} sh -c "aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/"
sed -i 's#/data/cmd_ad/frames/1fps_448x252/#/scratch/frames/1fps_448x252/#g' dd2b_f150_40k.json
$HOME/adlm/venv/bin/python - <<EOF || { echo DATASET_CHECK_FAILED; exit 1; }
import json, os
d = json.load(open("$D/dd2b_f150_40k.json"))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"DATASET {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing {miss}")
assert miss == 0
EOF
export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$W/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=$W/wandb WANDB_PROJECT=adlm-dd2b
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p $W/wandb $W/tokcache $W/runs
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo "GPUS $(nvidia-smi -L | wc -l) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
OUT=$W/runs/qwen35_2b-dd_bins; SV=${SAVE_STEPS:-20}; NG=4
complete(){ local c=$1; for f in trainer_state.json optimizer.pt scheduler.pt adapter_model.safetensors; do [ -s $c/$f ] || return 1; done
  [ "$(ls $c/rng_state*.pth 2>/dev/null | wc -l)" = "$NG" ] || return 1; return 0; }
( while true; do . $HOME/.cr.env
    for c in $OUT/checkpoint-*; do n=$(basename $c); [ -f $W/b2b_synced_$n ] && continue
      complete $c || continue; sleep 30
      aws s3 sync --only-show-errors $c $M/$n/ || continue
      L=$(find $c -type f -printf "%s\n" | awk '{s+=$1} END {print s}')
      Rz=$(aws s3 ls --recursive $M/$n/ | awk '{s+=$3} END {print s}')
      if [ "$L" = "$Rz" ]; then touch $W/b2b_synced_$n; echo "$(date +%T) SYNCED+VERIFIED $n $L bytes" >> $W/b2b_ckpt_sync.log
      else echo "$(date +%T) SIZE_MISMATCH $n local $L s3 $Rz (retry)" >> $W/b2b_ckpt_sync.log; fi
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    aws s3 cp --only-show-errors $HOME/b2b_train.log $M/train.log 2>/dev/null
    sleep 60; done ) &
cd $LF
echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins_frames_full.yaml \
  dataset_dir=$D dataset=dd2b_f150_40k tokenized_path=$W/tokcache/dd2b_f150_40k \
  output_dir=$OUT run_name=qwen35_2b-dd_bins save_steps=$SV save_total_limit=null save_only_model=false > $W/runs/b2b_train.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
. $HOME/.cr.env; aws s3 cp --only-show-errors $W/runs/b2b_train.log $M/train_full.log
