# Box C: resume qwen38_bins02_frames_full from checkpoint-$FROM (default 500) and finish epoch 3 (step 729).
# /scratch was wiped on stop: re-pull code (git bundle, commit-checked), dataset and the full checkpoint-500 (optimizer + rng) from S3.
WANT=$1; FROM=${2:-500}
set -x
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme2n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme2n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mv /tmp/crC.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
export PATH=$HOME/.local/bin:$PATH
aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1/adlm
R=/scratch/adlm_repo; LF=$R/training/LLaMA-Factory; OUT=/scratch/runs/qwen38_bins02_frames_full; M=$B/models/cmdad/qwen38-dd_bins02
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle /scratch/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 cp --only-show-errors --recursive $B/datasets/cmdad_qwen38_bins02/ /scratch/bins02/
aws s3 cp --only-show-errors --recursive $M/checkpoint-$FROM/ $OUT/checkpoint-$FROM/
ls $OUT/checkpoint-$FROM | wc -l
for n in $(seq 20 20 $FROM); do touch /scratch/synced_checkpoint-$n; done      # already in S3

export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/scratch/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=/scratch/wandb WANDB_PROJECT=adlm-bins-frames
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p /scratch/wandb /scratch/tokcache
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"

( while true; do . /scratch/creds.env
    for c in $OUT/checkpoint-*; do [ -f $c/trainer_state.json ] || continue; n=$(basename $c)
      [ -f /scratch/synced_$n ] && continue
      sleep 20; aws s3 sync --only-show-errors $c $M/$n/ && touch /scratch/synced_$n && echo "$(date +%T) SYNCED $n" >> /scratch/ckpt_sync.log
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    sleep 60; done ) &

cd $LF && echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins02_frames_full.yaml resume_from_checkpoint=$OUT/checkpoint-$FROM > /scratch/runs/train_bins02_resume.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
