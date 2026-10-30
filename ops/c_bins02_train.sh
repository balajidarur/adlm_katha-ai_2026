# Box C (2026-10-31): train Qwen3.8 CMD-AD with 0.2 s budget bins (26 tokens), 3-epoch schedule,
# stopped once the epoch-2 checkpoints (480 = 1.98 ep, 500 = 2.06 ep) are saved and in S3.
# Code = git commit $WANT from the bundle; everything is written to /scratch (never /data on C).
WANT=$1
set -x
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme2n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme2n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mv /tmp/crC.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
export PATH=$HOME/.local/bin:$PATH
aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1/adlm
R=/scratch/adlm_repo; LF=$R/training/LLaMA-Factory
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle /scratch/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }

# dataset (built on box A by dataset/add_bins02.py)
until aws s3 ls $B/datasets/cmdad_qwen38_bins02/dataset_info.json >/dev/null 2>&1; do sleep 30; . /scratch/creds.env; done
aws s3 cp --only-show-errors --recursive $B/datasets/cmdad_qwen38_bins02/ /scratch/bins02/
~/adlm/venv/bin/python - <<'EOF'
import json, os, re, collections
d = json.load(open("/scratch/bins02/bins02_f150_40k.json"))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
toks = collections.Counter(re.findall(r"<\|budget_\d+\|>", " ".join(r["messages"][1]["content"] for r in d)))
print(f"DATASET {len(d)} clips | missing frames {miss} | {len(toks)} distinct budget tokens")
assert miss == 0
EOF
[ $? = 0 ] || { echo DATASET_CHECK_FAILED; exit 1; }

export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/scratch/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=/scratch/wandb WANDB_PROJECT=adlm-bins-frames
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p /scratch/wandb /scratch/runs /scratch/tokcache
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
OUT=/scratch/runs/qwen38_bins02_frames_full

# sync every saved checkpoint to S3 as it appears (re-reads creds each pass)
( while true; do . /scratch/creds.env
    for c in $OUT/checkpoint-*; do [ -f $c/trainer_state.json ] || continue; n=$(basename $c)
      [ -f /scratch/synced_$n ] && continue
      sleep 20; aws s3 sync --only-show-errors $c $B/models/cmdad/qwen38-dd_bins02/$n/ && touch /scratch/synced_$n && echo "$(date +%T) SYNCED $n" >> /scratch/ckpt_sync.log
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $B/models/cmdad/qwen38-dd_bins02/trainer_log.jsonl 2>/dev/null
    sleep 60; done ) &

# stop after the epoch-2 checkpoints are safely in S3
( while [ ! -f /scratch/synced_checkpoint-500 ]; do sleep 60; done
  echo "$(date +%T) EPOCH2_STOP" >> /scratch/ckpt_sync.log; pkill -f qwen38_bins02_frames_full.yaml ) &

cd $LF && echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins02_frames_full.yaml > /scratch/runs/train_bins02.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
