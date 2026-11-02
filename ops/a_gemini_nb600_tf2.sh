# Box A: fixed teacher-forced eval (both TF fixes, trainer md5 955de66d8a) of the Gemini-DD no-budget model at
# checkpoint-600 - the checkpoint whose free-running output is the registry M2 result (DVC 10.65), so TF and free-running
# use the same checkpoint. Same dataset/eval split/yaml as gemini_nobudget_696_tf2 (nobudget_frames, eval_full).
#   bash a_gemini_nb600_tf2.sh <commit>
WANT=$1
set -x
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme3n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mkdir -p /scratch/inf/out; mv /tmp/crA.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
export PATH=$HOME/.local/bin:$PATH
aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1/adlm
R=/scratch/adlm_repo; LF=$R/training/LLaMA-Factory; CK=/scratch/ckpt/gemini_nobudget_600; D=/data/cmd_ad/datasets/nobudget_frames
O=/scratch/inf/out; N=gemini_nobudget_600_tf2; P=$B/inference/cmdad/eval527/gemini-dd_nobudget_ckpt600/teacher_forced
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle /scratch/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 cp --only-show-errors --recursive $B/models/cmdad/gemini-dd_nobudget/checkpoint-600/ $CK/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
ls $CK/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
ls $D/dataset_info.json || { echo DATASET_MISSING; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/scratch/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
cd $LF
{ echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $CK"; echo "trust_remote_code: true"
  echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
  echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
  echo "dataset_dir: $D"; echo "eval_dataset: eval_full"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
  echo "overwrite_cache: true"; echo "output_dir: $O/$N"; echo "overwrite_output_dir: true"; echo "report_to: none"
  echo "per_device_eval_batch_size: 1"; echo "bf16: true"; echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048"
} > examples/budget_ad/q_$N.yaml
echo "$(date +%T) START $N" >> /scratch/inf/queue.log
llamafactory-cli train examples/budget_ad/q_$N.yaml > $O/$N.log 2>&1; rc=$?
echo "$(date +%T) END $N exit=$rc rows=$(cat $O/$N/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log
. /scratch/creds.env
aws s3 sync --only-show-errors $O/$N $P/ --exclude "*" --include "*.jsonl" --include "*.json"
aws s3 cp --only-show-errors $O/$N.log $P/ && echo "$(date +%T) SYNCED $N" >> /scratch/inf/queue.log
