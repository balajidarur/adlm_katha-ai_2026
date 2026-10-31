# Box A: eval qwen38_bins02 checkpoint-480 (1.98 ep) on bins02_eval527 - teacher-forced first, then free-running (max_new_tokens 4096).
# Same eval clips/frames as the 0.5 s model's eval527 (rebinned in place), fixed TF code (commit-checked clone).
WANT=$1
set -x
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme3n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mkdir -p /scratch/inf/out; mv /tmp/crA.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1/adlm
R=/scratch/adlm_repo; LF=$R/training/LLaMA-Factory; CK=/scratch/ckpt/bins02_480; D=/scratch/bins02; O=/scratch/inf/out
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle /scratch/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 cp --only-show-errors --recursive $B/datasets/cmdad_qwen38_bins02/ $D/ --exclude "bins02_f150_40k.json"
aws s3 cp --only-show-errors --recursive $B/models/cmdad/qwen38-dd_bins02/checkpoint-480/ $CK/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
export PYTHONPATH=$LF/src PATH=$HOME/.local/bin:$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/scratch/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
cd $LF
mk(){ { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $CK"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    echo "new_special_tokens_config: examples/budget_ad/bin_tokens_02.yaml"; echo "init_special_tokens: desc_init"
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $D"; echo "eval_dataset: bins02_eval527"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $2 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${2#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/q_$1.yaml; }
run(){ echo "$(date +%T) START $1" >> /scratch/inf/queue.log
  llamafactory-cli train examples/budget_ad/q_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log
  . /scratch/creds.env; aws s3 sync --only-show-errors $O/$1 $B/_runs/2026-11-01/$1/ --exclude "*" --include "*.jsonl" --include "*.json"; aws s3 cp --only-show-errors $O/$1.log $B/_runs/2026-11-01/$1/; }
mk qwen38_bins02_480_tf2 tf
mk qwen38_bins02_480_free free:4096
run qwen38_bins02_480_tf2
run qwen38_bins02_480_free
echo ALL_DONE >> /scratch/inf/queue.log
