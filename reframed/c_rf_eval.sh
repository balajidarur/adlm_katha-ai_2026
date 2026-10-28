# Box C (2026-10-28): REFRAMED budget ckpt-654 (3 epochs), test then validation; teacher-forced (fixed trainer) and
# free-running greedy run in parallel on the same 4 GPUs (4-GPU DDP each, own master port). max_new_tokens 6144:
# the GT test target reaches 4,235 tokens without bin tokens and the budget model adds one bin token per AD word.
. /scratch/creds.env
O=/scratch/inf/out; LF=~/adlm/LLaMA-Factory; mkdir -p $O; S=s3://balu-transfer-933516006154-apne1/adlm/preds/free4096
cd $LF && export PATH=$HOME/.local/bin:$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
mk(){ # name adapter budget(0/1) dataset_dir eval_dataset mode(tf|free:N)
  { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $2"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    [ $3 = 1 ] && { echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"; }
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $4"; echo "eval_dataset: $5"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $6 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${6#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/q_$1.yaml; }
run(){ echo "$(date +%T) START $1" >> /scratch/inf/queue.log
  MASTER_PORT=$2 llamafactory-cli train examples/budget_ad/q_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log
  aws s3 sync --quiet $O/$1 $S/$1/ --exclude "*" --include "*.jsonl" --include "*.json"; aws s3 cp --quiet $O/$1.log $S/$1/; }
RF=/scratch/rf_datasets/frame_variants_448x252; RC=/scratch/runs/reframed_qwen38_bins_frames/checkpoint-654
for sp in test validation; do
  mk rf_budget654_${sp}_tf   $RC 1 $RF reframed_bins_f150_999k_$sp tf
  mk rf_budget654_${sp}_free $RC 1 $RF reframed_bins_f150_999k_$sp free:6144
done
for sp in test validation; do
  run rf_budget654_${sp}_tf 29521 & sleep 60; run rf_budget654_${sp}_free 29522 & wait
done
echo "$(date +%T) ALL_C_DONE" >> /scratch/inf/queue.log
