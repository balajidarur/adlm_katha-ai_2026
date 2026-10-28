# shared: yaml generator + runner for the queues (sourced). mk name adapter budget dataset_dir eval_dataset mode(tf|free:N)
O=/scratch/inf/out; LF=~/adlm/LLaMA-Factory; S=s3://balu-transfer-933516006154-apne1/adlm/preds/free4096
cd $LF && export PATH=$HOME/.local/bin:$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
mk(){ { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $2"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    [ $3 = 1 ] && { echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"; }
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $4"; echo "eval_dataset: $5"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $6 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${6#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/q_$1.yaml; }
sync_out(){ aws s3 sync --quiet $O/$1 $S/$1/ --exclude "*" --include "*.jsonl" --include "*.json"; aws s3 cp --quiet $O/$1.log $S/$1/; }
run(){ echo "$(date +%T) START $1" >> /scratch/inf/queue.log
  MASTER_PORT=$((29600 + RANDOM % 300)) llamafactory-cli train examples/budget_ad/q_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log; sync_out $1; }
adopt(){ # wait for an already-running job (its queue script was replaced), then log + sync it
  while pgrep -f "q_$1.yaml" > /dev/null; do sleep 30; done
  echo "$(date +%T) END $1 rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log; sync_out $1; }
