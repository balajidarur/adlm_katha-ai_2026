# Box A queue (2026-10-28): REFRAMED no-budget ckpt-654 on the test set (teacher-forced with the fixed trainer, then
# free-running greedy with max_new_tokens 5120: 2/91 test GT targets exceed 4096), then CMD-AD free-running greedy
# max_new_tokens 4096 for Gemini budget 696 and Gemini no-budget 696 (527 eval clips). 4-GPU DDP per run.
# Outputs: /scratch/inf/out/<run>/{teacher_forced,generated}_predictions.jsonl, each synced to S3 adlm/preds/free4096/<run>/.
O=/scratch/inf/out; LF=~/adlm/LLaMA-Factory; mkdir -p $O
B=s3://balu-transfer-933516006154-apne1/adlm/preds/free4096
cd $LF && export PATH=$HOME/.local/bin:$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
mk(){ # name adapter budget(0/1) dataset_dir eval_dataset mode(tf|free:N) [extra]
  { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $2"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    [ $3 = 1 ] && { echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"; }
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $4"; echo "eval_dataset: $5"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $6 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${6#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
    [ -n "$7" ] && echo "$7"; } > examples/budget_ad/q_$1.yaml; }
RF=/scratch/rf_datasets/frame_variants_448x252; RC=/scratch/runs/reframed_qwen38_nobudget_frames/checkpoint-654
D=/data/cmd_ad/datasets
mk smoke_rf_free        $RC 0 $RF reframed_nobudget_f150_999k_test free:5120 "max_samples: 4"
mk rf_nobudget654_test_tf   $RC 0 $RF reframed_nobudget_f150_999k_test tf
mk rf_nobudget654_test_free $RC 0 $RF reframed_nobudget_f150_999k_test free:5120
mk gemini_budget_696_free   /data/cmd_ad/runs/bins_frames_full/checkpoint-696 1 $D/bins_frames eval_full free:4096
mk gemini_nobudget_696_free /scratch/inf/ckpts/gemini_nobudget_696 0 $D/nobudget_frames eval_full free:4096
for r in smoke_rf_free rf_nobudget654_test_tf rf_nobudget654_test_free gemini_budget_696_free gemini_nobudget_696_free; do
  f=$O/$r/generated_predictions.jsonl; [[ $r == *_tf ]] && f=$O/$r/teacher_forced_predictions.jsonl
  [ -f $f ] && [ $r != smoke_rf_free ] && { echo "SKIP $r (done)"; continue; }
  echo "$(date +%T) START $r"
  llamafactory-cli train examples/budget_ad/q_$r.yaml > $O/$r.log 2>&1; rc=$?
  n=$(wc -l < $f 2>/dev/null)
  echo "$(date +%T) END $r exit=$rc rows=${n:-0}"
  if [ $r = smoke_rf_free ]; then [ "${n:-0}" -ge 4 ] || { echo SMOKE_FAILED; exit 1; }
  else aws s3 sync --quiet $O/$r $B/$r/ --exclude "*" --include "*.jsonl" --include "*.json"; aws s3 cp --quiet $O/$r.log $B/$r/; fi
done
echo ALL_A_DONE
