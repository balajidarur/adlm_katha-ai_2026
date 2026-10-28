# Box C: CMD-AD free-running greedy max_new_tokens 4096 (527 clips) for Qwen3.8 budget ckpt-460 and no-budget ckpt-480,
# both at once after c_rf_eval.sh is done. Eval pack from S3 is unpacked under /scratch/inf/root and its image paths
# rewritten there (box C /data is read-only for us). ckpt-480 is read from box C /data.
. /scratch/creds.env
R=/scratch/inf/root; mkdir -p $R && cd /scratch/inf && tar xzf evaldata.tgz -C $R
python3 - <<'PY'
import glob, json
for f in glob.glob("/scratch/inf/root/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38/**/*eval527*.json", recursive=True):
    d = json.load(open(f))
    for r in d: r["images"] = [p.replace("/data/cmd_ad/", "/scratch/inf/root/data/cmd_ad/") for p in r["images"]]
    json.dump(d, open(f, "w"), ensure_ascii=False); print(f, len(d), d[0]["images"][0])
PY
while ! grep -q ALL_C_DONE /scratch/inf/queue.log; do sleep 60; done
O=/scratch/inf/out; LF=~/adlm/LLaMA-Factory; S=s3://balu-transfer-933516006154-apne1/adlm/preds/free4096
cd $LF && export PATH=$HOME/.local/bin:$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
mk(){ { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $2"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    [ $3 = 1 ] && { echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"; }
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $4"; echo "eval_dataset: $5"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    echo "predict_with_generate: true"; echo "max_new_tokens: 4096"; echo "do_sample: false"; echo "skip_special_tokens: false"
  } > examples/budget_ad/q_$1.yaml; }
run(){ echo "$(date +%T) START $1" >> /scratch/inf/queue.log
  MASTER_PORT=$2 llamafactory-cli train examples/budget_ad/q_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> /scratch/inf/queue.log
  aws s3 sync --quiet $O/$1 $S/$1/ --exclude "*" --include "*.jsonl" --include "*.json"; aws s3 cp --quiet $O/$1.log $S/$1/; }
Q=$R/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38
mk qwen38_budget_460_free   /scratch/inf/ckpts/qwen38_budget_460 1 $Q eval527
mk qwen38_nobudget_480_free /data/cmd_ad/runs/qwen38_nobudget_frames_full/checkpoint-480 0 $Q/nobudget nobudget_eval527
run qwen38_budget_460_free 29531 & sleep 60; run qwen38_nobudget_480_free 29532 & wait
echo "$(date +%T) ALL_C_CMD_DONE" >> /scratch/inf/queue.log
