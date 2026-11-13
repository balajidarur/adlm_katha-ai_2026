#!/bin/bash
# xeval shard runner (one GPU): teacher-forced (TF2 settings: budget_teacher_forced, prefill chunk 2048, cutoff 65536) or free-running
# (greedy, max_new_tokens N, cutoff 49152, as the REFRAMED challenge run). Bin tokens only for budget models.
#   bash xe_infer.sh <commit> <ckpt_dir> <tag> <json> <gpu> <shard> <nshard> <bins|nobins> <tf|free:N>
# -> s3 adlm/xeval/preds/<tag>_s<shard>of<nshard>/ (predictions jsonl + ids.json + run.yaml + run.log)
WANT=$1; CKD=$2; TAG=$3; SPLIT=$4; GPU=$5; SH=$6; NS=$7; TOK=$8; MODE=$9; W=/scratch; R=$W/adlm_repo; LF=$R/training/LLaMA-Factory
B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1 PATH=$HOME/.local/bin:/usr/local/bin:$PATH
HAVE=$(git -C $R rev-parse --short HEAD); [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH $HAVE; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets_g$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
NAME=${TAG}_s${SH}of${NS}; D=$W/evalds/$NAME; O=$W/inf/$NAME; mkdir -p $D $W/inf; rm -rf $O
python3 - <<PY
import json
d = json.load(open("$SPLIT")); keep = [i for i in range(len(d)) if i % $NS == $SH]
json.dump([d[i] for i in keep], open("$D/$NAME.json", "w"), ensure_ascii=False); json.dump(keep, open("$D/ids.json", "w"))
json.dump({"$NAME": {"file_name": "$NAME.json", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}},
  open("$D/dataset_info.json", "w"), indent=2)
print("SHARD", len(keep))
PY
Y=$LF/examples/budget_ad/$NAME.yaml
{ echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $CKD"; echo "trust_remote_code: true"
  echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
  [ "$TOK" = bins ] && { echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"; }
  echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
  echo "dataset_dir: $D"; echo "eval_dataset: $NAME"; echo "template: qwen3_5_nothink"
  echo "overwrite_cache: true"; echo "output_dir: $O"; echo "overwrite_output_dir: true"; echo "report_to: none"
  echo "per_device_eval_batch_size: 1"; echo "bf16: true"
  case $MODE in tf) echo "cutoff_len: 65536"; echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
    free:*) echo "cutoff_len: 49152"; echo "predict_with_generate: true"; echo "max_new_tokens: ${MODE#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
} > $Y
( cd $LF && CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$((29500 + GPU)) llamafactory-cli train $Y > $O.log 2>&1 ); rc=$?
cp $D/ids.json $O/ 2>/dev/null; cp $Y $O/run.yaml; n=$(cat $O/*_predictions.jsonl 2>/dev/null | wc -l)
echo "END $NAME exit=$rc rows=$n"
. $HOME/.cr.env; aws s3 sync --only-show-errors $O $B/adlm/xeval/preds/$NAME/ --exclude "*" --include "*.jsonl" --include "*.json" --include "*.yaml" \
  && aws s3 cp --only-show-errors $O.log $B/adlm/xeval/preds/$NAME/run.log && echo "SYNCED $NAME"
