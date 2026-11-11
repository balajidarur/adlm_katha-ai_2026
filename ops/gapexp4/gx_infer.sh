#!/bin/bash
# gapexp4 free-running shard runner (CMD-AD gap-shift records): M3/E1 predict settings, greedy, bs1, cutoff 65536, max_new_tokens 2048.
# cutoff 65536) with max_new_tokens 8192 (whole 1-9 min videos).   bash gx_infer.sh <commit> <ckpt_dir> <tag> <json> <gpu> <shard> <nshard>
WANT=$1; CKD=$2; TAG=$3; SPLIT=$4; GPU=$5; SH=$6; NS=$7; W=/scratch; R=$W/adlm_repo; LF=$R/training/LLaMA-Factory
B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
HAVE=$(git -C $R rev-parse --short HEAD); [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH $HAVE; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets_g$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
NAME=${TAG}_s${SH}of${NS}; D=$W/evalds/$NAME; O=$W/inf/$NAME; mkdir -p $D $W/inf; rm -rf $O
python3 - <<PY
import json
d = json.load(open("$SPLIT")); ids = list(range(len(d)))
keep = [i for i in range(len(d)) if i % $NS == $SH]
json.dump([d[i] for i in keep], open("$D/$NAME.json", "w"), ensure_ascii=False); json.dump([ids[i] for i in keep], open("$D/ids.json", "w"))
json.dump({"$NAME": {"file_name": "$NAME.json", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}},
  open("$D/dataset_info.json", "w"), indent=2)
print("SHARD", len(keep))
PY
Y=$LF/examples/budget_ad/$NAME.yaml
cat > $Y <<Y2
model_name_or_path: Qwen/Qwen3.5-4B
adapter_name_or_path: $CKD
trust_remote_code: true
image_max_pixels: 589824
image_min_pixels: 1024
enable_liger_kernel: true
new_special_tokens_config: examples/budget_ad/bin_tokens.yaml
init_special_tokens: desc_init
resize_vocab: false
stage: sft
do_predict: true
finetuning_type: lora
budget_variant: none
dataset_dir: $D
eval_dataset: $NAME
template: qwen3_5_nothink
cutoff_len: 65536
overwrite_cache: true
output_dir: $O
overwrite_output_dir: true
report_to: none
per_device_eval_batch_size: 1
bf16: true
predict_with_generate: true
max_new_tokens: 2048
do_sample: false
skip_special_tokens: false
Y2
( cd $LF && CUDA_VISIBLE_DEVICES=$GPU MASTER_PORT=$((29500 + GPU)) llamafactory-cli train $Y > $O.log 2>&1 ); rc=$?
cp $D/ids.json $O/ 2>/dev/null; cp $Y $O/run.yaml; n=$(cat $O/stream/*.jsonl 2>/dev/null | wc -l)
echo "END $NAME exit=$rc rows=$n"
. $HOME/.cr.env; aws s3 sync --only-show-errors $O $B/gapexp4/preds/$NAME/ --exclude "*" --include "*.jsonl" --include "*.json" --include "*.yaml" && aws s3 cp --only-show-errors $O.log $B/gapexp4/preds/$NAME/run.log && echo "SYNCED $NAME"
