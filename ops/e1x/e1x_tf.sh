# E1X: teacher-forced eval of the no-budget model (qwen38-dd_nobudget ckpt-480) on shard SH of NS (clips i with i % NS == SH)
# of eval591_nobudget.json. Settings = E1/T2 runs (cutoff 65536, bs 1, TF prefill chunk 2048, trainer md5 955de66d8a) minus the
# budget tokens (no new_special_tokens_config, as the no-budget eval527 run / ops/a_gemini_nb600_tf2.sh).
#   bash e1x_tf.sh <commit> <SH> <NS>   -> s3 adlm/inference/cmdad/eval591/_e1x_raw/nobudget_ckpt480/tf/s<SH>of<NS>/
WANT=$1; SH=$2; NS=$3; W=/scratch; R=$W/adlm_repo; LF=$R/training/LLaMA-Factory
B=s3://balu-transfer-933516006154-apne1; P=$B/adlm/inference/cmdad/eval591/_e1x_raw/nobudget_ckpt480/tf/s${SH}of${NS}
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
NAME=e1x_nb480_tf_s${SH}of${NS}; D=$W/e1x/evalds/$NAME; O=$W/e1x/inf/$NAME; CKD=$W/ckpt/qwen38-dd_nobudget_ckpt480; mkdir -p $D; rm -rf $O
python3 - <<PY || { echo PREP_FAIL; exit 1; }
import json, os
d = json.load(open("$W/e1x/ds/eval591_nobudget.json")); keep = [i for i in range(len(d)) if i % $NS == $SH]
json.dump([d[i] for i in keep], open("$D/$NAME.json", "w"), ensure_ascii=False); json.dump(keep, open("$D/idxmap.json", "w"))
json.dump({"$NAME": {"file_name": "$NAME.json", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}},
  open("$D/dataset_info.json", "w"), indent=2)
miss = sum(not os.path.exists(p) for i in keep for p in d[i]["images"]); print("SHARD", len(keep), "missing", miss); assert miss == 0
PY
Y=$LF/examples/budget_ad/$NAME.yaml
{ echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $CKD"; echo "trust_remote_code: true"
  echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
  echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
  echo "dataset_dir: $D"; echo "eval_dataset: $NAME"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 65536"
  echo "overwrite_cache: true"; echo "output_dir: $O"; echo "overwrite_output_dir: true"; echo "report_to: none"
  echo "per_device_eval_batch_size: 1"; echo "bf16: true"; echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048"
} > $Y
echo "$(date -u +%T) START $NAME" >> $W/e1x/queue.log
( cd $LF && CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 llamafactory-cli train $Y > $O.log 2>&1 ); rc=$?
n=$(cat $O/stream/*.jsonl 2>/dev/null | wc -l)
echo "$(date -u +%T) END $NAME exit=$rc rows=$n" >> $W/e1x/queue.log
. $W/cr.env; cp $D/idxmap.json $O/; cp $Y $O/run.yaml
aws s3 sync --only-show-errors $O $P/ --exclude "*" --include "*.jsonl" --include "*.json" --include "*.yaml" && aws s3 cp --only-show-errors $O.log $P/run.log \
  && echo "$(date -u +%T) SYNCED $NAME" >> $W/e1x/queue.log
