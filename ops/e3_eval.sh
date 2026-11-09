# E3 / M8 eval on s4 (4x L40S, after training stops): a Qwen3-VL-8B-DD budget checkpoint on all 591 CMD-AD eval clips,
# exactly like M3-591 (t2/ops/m3_run.sh, g_dd4b_infer.sh): teacher-forced (fixed TF code, trainer md5 955de66d8a), then
# free-running (greedy, max_new_tokens 2048). cutoff_len 65536 as for M3-591's long clips (max here 51.5k tokens).
# The rows-only checkpoint's budget rows (new_token_rows.safetensors) are written into embed_tokens/lm_head at load.
#   bash e3_eval.sh <commit> <step> [modes: "tf free" | trainchk]
#   trainchk = TF on the first 8 TRAIN clips (overfit-style sanity check of the TF path on Qwen3-VL).
WANT=$1; STEP=$2; MODES=${3:-tf free}
set -x
W=/opt/dlami/nvme/e3; . $W/cr.env; B=s3://balu-transfer-933516006154-apne1
R=$W/adlm_repo_eval; LF=$R/training/LLaMA-Factory; MD=$W/models/Qwen3-VL-8B-Instruct
N=qwen3vl8b-dd_budget_ckpt$STEP; CK=$W/runs/qwen3vl8b-dd_budget/checkpoint-$STEP; O=$W/inf; D=$W/evalds
P=$B/adlm/inference/cmdad/eval591/$N
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle $W/adlm_eval.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm_eval.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
# fresh instance disk: base model, and the eval clips' frames (shared pool + the frames E3's builds added)
sudo touch /etc/gpu-idle-shutdown.disable
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
[ -f $MD/model-00004-of-00004.safetensors ] || aws s3 sync --only-show-errors $B/models/Qwen3-VL-8B-Instruct/ $MD/
mkdir -p $W/frames/1fps_448x252
sudo mkdir -p /data/cmd_ad/frames; [ "$(readlink /data/cmd_ad/frames/1fps_448x252)" = $W/frames/1fps_448x252 ] || { sudo rm -f /data/cmd_ad/frames/1fps_448x252; sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252; }
[ -d $CK ] || aws s3 cp --only-show-errors --recursive $B/adlm/models/cmdad/qwen3vl8b-dd_budget/checkpoint-$STEP/ $CK/ \
  --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
ls $CK/adapter_model.safetensors $CK/new_token_rows.safetensors || { echo CKPT_INCOMPLETE; exit 1; }
mkdir -p $D $O
DS=$B/cmd_ad_dense/datasets/cmdad_q3vl8b_dd
for f in SHA256SUMS q3vl8b_eval591.json q3vl8b_f150_40k.json; do [ -f $D/$f ] || aws s3 cp --only-show-errors $DS/$f $D/; done
(cd $D && grep -E "q3vl8b_eval591.json|q3vl8b_f150_40k.json" SHA256SUMS | sha256sum -c -) || { echo SHA_MISMATCH; exit 1; }
if [ ! -f $W/frames/.eval591_ok ]; then
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for f in ['$D/q3vl8b_eval591.json'] for r in json.load(open(f)) for p in r['images']})))" > $D/vids.txt
  cat $D/vids.txt | xargs -P 48 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/
  # per source, as T2/E1X: 16 HF clips and the 7 T2 rebuilds have their frames under cmd_ad_eval591/ (S3 mtimes predate
  # E3's builds, so they are not in frames_new); then E3's newly extracted frames on top
  F=$W/frames/1fps_448x252; E=$R/ops/e1x
  for v in $(cat $E/vids_hf.txt); do rm -rf $F/$v; aws s3 sync --only-show-errors $B/cmd_ad_eval591/frames_hf_1fps_448x252/$v/ $F/$v/; done
  for v in $(cat $E/vids_t2.txt); do rm -rf $F/$v; aws s3 sync --only-show-errors $B/cmd_ad_eval591/frames_1fps_448x252/$v/ $F/$v/; done
  aws s3 cp --only-show-errors $DS/frames_new_1fps_448x252.tar - | tar -x -C $W/frames && touch $W/frames/.eval591_ok
fi
python3 - <<PY
import json, os
d = json.load(open("$D/q3vl8b_eval591.json"))
json.dump(json.load(open("$D/q3vl8b_f150_40k.json"))[:8], open("$D/trainchk8.json", "w"), ensure_ascii=False)
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"EVAL591 {len(d)} clips | {sum(len(r['images']) for r in d):,} frames | missing {miss}"); assert miss == 0
t = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
     "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}
json.dump({"q3vl8b_eval591": dict(t, file_name="q3vl8b_eval591.json"), "trainchk8": dict(t, file_name="trainchk8.json")},
          open("$D/dataset_info.json", "w"), indent=2)
PY
[ $? = 0 ] || { echo EVAL_DATA_CHECK_FAILED; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$W/hf_home HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
cd $LF
mk(){ { echo "model_name_or_path: $MD"; echo "adapter_name_or_path: $CK"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $D"; echo "eval_dataset: $3"; echo "template: qwen3_vl_nothink"; echo "cutoff_len: 65536"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $2 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${2#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/e3_$1.yaml; }
run(){ echo "$(date +%T) START $1" >> $O/queue.log
  llamafactory-cli train examples/budget_ad/e3_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> $O/queue.log
  grep -m3 -E "new_token_rows_only|budget rows written" $O/$1.log >> $O/queue.log
  [ -n "$2" ] && { . $W/cr.env; aws s3 sync --only-show-errors $O/$1 $P/$2/ --exclude "*" --include "*.jsonl" --include "*.json"
    aws s3 cp --only-show-errors $O/$1.log $P/$2/run.log && echo "$(date +%T) SYNCED $1 -> $P/$2" >> $O/queue.log; }; }
for m in $MODES; do case $m in
  trainchk) mk ${N}_trainchk tf trainchk8; run ${N}_trainchk "";;
  tf) mk ${N}_tf tf q3vl8b_eval591; run ${N}_tf teacher_forced;;
  free) mk ${N}_free free:2048 q3vl8b_eval591; run ${N}_free ${FREE_SUB:-free_running_max2048};;
esac; done
echo "ALL_DONE $N $MODES" >> $O/queue.log
