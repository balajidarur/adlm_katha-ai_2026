# E2: Qwen3.5-9B writer on all 591 CMD-AD eval clips, teacher-forced and/or free-running (max 2048), 1 shard per GPU.
#   [N_CLIPS=k] [GPU0=g] bash e2_eval.sh <commit> <ckpt_step> <tf|free|both> [n_gpus=4]
#   (N_CLIPS: first k clips, smoke test; GPU0: first GPU index, so two checkpoints can run side by side on 2+2 GPUs)
# Same inference settings as T2's m3_run.sh (M3 ckpt-460), except model_name_or_path = local Qwen3.5-9B snapshot and the
# adapter = qwen35_9b-dd_budget/checkpoint-N; its new_token_rows.safetensors (9 budget rows of embed_tokens/lm_head) is
# written in place by workflow.load_new_token_rows. Eval JSON: s3 adlm/datasets/cmdad_eval591_9b/eval591_9b.json
# (eval527 + T2's 64 rows, in T2's merged-prediction order). Outputs -> s3 adlm/inference/cmdad/eval591/qwen35_9b-dd_budget_ckpt<N>/
WANT=$1; CK=$2; MODES=$3; NG=${4:-4}; G0=${GPU0:-0}
set -x
W=/scratch; [ -d /opt/dlami/nvme ] && W=/opt/dlami/nvme
. $W/creds.env
B=s3://balu-transfer-933516006154-apne1; R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; MD=$W/models/Qwen3.5-9B
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
[ -f $MD/.complete ] || { echo NO_BASE_MODEL; exit 1; }
CKD=$W/eval_ckpt/checkpoint-$CK
if [ ! -f $CKD/new_token_rows.safetensors ]; then
  aws s3 sync --only-show-errors $B/adlm/models/cmdad/qwen35_9b-dd_budget/checkpoint-$CK/ $CKD/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
fi
ls -la $CKD; [ -f $CKD/new_token_rows.safetensors ] && [ -f $CKD/adapter_model.safetensors ] || { echo CKPT_INCOMPLETE; exit 1; }
E=$W/eval591_n${NG}_g$G0${N_CLIPS:+_c$N_CLIPS}; mkdir -p $E; cd $E
aws s3 sync --only-show-errors $B/adlm/datasets/cmdad_eval591_9b/ .
sha256sum -c SHA256SUMS || { echo SHA_MISMATCH; exit 1; }
# frames: 527 + 43 from adlm/frames, the other 21 from T2's cmd_ad_eval591 prefixes (frames_src64.json)
python3 - <<EOF > $E/frame_jobs.txt
import json
d = json.load(open("$E/eval591_9b.json")); src = json.load(open("$E/frames_src64.json"))
for v in sorted({r["images"][0].split("/")[-2] for r in d}):
    print(v, src.get(v, "adlm/frames/1fps_448x252"))
EOF
mkdir -p $W/frames/1fps_448x252
cat $E/frame_jobs.txt | xargs -P 48 -n 2 sh -c "[ -d $W/frames/1fps_448x252/\$0 ] || aws s3 sync --only-show-errors $B/\$1/\$0/ $W/frames/1fps_448x252/\$0/"
sudo mkdir -p "$(readlink -m /data)/cmd_ad/frames" && [ -e /data/cmd_ad/frames/1fps_448x252 ] || sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252
# shards: contiguous, idx offset recorded so predictions can be put back in eval591 order
python3 - <<EOF
import json, os
d = json.load(open("$E/eval591_9b.json")); d = d[:int(os.environ.get("N_CLIPS") or len(d))]  # N_CLIPS: smoke test only
n = $NG; k = -(-len(d) // n)
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"EVAL {len(d)} clips | {sum(len(r['images']) for r in d):,} frames | missing {miss}"); assert miss == 0
info = {}
for i in range(n):
    part = d[i * k:(i + 1) * k]; name = f"e591s{i}"
    json.dump(part, open(f"$E/{name}.json", "w"), ensure_ascii=False)
    info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
                  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}
    json.dump({"offset": i * k, "n": len(part)}, open(f"$E/{name}.offset.json", "w"))
json.dump(info, open("$E/dataset_info.json", "w"), indent=2)
EOF
[ $? = 0 ] || { echo FRAMES_MISSING; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$W/hf_home HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
cd $LF
O=$W/inf9b/ckpt$CK; mkdir -p $O
mk(){ { echo "model_name_or_path: $MD"; echo "adapter_name_or_path: $CKD"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $E"; echo "eval_dataset: $3"; echo "template: qwen3_5_nothink"; echo "cutoff_len: ${CUTOFF:-49152}"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $2 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${2#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/e2_$1.yaml; }
shard(){ i=$1
  for m in $MODES; do
    case $m in tf) T=tf; sub=teacher_forced;; free) T=free:2048; sub=free_running_max2048;; esac
    mk s${i}_$m $T e591s$i
    echo "$(date -u +%T) START s$i $m" >> $O/queue.log
    CUDA_VISIBLE_DEVICES=$((G0 + i)) MASTER_PORT=$((29500 + G0 + i)) llamafactory-cli train examples/budget_ad/e2_s${i}_$m.yaml > $O/s${i}_$m.log 2>&1; rc=$?
    echo "$(date -u +%T) END s$i $m exit=$rc rows=$(cat $O/s${i}_$m/*_predictions.jsonl 2>/dev/null | wc -l)" >> $O/queue.log
    . $W/creds.env; aws s3 sync --only-show-errors $O/s${i}_$m $B/adlm/inference/cmdad/eval591/qwen35_9b-dd_budget_ckpt$CK/shards/$sub/s$i/ \
      --exclude "*" --include "*.jsonl" --include "*.json"
    aws s3 cp --only-show-errors $O/s${i}_$m.log $B/adlm/inference/cmdad/eval591/qwen35_9b-dd_budget_ckpt$CK/shards/$sub/s$i/run.log
  done; }
[ "$MODES" = both ] && MODES="tf free"
for i in $(seq 0 $((NG - 1))); do shard $i & done; wait
aws s3 cp --only-show-errors $E/ $B/adlm/inference/cmdad/eval591/qwen35_9b-dd_budget_ckpt$CK/shards/ --recursive --exclude "*" --include "*.offset.json"
echo "ALL_DONE ckpt$CK $MODES" >> $O/queue.log
aws s3 cp --only-show-errors $O/queue.log $B/adlm/inference/cmdad/eval591/qwen35_9b-dd_budget_ckpt$CK/queue.log
