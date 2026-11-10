# MADX: zero-shot MAD-Eval writer inference, teacher-forced on the GT AD slots only (T3 protocol: dialogue + every
# "M:SS.s [AD]" prefix at the GT AD start forced, AD text generated; fixed TF code, trainer md5 955de66d8a).
# Per-model settings = that checkpoint's own CMD-AD eval script:
#   e2 : Qwen3.5-9B (s3 adlm/models/base) + qwen35_9b-dd_budget/checkpoint-488 (new_token_rows), qwen3_5_nothink, cutoff 65536 (ops/e2_eval.sh)
#        dataset = T3's M3 MAD dataset unchanged (same Qwen3.8 DDs): s3 mad/run/ds/{ds_json,frames_*}.tgz
#   m6 : Qwen/Qwen3.5-4B (tf_rerun pack) + qwen35_4b-dd_bins/checkpoint-440, qwen3_5_nothink, cutoff 49152 (ops/g_dd4b_infer.sh)
#   e3 : Qwen3-VL-8B-Instruct (s3 models/) + qwen3vl8b-dd_budget/checkpoint-376 (new_token_rows), qwen3_vl_nothink, cutoff 65536 (ops/e3_eval.sh)
# Shards: interleaved (shard i = windows i, i+K, ...) of the full mad_bins_f150_999k_challenge.json; one GPU per box.
#   bash infer.sh <commit> <e2|m6|e3> <i> <K>
# Out: s3 adlm/inference/mad/<NAME>/teacher_forced/shards/s<i>of<K>/{teacher_forced_predictions.jsonl, idx.json, run.log}
WANT=$1; MODEL=$2; I=$3; K=$4
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
R=/scratch/adlm_repo; LF=$R/training/LLaMA-Factory; B=s3://balu-transfer-933516006154-apne1
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
cr(){ set -a; . ~/cr.env; set +a; }
cr
log(){ echo "$(date -u +%T) $*" | tee -a /scratch/madx_infer_$MODEL.log; }
case $MODEL in
  e2) NAME=e2_qwen35_9b-dd_budget_ckpt488; BASE=/scratch/models/Qwen3.5-9B; CKS=qwen35_9b-dd_budget/checkpoint-488; TPL=qwen3_5_nothink; CUT=65536;;
  m6) NAME=m6_qwen35_4b-dd_bins_ckpt440; BASE=Qwen/Qwen3.5-4B; CKS=qwen35_4b-dd_bins/checkpoint-440; TPL=qwen3_5_nothink; CUT=49152;;
  e3) NAME=e3_qwen3vl8b-dd_budget_ckpt376; BASE=/scratch/models/Qwen3-VL-8B-Instruct; CKS=qwen3vl8b-dd_budget/checkpoint-376; TPL=qwen3_vl_nothink; CUT=65536;;
  *) echo BAD_MODEL; exit 1;;
esac
P=$B/adlm/inference/mad/$NAME
# env: packed venv (+ Qwen3.5-4B HF cache) from s3 adlm/tf_rerun, as e2_box_setup.sh / g_dd4b_infer.sh
if ! $HOME/adlm/venv/bin/python -c "import liger_kernel, fla, transformers; assert transformers.__version__ == '5.8.0'" 2>/dev/null \
   || [ ! -d $HOME/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B ]; then
  mkdir -p /scratch/pack; aws s3 cp --only-show-errors --recursive $B/adlm/tf_rerun/ /scratch/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc /scratch/pack/$f.tgz | tar -x -C / ; done; rm -rf /scratch/pack
  ln -sfn cpython-3.11.16-linux-x86_64-gnu ~/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu 2>/dev/null
fi
case $MODEL in
  e2) [ -f $BASE/.complete ] || aws s3 sync --only-show-errors $B/adlm/models/base/Qwen3.5-9B/ $BASE/; [ -f $BASE/.complete ] || { log NO_BASE; exit 1; };;
  e3) [ -f $BASE/model-00004-of-00004.safetensors ] || aws s3 sync --only-show-errors $B/models/Qwen3-VL-8B-Instruct/ $BASE/;;
esac
CK=/scratch/ckpt/$NAME
[ -f $CK/adapter_model.safetensors ] || aws s3 cp --only-show-errors --recursive $B/adlm/models/cmdad/$CKS/ $CK/ \
  --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
ls -la $CK | tee -a /scratch/madx_infer_$MODEL.log
# dataset
D=/scratch/madx_ds_$MODEL; mkdir -p $D
if [ $MODEL = e2 ]; then
  if [ ! -f /scratch/rc_ds/.frames_ok ]; then
    mkdir -p /scratch/rc_ds && aws s3 cp --only-show-errors $B/mad/run/ds/ds_json.tgz - | tar xz -C /scratch --strip-components=1
    for b in A s4 s1a s1b; do aws s3 cp --only-show-errors $B/mad/run/ds/frames_$b.tgz - | pigz -dc | tar x -C /scratch --strip-components=1; done
    touch /scratch/rc_ds/.frames_ok
  fi
  SRC=/scratch/rc_ds/mad_bins_f150_999k_challenge.json; ORDER=/scratch/rc_ds/order.json
else
  if [ ! -f $D/.frames_ok ]; then
    for f in mad_bins_f150_999k_challenge.json order.json SHA256SUMS; do aws s3 cp --only-show-errors $P/dataset/$f $D/$f; done
    (cd $D && grep -E "mad_bins_f150_999k_challenge.json|order.json" SHA256SUMS | sha256sum -c -) || { log SHA_MISMATCH; exit 1; }
    aws s3 cp --only-show-errors $P/dataset/frames.tgz - | pigz -dc | tar x -C /scratch --strip-components=1 && touch $D/.frames_ok
  fi
  SRC=$D/mad_bins_f150_999k_challenge.json; ORDER=$D/order.json
fi
python3 - <<EOF || { log DATA_CHECK_FAILED; exit 1; }
import json, os
d = json.load(open("$SRC")); order = json.load(open("$ORDER")); assert len(d) == len(order) == 576, (len(d), len(order))
idx = list(range($I, len(d), $K)); part = [d[j] for j in idx]
miss = sum(not os.path.exists(p) for r in part for p in r["images"])
print(f"SHARD $I/$K: {len(part)} windows | {sum(len(r['images']) for r in part):,} frames | missing {miss}"); assert miss == 0
name = "mad_s${I}of$K"
json.dump(part, open(f"$D/{name}.json", "w"), ensure_ascii=False)
json.dump({"idx": idx, "wid": [order[j] for j in idx]}, open(f"$D/{name}.idx.json", "w"))
t = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
     "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system"}}
info = json.load(open("$D/dataset_info.json")) if os.path.exists("$D/dataset_info.json") else {}
info[name] = dict(t, file_name=f"{name}.json"); json.dump(info, open("$D/dataset_info.json", "w"), indent=2)
EOF
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=/scratch/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)
log "ENV commit=$HAVE trainer=$MD5 lf=$(python -c 'import llamafactory,os;print(os.path.dirname(llamafactory.__file__))') model=$NAME shard=$I/$K"
[ "$MD5" = 955de66d8a ] || { log TRAINER_MISMATCH; exit 1; }
N=${NAME}_tf_s${I}of$K; O=/scratch/madx_out/$N; mkdir -p /scratch/madx_out
cd $LF
{ echo "model_name_or_path: $BASE"; echo "adapter_name_or_path: $CK"; echo "trust_remote_code: true"
  echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
  echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"
  echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
  echo "dataset_dir: $D"; echo "eval_dataset: mad_s${I}of$K"; echo "template: $TPL"; echo "cutoff_len: $CUT"
  echo "overwrite_cache: true"; echo "output_dir: $O"; echo "overwrite_output_dir: true"; echo "report_to: none"
  echo "per_device_eval_batch_size: 1"; echo "bf16: true"; echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048"
} > /scratch/madx_out/$N.yaml
log "START $N"
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=$((29500 + I)) llamafactory-cli train /scratch/madx_out/$N.yaml > /scratch/madx_out/$N.log 2>&1; rc=$?
n=$(cat $O/teacher_forced_predictions.jsonl 2>/dev/null | wc -l)
grep -m3 -E "new_token_rows|budget rows written" /scratch/madx_out/$N.log | tee -a /scratch/madx_infer_$MODEL.log
cr; S3S=$P/teacher_forced/shards/s${I}of$K
aws s3 sync --only-show-errors $O $S3S/ --exclude "*" --include "*.jsonl" --include "*.json"
aws s3 cp --only-show-errors $D/mad_s${I}of$K.idx.json $S3S/idx.json; aws s3 cp --only-show-errors /scratch/madx_out/$N.log $S3S/run.log
aws s3 cp --only-show-errors /scratch/madx_out/$N.yaml $S3S/run.yaml
log "END $N exit=$rc rows=$n"
