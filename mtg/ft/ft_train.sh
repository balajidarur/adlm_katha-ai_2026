#!/bin/bash
# T5 MTG fine-tune on a 4xL40S box.  bash ft_train.sh probe <commit>   |   bash ft_train.sh train <commit> [STOP_AT]
#  probe: 2 steps on the 8 longest train videos (bs1 x GA2 x 4 GPUs) with the E2/E3 memory-only options
#         OFFLOAD_GC (use_unsloth_gc) + NOPAD; logs peak GB per GPU.
#  train: mtg_ft440.yaml (cosine over 5 epochs), full checkpoints every 9 steps synced + size-verified to
#         s3 adlm/models/mtg/qwen38-dd_budget_ft440/checkpoint-N; STOP_AT=<step> stops after that checkpoint is in S3.
MODE=$1; WANT=$2; STOP=${3:-}; W=/scratch; R=$W/adlm_repo; LF=$R/training/LLaMA-Factory
B=s3://balu-transfer-933516006154-apne1; M=$B/adlm/models/mtg/qwen38-dd_budget_ft440; export AWS_DEFAULT_REGION=ap-northeast-1
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 LF_PAD_TO_MULTIPLE_OF=0
CK=$W/ckpt/qwen38-dd_budget_ckpt440
[ -s $CK/adapter_model.safetensors ] || { . $HOME/.cr.env; aws s3 cp --only-show-errors --recursive $B/adlm/models/cmdad/qwen38-dd_budget/checkpoint-440/ $CK/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"; }
CFG=examples/budget_ad/mtg_ft440.yaml; GC="use_unsloth_gc=${OFFLOAD_GC:-true}"
cd $LF
if [ "$MODE" = probe ]; then
  python - <<'PY'
import json
d = json.load(open("/scratch/mtg/ds/mtg_bins_f250_train.json"))
est = lambda r: 112 * len(r["images"]) + (len(r["messages"][0]["content"]) + len(r["messages"][1]["content"])) / 3.6
top = sorted(d, key=est, reverse=True)[:8]; print("probe est tokens", [int(est(r)) for r in top], "frames", [len(r["images"]) for r in top])
json.dump(top, open("/scratch/mtg/ds/mtg_probe8.json", "w"), ensure_ascii=False)
i = json.load(open("/scratch/mtg/ds/dataset_info.json")); i["mtg_probe8"] = dict(i["mtg_bins_f250_train"], file_name="mtg_probe8.json")
json.dump(i, open("/scratch/mtg/ds/dataset_info.json", "w"), indent=2)
PY
  OUT=$W/probe_runs/mtg_${GC}; rm -rf $OUT; mkdir -p $OUT
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 1; done ) > $OUT/smi.csv & SMI=$!
  llamafactory-cli train $CFG $GC dataset=mtg_probe8 tokenized_path=$W/tokcache/mtg_probe8 output_dir=$OUT \
    disable_shuffling=true max_steps=2 logging_steps=1 save_steps=1000 overwrite_output_dir=true > $OUT/train.log 2>&1
  RC=$?; kill $SMI
  echo "PROBE_RC $RC peak_MiB_per_gpu $(awk -F', ' '{if($2>m[$1])m[$1]=$2} END {for(g in m) printf "%s:%s ", g, m[g]}' $OUT/smi.csv)"
  grep -E "OutOfMemory|out of memory|'loss'" $OUT/train.log | tail -3; exit $RC
fi
OUT=$W/runs/mtg_ft440; mkdir -p $OUT; SYN=$W/synced; mkdir -p $SYN
complete(){ local c=$1; for f in trainer_state.json optimizer.pt scheduler.pt adapter_model.safetensors; do [ -s $c/$f ] || return 1; done
  [ "$(ls $c/rng_state*.pth 2>/dev/null | wc -l)" = 4 ]; }
( while true; do . $HOME/.cr.env
    for c in $OUT/checkpoint-*; do n=$(basename $c); [ -f $SYN/$n ] && continue; complete $c || continue; sleep 30
      aws s3 sync --only-show-errors $c $M/$n/ || continue
      L=$(find $c -type f -printf "%s\n" | awk '{s+=$1} END {print s}'); Rz=$(aws s3 ls --recursive $M/$n/ | awk '{s+=$3} END {print s}')
      if [ "$L" = "$Rz" ]; then touch $SYN/$n; echo "$(date +%T) SYNCED+VERIFIED $n $L" >> $W/ckpt_sync.log; else echo "$(date +%T) SIZE_MISMATCH $n" >> $W/ckpt_sync.log; fi
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null; aws s3 cp --only-show-errors $W/runs/train.log $M/train.log 2>/dev/null
    sleep 60; done ) &
[ -n "$STOP" ] && ( while [ ! -f $SYN/checkpoint-$STOP ]; do sleep 60; done; echo "$(date +%T) STOP_AT_$STOP" >> $W/ckpt_sync.log; pkill -f "mtg_ft440[.]yaml" ) &
echo "TRAIN_START $(date +%T)"
llamafactory-cli train $CFG $GC > $W/runs/train.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
