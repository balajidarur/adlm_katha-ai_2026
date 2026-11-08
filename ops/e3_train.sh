# E3 / M8: M3 recipe with the Qwen3-VL-8B writer on Qwen3-VL-8B dense descriptions (adapted from e2_9b_train.sh).
#   bash e3_train.sh <mode: probe|train> <commit> <n_gpus> <per_device_bs> <grad_accum> <ds: none|z2|z3> [probe_dataset|resume_step] [max_steps]
# Config = examples/budget_ad/q3vl8b_bins_frames_full.yaml = M3's qwen38_bins_frames_full.yaml with: model Qwen3-VL-8B-Instruct
# (local copy of s3 models/Qwen3-VL-8B-Instruct), template qwen3_vl_nothink, additional_target embed_tokens,lm_head (untied
# lm_head; restrict_new_token_grad masks both to the 9 new rows). Optional ZeRO-2/3. Effective batch 32 asserted.
# probe: s3 adlm/datasets/cmdad_q3vl8b_probe (longest 10% by Qwen3-VL tokens, sorted desc, + random 96), sequential sampler.
# train: s3 cmd_ad_dense/datasets/cmdad_q3vl8b_dd (SHA256 verified); checkpoints every 20 steps -> s3
#        adlm/models/cmdad/qwen3vl8b-dd_budget/. STOP_AT=<step> stops after that checkpoint is in S3.
MODE=$1; WANT=$2; NG=$3; BS=$4; GA=$5; DSC=$6; ARG7=${7:-}; MAXS=${8:-3}
set -x
[ $((NG * BS * GA)) = 32 ] || { echo "EFFECTIVE_BATCH $((NG * BS * GA)) != 32"; exit 1; }
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
if [ -d /opt/dlami/nvme ]; then W=/opt/dlami/nvme/e3; sudo chown ubuntu:ubuntu /opt/dlami/nvme; mkdir -p $W
else W=/scratch
  if ! mountpoint -q $W; then
    DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
    sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p $W && sudo mount /dev/$DEV $W && sudo chown ubuntu:ubuntu $W
  fi
fi
echo "WORKDIR $W"
[ -f /tmp/cr.env ] && mv /tmp/cr.env $W/cr.env; ln -sfn $W/cr.env $W/creds.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1/adlm
R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; M=$B/models/cmdad/qwen3vl8b-dd_budget; CFG=examples/budget_ad/q3vl8b_bins_frames_full.yaml
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }

# environment: packed venv from tf_rerun (same as M3 eval boxes) + deepspeed if a ZeRO stage is requested
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz"
  for f in uvpython venv; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi
if [ "$DSC" != none ] && ! $HOME/adlm/venv/bin/python -c "import deepspeed" 2>/dev/null; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_CACHE_DIR=$W/uvcache uv pip install -q --python $HOME/adlm/venv/bin/python "deepspeed==0.16.9" || { echo DS_INSTALL_FAIL; exit 1; }
fi
$HOME/adlm/venv/bin/python -c "import torch, transformers, liger_kernel; print('ENV torch', torch.__version__, 'transformers', transformers.__version__)"

# base model: Qwen/Qwen3-VL-8B-Instruct, the copy at s3 models/Qwen3-VL-8B-Instruct (same weights the DD run served)
MD=$W/models/Qwen3-VL-8B-Instruct
[ -f $MD/model-00004-of-00004.safetensors ] || aws s3 sync --only-show-errors s3://balu-transfer-933516006154-apne1/models/Qwen3-VL-8B-Instruct/ $MD/
md5sum $MD/config.json; du -sh $MD

# dataset
if [ "$MODE" = probe ]; then
  D=$W/probe; DSN=${ARG7:-probe_top10}; mkdir -p $D && cd $D
  aws s3 sync --only-show-errors $B/datasets/cmdad_q3vl8b_probe/ .
  sha256sum -c SHA256SUMS || { echo SHA_MISMATCH; exit 1; }
else
  D=$W/ds; DSN=q3vl8b_f150_40k; mkdir -p $D && cd $D
  if ! sha256sum -c --quiet SHA256SUMS 2>/dev/null; then
    aws s3 sync --only-show-errors s3://balu-transfer-933516006154-apne1/cmd_ad_dense/datasets/cmdad_q3vl8b_dd/ . --exclude "*" --include "SHA256SUMS" --include "dataset_info.json" --include "q3vl8b_f150_40k.json"
    grep -E "q3vl8b_f150_40k.json|dataset_info" SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
  fi
fi
cd $HOME
# frames: only the clips this dataset needs, from the S3 copy of box A's 1fps_448x252 pool
python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('$D/$(python3 -c "import json;print(json.load(open('$D/dataset_info.json'))['$DSN']['file_name'])")')) for p in r['images']})))" > $W/vids_$DSN.txt
mkdir -p $W/frames/1fps_448x252
cat $W/vids_$DSN.txt | xargs -P 48 -I{} sh -c "[ -d $W/frames/1fps_448x252/{} ] || aws s3 sync --only-show-errors $B/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/"
sudo mkdir -p /data/cmd_ad/frames; [ "$(readlink /data/cmd_ad/frames/1fps_448x252)" = $W/frames/1fps_448x252 ] || { sudo rm -f /data/cmd_ad/frames/1fps_448x252; sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252; }
$HOME/adlm/venv/bin/python - <<EOF
import json, os
f = json.load(open("$D/dataset_info.json"))["$DSN"]["file_name"]
d = json.load(open("$D/" + f))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"DATASET $DSN {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing {miss}")
assert miss == 0
EOF
[ $? = 0 ] || { echo DATASET_CHECK_FAILED; exit 1; }

export PYTHONPATH=$LF/src HF_HOME=$W/hf_home HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$W/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=$W/wandb WANDB_PROJECT=adlm-e3
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p $W/wandb $W/tokcache $W/runs
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo "GPUS $(nvidia-smi -L | wc -l) (want $NG) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) | effective batch = $BS x $GA x $NG = $((BS * GA * NG))"
[ "$(nvidia-smi -L | wc -l)" = "$NG" ] || { echo GPU_COUNT_MISMATCH; exit 1; }

DSARG=""; [ "$DSC" != none ] && DSARG="deepspeed=examples/deepspeed/ds_${DSC}_config.json"
# ROWS_ONLY=1: train only the 9 budget rows of embed_tokens/lm_head (new_token_rows_only delta, d14e944 + ae44d93) instead
# of full modules_to_save copies; mathematically the same update, ~1.24B fewer trainable params
if [ "${ROWS_ONLY:-0}" = 1 ]; then TOK="additional_target=null new_token_rows_only=true"; else TOK=""; fi
COMMON="model_name_or_path=$MD $TOK dataset_dir=$D dataset=$DSN \
  per_device_train_batch_size=$BS gradient_accumulation_steps=$GA preprocessing_num_workers=$(( $(nproc) > 48 ? 48 : $(nproc) )) $DSARG"

cd $LF
if [ "$MODE" = probe ]; then
  TAG=${DSN}_${DSC}_bs${BS}_ga${GA}_n${NG}_rows${ROWS_ONLY:-0}; OUT=$W/probe_runs/$TAG; rm -rf $OUT; mkdir -p $OUT
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 1; done ) > $OUT/smi.csv &
  SMI=$!
  echo "PROBE_START $TAG $(date +%T)"
  llamafactory-cli train $CFG $COMMON \
    tokenized_path=$W/tokcache/$DSN output_dir=$OUT run_name=e3-probe-$TAG report_to=none \
    disable_shuffling=true max_steps=$MAXS logging_steps=1 save_steps=$MAXS save_only_model=true overwrite_output_dir=true \
    ${EXTRA_ARGS:-} > $OUT/train.log 2>&1
  RC=$?; kill $SMI
  # did the 9 new rows of embed_tokens AND lm_head move, and nothing else? (lm_head is untied on Qwen3-VL-8B; liger's fused CE
  # must read the trainable copy, not the frozen base weight)
  python - $OUT $MD <<'EOF' 2>&1 | tee $OUT/rowcheck.txt
import sys, glob, json, torch
from safetensors import safe_open
out, md = sys.argv[1], sys.argv[2]
rf = sorted(glob.glob(out + "/checkpoint-*/new_token_rows.safetensors"))
if rf:
    with safe_open(rf[-1], "pt") as f:
        for k in f.keys():
            if k.endswith(".delta"):
                d = f.get_tensor(k).float(); print(f"ROWCHECK rows-only {k}: shape {tuple(d.shape)}, rows moved {(d.abs().sum(1) > 0).sum().item()}/{d.shape[0]}, |delta| mean {d.abs().mean():.3e}")
    sys.exit()
ad = sorted(glob.glob(out + "/checkpoint-*/adapter_model.safetensors"))
if not ad: print("ROWCHECK no adapter saved"); sys.exit()
tok = json.load(open(glob.glob(out + "/checkpoint-*/tokenizer.json")[0]))
ids = sorted(t["id"] for t in tok["added_tokens"] if t["content"].startswith("<|budget_"))
idx = json.load(open(md + "/model.safetensors.index.json"))["weight_map"]
def base(name):
    with safe_open(md + "/" + idx[name], "pt") as f: return f.get_tensor(name)
with safe_open(ad[-1], "pt") as f:
    for k in f.keys():
        if "embed_tokens" not in k and "lm_head" not in k: continue
        a = f.get_tensor(k).float()
        bname = "lm_head.weight" if "lm_head" in k else [n for n in idx if n.endswith("embed_tokens.weight") and "visual" not in n][0]
        b = base(bname).float()
        d = (a - b).abs().sum(1)
        moved = (d > 0).nonzero().flatten().tolist()
        print(f"ROWCHECK {k}: rows changed vs base = {len(moved)}; budget ids {ids[:2]}..{ids[-1]} changed: "
              f"{sum(i in moved for i in ids)}/{len(ids)}; other rows changed: {len([i for i in moved if i not in ids])}")
EOF
  rm -rf $OUT/checkpoint-*/*.safetensors
  python3 - $OUT $RC <<'EOF'
import sys, json, collections, re
out, rc = sys.argv[1], sys.argv[2]
peak = collections.defaultdict(int)
for l in open(out + "/smi.csv"):
    try: i, m = [int(x) for x in l.split(",")]; peak[i] = max(peak[i], m)
    except ValueError: pass
log = open(out + "/train.log", errors="ignore").read()
oom = "OutOfMemoryError" in log or "CUDA out of memory" in log
steps = []
try:
    for l in open(out + "/trainer_log.jsonl"):
        r = json.loads(l); h, m, s = r["elapsed_time"].split(":"); steps.append((r["current_steps"], int(h) * 3600 + int(m) * 60 + int(s), r.get("loss")))
except FileNotFoundError: pass
sps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(steps, steps[1:])]
res = dict(tag=out.split("/")[-1], rc=int(rc), oom=oom, peak_mib={k: peak[k] for k in sorted(peak)}, steps=steps,
           s_per_step_after_first=(sum(sps) / len(sps) if sps else None), first_step_s=(steps[0][1] if steps else None))
print("PROBE_RESULT " + json.dumps(res))
open(out + "/result.json", "w").write(json.dumps(res, indent=1))
EOF
  . $W/creds.env; aws s3 sync --only-show-errors $OUT $M/_probe/$TAG/ --exclude "*.safetensors" --exclude "*.pt"
  exit 0
fi

# train
OUT=$W/runs/qwen3vl8b-dd_budget
RESUME=""
if [ -n "$ARG7" ]; then
  aws s3 cp --only-show-errors --recursive $M/checkpoint-$ARG7/ $OUT/checkpoint-$ARG7/
  echo "RESUME_FROM checkpoint-$ARG7 ($(ls $OUT/checkpoint-$ARG7 | wc -l) files)"
  for n in $(seq 20 20 $ARG7); do touch $W/synced_checkpoint-$n; done
  RESUME="resume_from_checkpoint=$OUT/checkpoint-$ARG7"
fi
( while true; do . $W/creds.env
    for c in $OUT/checkpoint-*; do [ -f $c/trainer_state.json ] || continue; n=$(basename $c)
      [ -f $W/synced_$n ] && continue
      sleep 30; aws s3 sync --only-show-errors $c $M/$n/ && touch $W/synced_$n && echo "$(date +%T) SYNCED $n $(du -sb $c | cut -f1)" >> $W/ckpt_sync.log
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    aws s3 cp --only-show-errors $W/runs/train.log $M/train.log 2>/dev/null
    sleep 60; done ) &
[ -n "${STOP_AT:-}" ] && ( while [ ! -f $W/synced_checkpoint-$STOP_AT ]; do sleep 60; done
  echo "$(date +%T) STOP_AT_$STOP_AT" >> $W/ckpt_sync.log; pkill -f "q3vl8b_bins_frames_full[.]yaml" ) &
echo "TRAIN_START $(date +%T)"
llamafactory-cli train $CFG $COMMON \
  tokenized_path=$W/tokcache/q3vl8b output_dir=$OUT run_name=qwen3vl8b-dd_budget save_total_limit=40 $RESUME ${EXTRA_ARGS:-} \
  > $W/runs/train.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
. $W/creds.env; aws s3 cp --only-show-errors $W/runs/train.log $M/train.log
