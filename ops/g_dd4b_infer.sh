# Any fresh 4xL40S box: eval a Qwen3.5-4B-DD budget checkpoint (M3 recipe on 4B dense descriptions) on dd4b_eval527 -
# teacher-forced, then free-running (max_new_tokens 2048, as M3's free_running_max2048). Commit-checked clone, packed env
# from s3 adlm/tf_rerun, locked eval JSON (SHA256SUMS), frames of the 527 eval clips from s3 adlm/frames.
#   bash g_dd4b_infer.sh <commit> <step>
# Env overrides for other runs (e.g. the A2/A3/A4 input ablations): DSDIR (s3 datasets dir), EVAL (eval dataset key),
# MODEL_DIR (s3 models/cmdad dir), e.g. DSDIR=cmdad_qwen38_ablations EVAL=a3_nodds_eval527 MODEL_DIR=qwen38-dd_a3_nodds.
WANT=$1; STEP=$2; EVAL=${EVAL:-dd4b_eval527}; MODEL_DIR=${MODEL_DIR:-qwen35_4b-dd_bins}
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
if [ -d /opt/dlami/nvme ]; then W=/opt/dlami/nvme; sudo chown ubuntu:ubuntu $W
else W=/scratch
  if ! mountpoint -q $W; then
    DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
    sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p $W && sudo mount /dev/$DEV $W && sudo chown ubuntu:ubuntu $W
  fi
fi
mv /tmp/cr.env $W/creds.env 2>/dev/null; chmod 600 $W/creds.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1/adlm; DS=$B/datasets/${DSDIR:-cmdad_qwen35_4b_dd}; N=${MODEL_DIR}_ckpt$STEP
R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; D=$W/evalds; CK=$W/ckpt/$N; O=$W/inf; P=$B/inference/cmdad/eval527/$N
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi
mkdir -p $D $O && cd $D
for f in SHA256SUMS dataset_info.json $EVAL.json; do aws s3 cp --only-show-errors $DS/$f .; done
grep -E "$EVAL.json|dataset_info" SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
if [ ! -d /data/cmd_ad/frames/1fps_448x252 ]; then
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for r in json.load(open('$EVAL.json')) for p in r['images']})))" > vids.txt
  cat vids.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/
  sudo mkdir -p /data/cmd_ad/frames && sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252
fi
python3 -c "
import json, os; d = json.load(open('$EVAL.json')); miss = sum(not os.path.exists(p) for r in d for p in r['images'])
print(f'EVAL {len(d)} clips | {sum(len(r[\"images\"]) for r in d):,} frames | missing {miss}'); assert miss == 0" || { echo FRAMES_MISSING; exit 1; }
aws s3 cp --only-show-errors --recursive $B/models/cmdad/$MODEL_DIR/checkpoint-$STEP/ $CK/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
ls $CK/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
export PYTHONPATH=$LF/src PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 \
  HF_DATASETS_CACHE=$W/hf_datasets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
MD5=$(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
cd $LF
mk(){ { echo "model_name_or_path: Qwen/Qwen3.5-4B"; echo "adapter_name_or_path: $CK"; echo "trust_remote_code: true"
    echo "image_max_pixels: 589824"; echo "image_min_pixels: 1024"; echo "enable_liger_kernel: true"
    echo "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml"; echo "init_special_tokens: desc_init"
    echo "resize_vocab: false"; echo "stage: sft"; echo "do_predict: true"; echo "finetuning_type: lora"; echo "budget_variant: none"
    echo "dataset_dir: $D"; echo "eval_dataset: $EVAL"; echo "template: qwen3_5_nothink"; echo "cutoff_len: 49152"
    echo "overwrite_cache: true"; echo "output_dir: $O/$1"; echo "overwrite_output_dir: true"; echo "report_to: none"
    echo "per_device_eval_batch_size: 1"; echo "bf16: true"
    case $2 in tf) echo "budget_teacher_forced: true"; echo "budget_prefill_chunk: 2048";;
      free:*) echo "predict_with_generate: true"; echo "max_new_tokens: ${2#free:}"; echo "do_sample: false"; echo "skip_special_tokens: false";; esac
  } > examples/budget_ad/q_$1.yaml; }
run(){ echo "$(date +%T) START $1" >> $O/queue.log
  llamafactory-cli train examples/budget_ad/q_$1.yaml > $O/$1.log 2>&1; rc=$?
  echo "$(date +%T) END $1 exit=$rc rows=$(cat $O/$1/*_predictions.jsonl 2>/dev/null | wc -l)" >> $O/queue.log
  . $W/creds.env; aws s3 sync --only-show-errors $O/$1 $P/$2/ --exclude "*" --include "*.jsonl" --include "*.json"
  aws s3 cp --only-show-errors $O/$1.log $P/$2/run.log && echo "$(date +%T) SYNCED $1 -> $P/$2" >> $O/queue.log; }
mk ${N}_tf tf; mk ${N}_free free:2048
run ${N}_tf teacher_forced
run ${N}_free free_running_max2048
echo ALL_DONE >> $O/queue.log
