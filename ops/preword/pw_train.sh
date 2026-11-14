# M3 retrained on PRE-WORD budget labels (token before word k = bin(gap - prefix_duration_s[k-1])).
# Recipe = M3 exactly: examples/budget_ad/qwen38_bins_frames_full.yaml unchanged (Qwen/Qwen3.5-4B, LoRA r32/a64/drop 0.05,
# lora_target all + embed_tokens, frozen vision tower, 9 budget tokens desc_init, restrict_new_token_grad, template
# qwen3_5_nothink, cutoff 49152, LR 1e-4 cosine, warmup 0.03, 3 epochs = 729 steps, 1 x 8 accum x 4 GPUs = batch 32,
# save_steps 20, seed 42); only dataset_dir/dataset/tokenized_path/output_dir/run_name differ.
# Data: s3 adlm/datasets/cmdad_qwen38_preword (SHA256SUMS verified) built by dataset/make_preword_bins.py from M3's
# bins_f150_40k (sha256 0536f0ef...). Stops once checkpoint-440 (= M3's paper checkpoint, epoch 1.81) is synced to S3.
# Target box: Seoul s4 (i-0b9048f53a67519e2, ap-northeast-2, g6e.12xlarge 4x L40S); work dir /opt/dlami/nvme (instance store).
#
# Launch (from garuda, after `aws sso login`; NOT started automatically):
#   aws ec2 start-instances --region ap-northeast-2 --instance-ids i-0b9048f53a67519e2
#   scp creds -> ubuntu@<ip>:/tmp/cr.env ; scp ops/preword/pw_train.sh -> ubuntu@<ip>:~/
#   ssh ubuntu@<ip> "nohup bash ~/pw_train.sh <commit> > ~/pw_train.log 2>&1 &"
# Env: STOP_AT=<step> (default 440; STOP_AT= runs all 3 epochs).
WANT=$1; NG=4; GA=8
[ -n "$WANT" ] || { echo "usage: pw_train.sh <commit>"; exit 1; }
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
echo "WORKDIR $W"
mv /tmp/cr.env $W/creds.env 2>/dev/null; chmod 600 $W/creds.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1/adlm; DS=$B/datasets/cmdad_qwen38_preword; N=qwen38_bins_preword
R=$W/adlm_repo; LF=$R/training/LLaMA-Factory; D=$W/pw_ds; OUT=$W/runs/qwen38-dd_budget_preword
M=$B/models/cmdad/qwen38-dd_budget_preword
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
# the recipe file must be M3's, byte for byte
echo "YAML_MD5 $(md5sum $LF/examples/budget_ad/qwen38_bins_frames_full.yaml)"

# environment: packed venv + Qwen3.5-4B from tf_rerun unless this box already has a working one
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi

# dataset: locked JSONs, checksums verified
mkdir -p $D && cd $D
aws s3 cp --only-show-errors $DS/SHA256SUMS . && aws s3 cp --only-show-errors $DS/dataset_info.json .
for s in f150_40k eval527; do aws s3 cp --only-show-errors $DS/${N}_$s.json .; done
grep -E "${N}_(f150_40k|eval527)|dataset_info" SHA256SUMS | sha256sum -c - || { echo SHA_MISMATCH; exit 1; }
cd $HOME

# frames: S3 copy of the 1 FPS pool (box A's /data is not here); symlinked to the path the JSONs use
if [ ! -d /data/cmd_ad/frames/1fps_448x252 ]; then
  aws s3 ls $B/frames/_DONE >/dev/null || { echo NO_FRAMES_DONE_MARKER; exit 1; }
  python3 -c "import json;print('\n'.join(sorted({p.split('/')[-2] for f in ('$D/${N}_f150_40k.json','$D/${N}_eval527.json') for r in json.load(open(f)) for p in r['images']})))" > $W/pw_vids.txt
  mkdir -p $W/frames/1fps_448x252
  cat $W/pw_vids.txt | xargs -P 48 -I{} sh -c ". $W/creds.env; aws s3 sync --only-show-errors $B/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/"
  sudo mkdir -p /data/cmd_ad/frames && sudo ln -sfn $W/frames/1fps_448x252 /data/cmd_ad/frames/1fps_448x252
fi
$HOME/adlm/venv/bin/python - <<EOF || { echo DATASET_CHECK_FAILED; exit 1; }
import json, os
d = json.load(open("$D/${N}_f150_40k.json"))
miss = sum(not os.path.exists(p) for r in d for p in r["images"])
print(f"DATASET {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing {miss}")
assert len(d) == 7763 and miss == 0
EOF

export PYTHONPATH=$LF/src HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=$W/hf_datasets \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1 WANDB_DIR=$W/wandb WANDB_PROJECT=adlm-preword
[ -f ~/.wandb_key ] && export WANDB_API_KEY="$(cat ~/.wandb_key)" || export WANDB_MODE=offline
export PATH=$HOME/adlm/venv/bin:$PATH
mkdir -p $W/wandb $W/tokcache $W/runs
echo "LF_IMPORT $(python -c 'import llamafactory, os; print(os.path.dirname(llamafactory.__file__))')"
echo "TRAINER_MD5 $(md5sum $LF/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo "GPUS $(nvidia-smi -L | wc -l) (want $NG) | effective batch = 1 x $GA x $NG = $((GA * NG))"
[ "$(nvidia-smi -L | wc -l)" = "$NG" ] || { echo GPU_COUNT_MISMATCH; exit 1; }

# every checkpoint to S3 as it lands (size-verified); stop after STOP_AT is safely there
complete(){ local c=$1; for f in trainer_state.json optimizer.pt scheduler.pt adapter_model.safetensors; do [ -s $c/$f ] || return 1; done
  [ "$(ls $c/rng_state*.pth 2>/dev/null | wc -l)" = "$NG" ] || return 1; return 0; }
( while true; do . $W/creds.env
    for c in $OUT/checkpoint-*; do n=$(basename $c); [ -f $W/synced_$n ] && continue
      complete $c || continue; sleep 30
      aws s3 sync --only-show-errors $c $M/$n/ || continue
      L=$(find $c -type f -printf "%s\n" | awk '{s+=$1} END {print s}')
      Rz=$(aws s3 ls --recursive $M/$n/ | awk '{s+=$3} END {print s}')
      if [ "$L" = "$Rz" ]; then touch $W/synced_$n; echo "$(date +%T) SYNCED+VERIFIED $n $L bytes" >> $W/ckpt_sync.log
      else echo "$(date +%T) SIZE_MISMATCH $n local $L s3 $Rz (retry)" >> $W/ckpt_sync.log; fi
    done
    aws s3 cp --only-show-errors $OUT/trainer_log.jsonl $M/trainer_log.jsonl 2>/dev/null
    aws s3 cp --only-show-errors $W/runs/train_preword.log $M/train.log 2>/dev/null
    sleep 60; done ) &
STOP=${STOP_AT-440}
[ "$STOP" ] && ( while [ ! -f $W/synced_checkpoint-$STOP ]; do sleep 60; done
  echo "$(date +%T) STOP_AT_$STOP" >> $W/ckpt_sync.log; pkill -f "qwen38_bins_frames_full.yaml" ) &

cd $LF && echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/qwen38_bins_frames_full.yaml \
  dataset_dir=$D dataset=${N}_f150_40k tokenized_path=$W/tokcache/$N output_dir=$OUT run_name=qwen38-dd_budget_preword \
  gradient_accumulation_steps=$GA preprocessing_num_workers=$(( $(nproc) > 48 ? 48 : $(nproc) )) \
  > $W/runs/train_preword.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
. $W/creds.env; aws s3 cp --only-show-errors $W/runs/train_preword.log $M/train.log
