# E2 TF checkpoint sweep, one single-GPU box = one interleaved shard of the 591 eval clips.
#   bash e2_sweep_box.sh <commit> <shard i> <n_shards K> "<ckpt list, in order>"
# For each checkpoint (waits until it is complete in S3): BOX_SHARD=i/K e2_eval.sh <commit> <ckpt> tf 1.
# Writes $W/sweep_done when the list is finished. Setup = same packed venv + Qwen3.5-9B S3 mirror as e2_9b_train.sh.
WANT=$1; SI=$2; SK=$3; CKS=$4
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
[ -f /tmp/cr.env ] && mv /tmp/cr.env $W/creds.env; chmod 600 $W/creds.env; . $W/creds.env
B=s3://balu-transfer-933516006154-apne1/adlm; R=$W/adlm_repo
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel, fla, transformers; assert transformers.__version__ == '5.8.0'" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz"
  for f in uvpython venv; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi
$HOME/adlm/venv/bin/python -c "import torch, transformers, fla; print('ENV torch', torch.__version__, 'transformers', transformers.__version__)"
MD=$W/models/Qwen3.5-9B
[ -f $MD/.complete ] || aws s3 sync --only-show-errors $B/models/base/Qwen3.5-9B/ $MD/
cat $MD/.complete || { echo NO_BASE_MODEL; exit 1; }
for CK in $CKS; do
  C=$B/models/cmdad/qwen35_9b-dd_budget/checkpoint-$CK
  # complete = adapter, rows and trainer_state all present in S3 with their final sizes
  until [ "$(aws s3 ls $C/adapter_model.safetensors | awk '{print $3}')" = 346302176 ] && aws s3 ls $C/new_token_rows.safetensors >/dev/null && aws s3 ls $C/trainer_state.json >/dev/null; do
    echo "$(date -u +%T) waiting for checkpoint-$CK"; sleep 120; . $W/creds.env; done
  sleep 60   # let the sync of the remaining small files finish
  echo "$(date -u +%T) SWEEP_START ckpt$CK shard $SI/$SK" >> $W/sweep.log
  BOX_SHARD=$SI/$SK bash $R/ops/e2_eval.sh $WANT $CK tf 1 > $W/sweep_ckpt$CK.log 2>&1
  echo "$(date -u +%T) SWEEP_END ckpt$CK exit=$? rows=$(cat $W/inf9b/ckpt$CK/s${SI}_tf/teacher_forced_predictions.jsonl 2>/dev/null | wc -l)" >> $W/sweep.log
  . $W/creds.env; aws s3 cp --only-show-errors $W/sweep.log $B/inference/cmdad/eval591/_sweep_logs/sweep_s$SI.log
done
echo "$(date -u +%T) SWEEP_ALL_DONE" >> $W/sweep.log; touch $W/sweep_done
. $W/creds.env; aws s3 cp --only-show-errors $W/sweep.log $B/inference/cmdad/eval591/_sweep_logs/sweep_s$SI.log
