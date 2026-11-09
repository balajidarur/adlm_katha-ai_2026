# E2 eval box setup (g2 / box A, any GPU count): instance store at /scratch (or /opt/dlami/nvme), creds, commit-checked clone,
# packed venv (as e2_9b_train.sh), Qwen3.5-9B from the S3 mirror. No xtrace (creds are sourced).
#   bash e2_box_setup.sh <commit>      (expects ~/cr.env or /tmp/cr.env)
WANT=$1
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
if [ -d /opt/dlami/nvme ]; then W=/opt/dlami/nvme; sudo chown ubuntu:ubuntu $W
else W=/scratch
  if ! mountpoint -q $W; then
    DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
    sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p $W && sudo mount /dev/$DEV $W && sudo chown ubuntu:ubuntu $W
  fi
fi
echo "WORKDIR $W $(df -h $W | tail -1 | awk '{print $2, $4}')"
for f in /tmp/cr.env ~/cr.env; do [ -f $f ] && cp $f $W/creds.env && cp $f $W/cr.env; done; chmod 600 $W/creds.env $W/cr.env; . $W/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1/adlm; R=$W/adlm_repo
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle $W/adlm.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
echo "TRAINER_MD5 $(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel, fla, transformers; assert transformers.__version__ == '5.8.0'" 2>/dev/null; then
  mkdir -p $W/pack; aws s3 cp --only-show-errors --recursive $B/tf_rerun/ $W/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz"
  for f in uvpython venv; do pigz -dc $W/pack/$f.tgz | tar -x -C / ; done; rm -rf $W/pack
fi
$HOME/adlm/venv/bin/python -c "import torch, transformers, fla; print('ENV torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
MD=$W/models/Qwen3.5-9B
[ -f $MD/.complete ] || aws s3 sync --only-show-errors $B/models/base/Qwen3.5-9B/ $MD/
cat $MD/.complete || { echo NO_BASE_MODEL; exit 1; }
echo SETUP_DONE
