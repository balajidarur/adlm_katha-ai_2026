# MADX (zero-shot MAD-Eval for E2 / M6 / E3) box setup, any 1xL40S g6e.2xlarge: instance store at /scratch,
# uv + awscli + pigz, commit-checked clone at /scratch/adlm_repo. Creds live only in ~/cr.env (pushed from garuda,
# refreshed every ~15 min, removed before stop). No xtrace anywhere (creds are sourced).
#   bash setup.sh <commit>     (run from a copy outside the repo, e.g. ~/madx_setup.sh)
WANT=$1
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
if [ -d /opt/dlami/nvme ]; then sudo chown ubuntu:ubuntu /opt/dlami/nvme; [ -e /scratch ] || sudo ln -sfn /opt/dlami/nvme /scratch
elif ! mountpoint -q /scratch; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p /scratch && sudo mount /dev/$DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
echo "SCRATCH $(readlink -f /scratch) $(df -h /scratch/ | tail -1 | awk '{print $2, $4}')"
command -v pigz > /dev/null || sudo apt-get install -y -q pigz > /dev/null 2>&1
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
command -v aws > /dev/null || ~/.local/bin/uv tool install -q awscli > /dev/null 2>&1
set -a; . ~/cr.env; set +a
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1; R=/scratch/adlm_repo
if [ "$(git -C $R rev-parse --short HEAD 2>/dev/null)" != "$WANT" ]; then
  aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle /scratch/adlm_$WANT.bundle
  rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm_$WANT.bundle $R
fi
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
echo "TRAINER_MD5 $(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
echo SETUP_DONE
