# usage: box_setup.sh NGPU — prepare any L40S box for the REFRAMED Qwen3.8 DD run (bundle at /tmp/rf_bundle.tgz)
# instance NVMe only (never an EBS data volume): first -> /scratch, second (if any) -> /scratch2, else /scratch2 -> /scratch
NGPU=${1:-1}
set -x
sudo touch /etc/gpu-idle-shutdown.disable
command -v pigz > /dev/null || sudo apt-get install -y -q pigz > /dev/null 2>&1
if ! mountpoint -q /scratch; then
  if mountpoint -q /opt/dlami/nvme; then sudo ln -sfn /opt/dlami/nvme /scratch; sudo chown ubuntu:ubuntu /opt/dlami/nvme
  else
    DEVS=$(ls /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_* | grep -v -- '_1$' | grep -v -- -ns-)
    i=0; for d in $DEVS; do mp=/scratch; [ $i -eq 1 ] && mp=/scratch2; [ $i -ge 2 ] && break
      sudo mkfs.ext4 -q -F $d && sudo mkdir -p $mp && sudo mount $d $mp && sudo chown ubuntu:ubuntu $mp; i=$((i+1)); done
  fi
fi
[ -e /scratch2 ] || sudo ln -s /scratch /scratch2
df -h /scratch /scratch2 | tail -2
tar xzf /tmp/rf_bundle.tgz -C /scratch2 && rm -rf ~/rf_code && mv /scratch2/rf_code ~/rf_code
mkdir -p /scratch2/rf_srt && tar xzf /scratch2/rf_srt.tgz -C /scratch2/rf_srt && mv /scratch2/rf_srt/srt/dialogue /scratch2/rf_srt/dialogue
chmod 600 /scratch2/awscreds.env
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
command -v aws > /dev/null || uv tool install -q awscli > /dev/null 2>&1
bash /scratch2/setupQ.sh > /scratch2/qsetup.log 2>&1 &
SETUP=$!
# prep data from the S3 backup (bucket is in Tokyo)
. /scratch2/awscreds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
mkdir -p /scratch/rf_prep_360p /scratch2/shards
aws s3 cp --recursive --only-show-errors s3://balu-transfer-933516006154-apne1/reframed/prep_360p/ /scratch2/shards/
(cd /scratch2/shards && md5sum -c md5.txt | grep -v ': OK$')
for f in /scratch2/shards/*.tar.gz; do (pigz -dc $f | tar -x -C /scratch/rf_prep_360p && rm $f) & done; wait $(jobs -p | grep -v "^$SETUP$")
echo "PREP_READY $(ls /scratch/rf_prep_360p/*/*/meta.json | wc -l)"
wait $SETUP
grep -q SETUP_DONE /scratch2/qsetup.log || { echo SETUP_FAILED; exit 1; }
for g in $(seq 0 $((NGPU-1))); do nohup bash /scratch2/serveQ.sh $g > /scratch2/vllm_$g.log 2>&1 < /dev/null & done
for i in $(seq 1 120); do n=0; for g in $(seq 0 $((NGPU-1))); do curl -s localhost:$((8000+g))/v1/models > /dev/null && n=$((n+1)); done
  [ $n -eq $NGPU ] && { echo "SERVERS_READY $n"; exit 0; }; sleep 10; done
echo SERVERS_FAILED
