# runs on a new box: idle-shutdown off, local NVMe -> /scratch, venv + vLLM + model
set -x
sudo touch /etc/gpu-idle-shutdown.disable; sudo apt-get install -y -q curl >/dev/null 2>&1
if mountpoint -q /opt/dlami/nvme; then sudo ln -sfn /opt/dlami/nvme /scratch; sudo chown ubuntu:ubuntu /opt/dlami/nvme; else
DEV=$(ls /dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_* | grep -v -- -ns- | grep -v _1$ | head -1)
sudo mkfs.ext4 -q -F -L scratch $DEV && sudo mkdir -p /scratch && sudo mount $DEV /scratch && sudo chown ubuntu:ubuntu /scratch; fi
mkdir -p /scratch/prep_360p /scratch/asr ~/qwen_dense
cp /tmp/envA.sh /tmp/serveA.sh /tmp/setupA.sh /scratch/
sed -i "s#/data/cmd_ad/asr/SRT#/scratch/asr/SRT#" /scratch/envA.sh
echo MOUNTED
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
bash /scratch/setupA.sh > /scratch/setup.log 2>&1
nohup bash /scratch/serveA.sh 0 > /scratch/vllm_0.log 2>&1 &
for i in $(seq 1 90); do curl -s localhost:8000/v1/models >/dev/null && { echo SERVER_READY; exit 0; }; sleep 10; done
echo SERVER_FAILED
