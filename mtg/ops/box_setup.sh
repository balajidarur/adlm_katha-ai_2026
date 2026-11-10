#!/bin/bash
# T4 MTG box setup: instance-store /scratch, vLLM 0.31 venv, Qwen3.8-27B-FP8 weights from S3, MTG data.
exec > /scratch_setup.log 2>&1 || exec > $HOME/scratch_setup.log 2>&1
echo "START $(date -u)"
sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  # largest unmounted, partition-less nvme disk (instance store)
  DEV=$(lsblk -dpbno NAME,SIZE,TYPE | awk '$3=="disk"' | sort -k2 -n -r | while read d sz t; do
          [ -z "$(lsblk -no MOUNTPOINT $d | tr -d '[:space:]')" ] && [ $(lsblk -no NAME $d | wc -l) = 1 ] && { echo $d; break; }; done)
  echo "scratch dev $DEV"
  sudo mkfs.ext4 -q -F $DEV && sudo mkdir -p /scratch && sudo mount $DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
df -h /scratch
. $HOME/.cr.env
export AWS_DEFAULT_REGION=ap-northeast-1
B=s3://balu-transfer-933516006154-apne1
(aws s3 cp --recursive --only-show-errors $B/models/Qwen3.8-27B-FP8/ /scratch/models/Qwen3.8-27B-FP8/ && echo MODEL_DONE) &
(mkdir -p /scratch/mtg && aws s3 cp --recursive --only-show-errors $B/mtg/hf_mind-the-gap-inline-ad/ /scratch/mtg/hf/ && \
 { [ -n "$NO_VIDEOS" ] || aws s3 cp --recursive --only-show-errors $B/mtg/videos_480p/ /scratch/mtg/videos_480p/; } && echo DATA_DONE) &
export PATH=$HOME/.local/bin:$PATH
which uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
cd /scratch && uv venv -q -p 3.12 cbenv && . cbenv/bin/activate
export UV_CACHE_DIR=/scratch/uvcache
uv pip install -q vllm==0.31.0 "huggingface_hub[hf_xet]>=1.5,<2" openai omegaconf opencv-python-headless srt scenedetect \
  "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*"
C=/scratch/cbenv/lib/python3.12/site-packages/nvidia/cu13; [ -e $C/lib64 ] || ln -s lib $C/lib64
python -c "import vllm;print('VERS',vllm.__version__)"
wait
echo "SETUP_DONE $(date -u)"
