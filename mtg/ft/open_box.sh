#!/bin/bash
# Open-weight MTG baseline box (4xL40S): /scratch, vLLM 0.31 venv, model + MTG test videos/transcripts, 4 vLLM servers (1/GPU),
# then open_gen.py over the test ids.   bash open_box.sh <commit> <model_dir_name> <served_tag>
WANT=$1; MD=$2; TAG=$3; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
exec > $HOME/open_$TAG.log 2>&1; echo "START $(date -u)"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dpbno NAME,SIZE,TYPE | awk '$3=="disk"' | sort -k2 -n -r | while read d sz t; do
          [ -z "$(lsblk -no MOUNTPOINT $d | tr -d '[:space:]')" ] && [ $(lsblk -no NAME $d | wc -l) = 1 ] && { echo $d; break; }; done)
  sudo mkfs.ext4 -q -F $DEV && sudo mkdir -p /scratch && sudo mount $DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
cd /scratch
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && { [ -d adlm_repo ] || git clone -q adlm_$WANT.bundle adlm_repo; } && git -C adlm_repo fetch -q /scratch/adlm_$WANT.bundle qwen38-dense-dataset-training && git -C adlm_repo checkout -q $WANT
(aws s3 cp --recursive --only-show-errors $B/models/$MD/ /scratch/models/$MD/ && echo MODEL_DONE) &
(aws s3 cp --recursive --only-show-errors $B/mtg/videos_480p/test/ /scratch/mtg/videos/ && aws s3 cp --only-show-errors $B/mtg/ft/open/mtg-ad.tgz . && tar xzf mtg-ad.tgz \
  && aws s3 cp --recursive --only-show-errors $B/mtg/ft/open/transcript/ /scratch/mtg/transcript/ && aws s3 cp --only-show-errors $B/mtg/ft/open/test_ids.json /scratch/mtg/ && echo DATA_DONE) &
export PATH=$HOME/.local/bin:$PATH; which uv || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/scratch/uvcache
[ -x /scratch/cbenv/bin/vllm ] || { uv venv -q -p 3.12 cbenv && . cbenv/bin/activate && uv pip install -q vllm==0.31.0 openai "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" \
  && C=/scratch/cbenv/lib/python3.12/site-packages/nvidia/cu13 && { [ -e $C/lib64 ] || ln -s lib $C/lib64; }; }
. /scratch/cbenv/bin/activate; wait; echo "SETUP_DONE $(date -u)"
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g nohup vllm serve /scratch/models/$MD --served-model-name $TAG --port $((8000+g)) --max-model-len 65536 \
  --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"video":1,"image":0}' --allowed-local-media-path /scratch/mtg/videos \
  --max-num-seqs 8 --trust-remote-code > /scratch/vllm_$g.log 2>&1 & done
for i in $(seq 120); do n=0; for g in 0 1 2 3; do curl -s localhost:$((8000+g))/v1/models >/dev/null && n=$((n+1)); done; [ $n = 4 ] && break; sleep 15; done; echo "SERVERS_UP $n $(date -u)"
MTG_REPO=/scratch/mtg-ad python /scratch/adlm_repo/mtg/ft/open_gen.py --ids /scratch/mtg/test_ids.json --videos /scratch/mtg/videos \
  --transcripts /scratch/mtg/transcript --out /scratch/mtg/out_$TAG --ports 8000,8001,8002,8003 --model $TAG --workers 32
aws s3 sync --only-show-errors /scratch/mtg/out_$TAG $B/mtg/ft/open/out_$TAG/ && echo "SYNCED $(ls /scratch/mtg/out_$TAG | wc -l) $(date -u)"
