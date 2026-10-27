# box A: instance NVMe -> /scratch, pull REFRAMED train data (same paths as box C), train the no-budget model
set -x
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme3n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mv /tmp/crA.env /scratch/creds.env; chmod 600 /scratch/creds.env; . /scratch/creds.env
aws configure set default.s3.max_concurrent_requests 64
aws s3 cp --only-show-errors s3://balu-transfer-933516006154-apne1/reframed/train_data/rf_frames_datasets.tar.gz - | pigz -dc | tar -x -C /scratch
aws s3 cp --only-show-errors s3://balu-transfer-933516006154-apne1/reframed/train_data/reframed_qwen38_nobudget_frames.yaml ~/adlm/LLaMA-Factory/examples/budget_ad/
ls /scratch/rf_frames/1fps_448x252 | wc -l; ls /scratch/rf_datasets/frame_variants_448x252/reframed_nobudget_f150_40k_train.json
cd ~/adlm/LLaMA-Factory && export PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_API_KEY="$(cat ~/.wandb_key)" WANDB_DIR=/scratch/wandb WANDB_PROJECT=adlm-bins-frames FORCE_TORCHRUN=1
mkdir -p /scratch/wandb /scratch/runs
echo "TRAIN_START $(date +%T)"
llamafactory-cli train examples/budget_ad/reframed_qwen38_nobudget_frames.yaml > /scratch/runs/train_reframed_nobudget.log 2>&1
echo "TRAIN_EXIT $? $(date +%T)"
