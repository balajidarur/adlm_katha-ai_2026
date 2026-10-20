export PATH=/home/ubuntu/adlm/venv/bin:$PATH
export HF_HOME=/home/ubuntu/adlm/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="$(cat $HOME/.wandb_key)"
export WANDB_DIR=/home/ubuntu/adlm/wandb
export WANDB_PROJECT=adlm-bins-frames
export FORCE_TORCHRUN=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /home/ubuntu/adlm/LLaMA-Factory
llamafactory-cli train examples/budget_ad/qwen38_bins_frames_full.yaml
echo "===== FULL RUN DONE ====="
