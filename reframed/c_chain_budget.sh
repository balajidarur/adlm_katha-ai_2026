# box C: wait for inputs -> build frame datasets -> add budget bins + cap -> train (Qwen3.5-4B, 3 epochs)
until grep -q C_SETUP_DONE /tmp/c_setup_budget.log; do sleep 30; done
cd /scratch2/rfcode; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
P=~/adlm/venv/bin/python
echo "$(date +%T) BUILD_START"
$P -u rf_build.py --out /scratch/rf_datasets/frame_variants_448x252 --frame-root /scratch/rf_frames/1fps_448x252 \
  --workers 44 --ffmpeg-threads 1 --cap 200 --token-budget 48000 2>&1 | grep --line-buffered -v "it/s\|clip/s" > /scratch2/rf_build.log
echo "$(date +%T) BUILD_DONE $(tail -12 /scratch2/rf_build.log | head -3 | tr '\n' ' ')"
$P -u rf_bins.py --dir /scratch/rf_datasets/frame_variants_448x252 > /scratch2/rf_bins.log 2>&1
echo "$(date +%T) BINS_DONE"; cat /scratch2/rf_bins.log
[ -f /scratch2/NO_TRAIN ] && { echo "TRAIN_SKIPPED (NO_TRAIN flag)"; exit 0; }
cp /scratch2/rfcode/reframed_qwen38_bins_frames.yaml ~/adlm/LLaMA-Factory/examples/budget_ad/
cd ~/adlm/LLaMA-Factory && export PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_API_KEY="$(cat ~/.wandb_key)" WANDB_DIR=/scratch/wandb WANDB_PROJECT=adlm-bins-frames FORCE_TORCHRUN=1
mkdir -p /scratch/wandb /scratch/runs
echo "$(date +%T) TRAIN_START"
llamafactory-cli train examples/budget_ad/reframed_qwen38_bins_frames.yaml > /scratch/runs/train_reframed.log 2>&1
echo "$(date +%T) TRAIN_EXIT $?"
