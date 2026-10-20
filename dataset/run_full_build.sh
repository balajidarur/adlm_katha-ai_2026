#!/usr/bin/env bash
# Full build of datasets A / B / C — all usable clips, 1 FPS, cap 200, 60k token budget.
# Runs inside tmux session "cmdad_build":  tmux attach -t cmdad_build   (detach: Ctrl-b d)
set -o pipefail
cd /home/ubuntu/adlm/dataset_prep
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
mkdir -p logs
LOG=logs/build_full_$(date +%Y%m%d_%H%M%S).log
echo "log: $LOG"
../venv/bin/python -u build_frame_datasets.py --n-per-split 0 --workers 64 --ffmpeg-threads 1 2>&1 | tee "$LOG"
echo "BUILD_EXIT=$? log=$LOG" | tee -a "$LOG"
