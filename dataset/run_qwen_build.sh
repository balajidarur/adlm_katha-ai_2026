#!/usr/bin/env bash
# Same build as the Gemini full run (cap 200, token budget 48k, 1 FPS 448x252, same frame pool),
# dense descriptions from the Qwen3.8-27B charbank run instead of Gemini 3.5 Flash Lite.
set -o pipefail
cd /home/ubuntu/adlm/dataset_prep
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export DENSE_PATH_TMPL=/prep/qwen_dense_final/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json
OUT=/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38
LOG=logs/build_qwen38_$(date +%Y%m%d_%H%M%S).log
../venv/bin/python -u build_frame_datasets.py --n-per-split 0 --workers 44 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $OUT 2>&1 | tee "$LOG"
echo "BUILD_EXIT=$? log=$LOG" | tee -a "$LOG"
