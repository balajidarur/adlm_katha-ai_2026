# usage: serveA.sh GPU  -> port 8000+GPU
. /scratch/envA.sh
export CUDA_VISIBLE_DEVICES=$1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec vllm serve Qwen/Qwen3.8-27B-FP8 --port $((8000+$1)) --max-model-len 49152 \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.93 --attention-backend TRITON_ATTN \
  --limit-mm-per-prompt '{"image": 110, "video": 0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-num-seqs 16 --max-num-batched-tokens 16384 --enable-prefix-caching
