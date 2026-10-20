set -x
export PATH=$HOME/.local/bin:$PATH
cd /scratch && uv venv -p 3.12 cbenv && . cbenv/bin/activate
export HF_HOME=/scratch/hf_cache UV_CACHE_DIR=/scratch/uvcache
(uv pip install -U "huggingface_hub[hf_xet]" && hf download Qwen/Qwen3.8-27B-FP8 > /scratch/dl.log 2>&1; echo DL_DONE >> /scratch/dl.log) &
uv pip install -U vllm openai omegaconf opencv-python-headless srt "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*"
C=/scratch/cbenv/lib/python3.12/site-packages/nvidia/cu13; [ -e $C/lib64 ] || ln -s lib $C/lib64
python -c "import vllm;print('VERS',vllm.__version__)"
wait
echo SETUP_DONE
