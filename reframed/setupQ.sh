set -x
export PATH=$HOME/.local/bin:$PATH
cd /scratch2 && uv venv -p 3.12 cbenv && . cbenv/bin/activate
export HF_HOME=/scratch2/hf_cache UV_CACHE_DIR=/scratch2/uvcache
(uv pip install "huggingface_hub[hf_xet]>=1.5,<2" && hf download Qwen/Qwen3.8-27B-FP8 > /scratch2/dl.log 2>&1; echo DL_DONE >> /scratch2/dl.log) &
uv pip install vllm==0.31.0 "huggingface_hub[hf_xet]>=1.5,<2" openai omegaconf opencv-python-headless srt "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*"
C=/scratch2/cbenv/lib/python3.12/site-packages/nvidia/cu13; [ -e $C/lib64 ] || ln -s lib $C/lib64
python -c "import vllm;print('VERS',vllm.__version__)"
wait
echo SETUP_DONE
