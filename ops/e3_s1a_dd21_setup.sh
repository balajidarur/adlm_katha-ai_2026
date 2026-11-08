# s1a (1x L40S): Qwen3-VL-8B dense descriptions for the 21 eval591 clips the full run never had (16 A = HF official
# .mkv, 5 B = truncated AWS mp4). Same code (full_q3vl8b/prep_code.tgz), server flags (setup_box.sh) and settings as the
# full run: --prompt-variant q3_merge_cover --temperature 0.7 --top-p 0.8, bf16 KV. Prep + ASR = T2's (cmd_ad_eval591).
set -x
W=/opt/dlami/nvme/e3; cd $W; . $W/cr.env
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
B=s3://balu-transfer-933516006154-apne1; F=$B/dd_inputs/full_q3vl8b; T=$B/cmd_ad_eval591
sudo touch /etc/gpu-idle-shutdown.disable
( aws s3 sync --only-show-errors $B/models/Qwen3-VL-8B-Instruct/ $W/Qwen3-VL-8B-Instruct/ && echo MODEL_OK ) &
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x $W/vllm/bin/vllm ] || { uv venv -q -p 3.12 $W/vllm && uv pip install -q --python $W/vllm/bin/python vllm==0.28.0 omegaconf opencv-python-headless openai srt scenedetect; } && echo VENV_OK
mkdir -p $W/code && aws s3 cp --only-show-errors $F/prep_code.tgz $W/code/ && tar -xzf $W/code/prep_code.tgz -C $W/code
aws s3 cp --only-show-errors $T/prep_hf.tar - | tar xf - -C $W
aws s3 cp --only-show-errors $T/prep_360p_t2.tar - | tar xf - -C $W
ls $W
mkdir -p $W/asr_all/SRT
aws s3 sync --only-show-errors $T/inputs/srt/ $W/asr_all/SRT/
aws s3 sync --only-show-errors $T/asr_new/SRT/ $W/asr_all/SRT/
aws s3 sync --only-show-errors $T/asr_hf/SRT/ $W/asr_all/SRT/
aws s3 cp --only-show-errors $T/inputs/new21.txt $W/new21.txt
wait
echo SETUP_DONE
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup $W/vllm/bin/vllm serve $W/Qwen3-VL-8B-Instruct --served-model-name Qwen/Qwen3-VL-8B-Instruct --port 8200 --max-model-len 49152 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 110, "video": 0}' --max-num-seqs 48 --max-num-batched-tokens 16384 \
    --enable-prefix-caching > $W/vllm_0.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s localhost:8200/v1/models | grep -q Qwen3-VL && break; sleep 10; done
curl -s localhost:8200/v1/models | grep -q Qwen3-VL || { echo SERVER_FAILED; exit 1; }
echo SERVER_READY
