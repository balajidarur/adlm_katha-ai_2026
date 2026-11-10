# MADX: dense descriptions (DDs) on the T3 MAD-Eval windows with the CMD-AD full-run recipe of each DD model.
# Inputs are T3's, unchanged: s3 mad/run/rc_win.tgz (windows, per-window dialogue SRTs, GT) and prep_all.tgz
# (CMD-AD shot detector shots, 360p 3 FPS frames, InsightFace matches to the IMDb top-10 cast, face circles).
#   4b : Qwen/Qwen3.5-4B,  vLLM 0.28, --prompt-variant desc_action_v2, --temperature 0.7 --top-p 0.8, 24 clips/server
#        (= s3 dd_inputs/full4b/{setup_box.sh,full_worker.sh}, the M6 CMD-AD DDs)
#   q3 : Qwen3-VL-8B-Instruct (s3 models/ copy), vLLM 0.28, --prompt-variant q3_merge_cover, --temperature 0.7 --top-p 0.8,
#        16 clips/server, bf16 KV (= s3 dd_inputs/full_q3vl8b/{setup_box.sh,full_worker.sh}, the E3 CMD-AD DDs)
#        KV=fp8 env: --kv-cache-dtype fp8 --attention-backend TRITON_ATTN (used for ~1.9k CMD-AD clips; only if time-pressed)
# DD code: ops/madx/dd_code = s3 dd_inputs/full_q3vl8b/prep_code.tgz (a strict superset of full4b's: adds the q3_* prompt
# variants; every other file identical). Clip list: /scratch/madx_ready_<m>.txt (follow mode: clips can be appended
# while it runs; touch /scratch/madx_done_<m> to let it finish). Outputs -> s3 adlm/inference/mad/dd/<m>/out/{wid}/<model>/
#   bash dd.sh <commit> <4b|q3>
WANT=$1; M=$2
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
R=/scratch/adlm_repo; C=$R/ops/madx/dd_code; B=s3://balu-transfer-933516006154-apne1; MR=$B/mad/run
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
cr(){ set -a; . ~/cr.env; set +a; }
cr
case $M in
  4b) MODEL=Qwen/Qwen3.5-4B; PV=desc_action_v2; PER=24; PORT=8100;;
  q3) MODEL=Qwen/Qwen3-VL-8B-Instruct; PV=q3_merge_cover; PER=16; PORT=8200;;
  *) echo BAD_MODEL; exit 1;;
esac
TAG=${MODEL#*/}; O=/scratch/dd_$M; S3O=$B/adlm/inference/mad/dd/$M; mkdir -p $O
log(){ echo "$(date -u +%T) $*" | tee -a /scratch/madx_dd_$M.log; }
log "START $M model=$MODEL pv=$PV commit=$HAVE"
# vLLM 0.28 venv (as both CMD-AD full runs)
if [ ! -x /scratch/vllm/bin/vllm ]; then
  uv venv -q -p 3.12 /scratch/vllm && UV_CACHE_DIR=/scratch/uvcache uv pip install -q --python /scratch/vllm/bin/python \
    vllm==0.28.0 omegaconf opencv-python-headless openai srt scenedetect || { log VENV_FAILED; exit 1; }
fi
log "VENV $(/scratch/vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)"
# weights
if [ $M = 4b ]; then
  HF_HOME=/scratch/hf /scratch/vllm/bin/python -c "
from huggingface_hub import snapshot_download; p = snapshot_download('Qwen/Qwen3.5-4B'); print('MODEL_PATH', p)" | tail -1 | tee -a /scratch/madx_dd_$M.log
  SERVE=(Qwen/Qwen3.5-4B --reasoning-parser qwen3); GREP=4B
else
  [ -f /scratch/Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors ] || aws s3 sync --only-show-errors $B/models/Qwen3-VL-8B-Instruct/ /scratch/Qwen3-VL-8B-Instruct/
  SERVE=(/scratch/Qwen3-VL-8B-Instruct --served-model-name Qwen/Qwen3-VL-8B-Instruct); GREP=Qwen3-VL
  [ "${KV:-bf16}" = fp8 ] && SERVE+=(--kv-cache-dtype fp8 --attention-backend TRITON_ATTN)
fi
# T3 inputs (unchanged)
if [ ! -f /scratch/rc_prep_360p/.ok ]; then
  aws s3 cp --only-show-errors $MR/rc_win.tgz - | tar xz -C /scratch/
  mkdir -p /scratch/rc_prep_360p && aws s3 cp --only-show-errors $MR/prep_all.tgz - | pigz -dc | tar x -C /scratch/rc_prep_360p && touch /scratch/rc_prep_360p/.ok
fi
log "INPUTS windows=$(ls -d /scratch/rc_prep_360p/tt*/*/ | wc -l) srt=$(ls /scratch/rc_win/srt/dialogue/*/*.srt | wc -l)"
touch /scratch/madx_ready_$M.txt
CUDA_VISIBLE_DEVICES=0 HF_HOME=/scratch/hf HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /scratch/vllm/bin/vllm serve "${SERVE[@]}" --port $PORT --max-model-len 49152 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 110, "video": 0}' --max-num-seqs 48 --max-num-batched-tokens 16384 --enable-prefix-caching \
    > /scratch/vllm_$M.log 2>&1 < /dev/null &
for i in $(seq 1 120); do curl -s localhost:$PORT/v1/models | grep -q $GREP && break; sleep 10; done
curl -s localhost:$PORT/v1/models | grep -q $GREP || { log SERVER_FAILED; exit 1; }
log "SERVER_READY port=$PORT"
( while sleep 300; do cr; aws s3 sync --only-show-errors $O $S3O/out --exclude "*" --include "*/adlm_descriptions.json" --include "*/adlm_timeline.json"
    log "PROGRESS $(ls $O/*/$TAG/adlm_descriptions.json 2>/dev/null | wc -l) done / $(wc -l < /scratch/madx_ready_$M.txt) listed"
    [ -f /scratch/madx_finished_$M ] && break; done ) &
cd $C; export WHISPERX_SRT=/scratch/rc_win/srt/dialogue CMD_VIDEO_BASE=/scratch/rc_data/videos PYTHONUNBUFFERED=1 FOLLOW_DONE=/scratch/madx_done_$M
while true; do
  /scratch/vllm/bin/python -u run_dense_all.py --clips /scratch/madx_ready_$M.txt --prep-root /scratch/rc_prep_360p --out-root $O \
    --ports $PORT --per-server $PER --prompt-variant $PV --model $MODEL --temperature 0.7 --top-p 0.8 >> $O/driver.log 2>&1
  left=$(tail -50 $O/driver.log | grep -oE "[0-9]+ to do" | tail -1)
  [ -f /scratch/madx_done_$M ] && [ "$left" = "0 to do" ] && break
  sleep 30
done
touch /scratch/madx_finished_$M; cr
aws s3 sync --only-show-errors $O $S3O/out --exclude "*" --include "*/adlm_descriptions.json" --include "*/adlm_timeline.json"
aws s3 cp --only-show-errors $O/driver.log $S3O/logs/driver_$(hostname).log; aws s3 cp --only-show-errors $O/run_log.jsonl $S3O/logs/run_log_$(hostname).jsonl
log "DD_DONE $(ls $O/*/$TAG/adlm_descriptions.json | wc -l) clips"
