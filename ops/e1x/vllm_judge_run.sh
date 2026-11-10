# Serve one judge with vLLM and run llm_ad_eval_vllm.py jobs against it; outputs -> s3 adlm/e1x/metrics/llm_vllm/.
#   bash vllm_judge_run.sh <commit> <l2|l3> "<job args>;<job args>;..."   (job args are passed to llm_ad_eval_vllm.py
#   after --model_dir/--out_dir, e.g. "--csv /scratch/e1x/csv/m3_ckpt440.csv --variant l2 --first_clips 50 --tag s0")
WANT=$1; J=$2; JOBS=$3; W=/scratch; R=$W/adlm_repo; E=$R/ops/e1x; B=s3://balu-transfer-933516006154-apne1/adlm/e1x
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
[ $J = l2 ] && M=$W/judges/Llama-2-7b-chat-hf || M=$W/judges/Meta-Llama-3-8B-Instruct
O=$W/e1x/out_vllm; L=$W/e1x/vllm_$J.log; mkdir -p $O
# VLLM_USE_FLASHINFER_SAMPLER=0: the flashinfer sampler JIT-compiles and needs nvcc (absent on the q boxes);
# the PyTorch sampler applies top-k then top-p by sorting, like HF.
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
$W/vllm/bin/vllm serve $M --dtype float16 --generation-config vllm --port 8000 --max-model-len 4096 --seed 0 \
  --gpu-memory-utilization 0.9 > $W/e1x/server_$J.log 2>&1 < /dev/null &
SP=$!
for i in $(seq 600); do curl -sf localhost:8000/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo SERVER_DIED; tail -30 $W/e1x/server_$J.log; exit 1; }; sleep 2; done
echo "$(date -u +%T) SERVER_UP $J" | tee -a $L
IFS=';' read -ra JA <<< "$JOBS"
for j in "${JA[@]}"; do
  echo "$(date -u +%T) START $j" | tee -a $L
  $W/vllm/bin/python $E/llm_ad_eval_vllm.py --model_dir $M --out_dir $O $j 2>&1 | grep -v Warning | tee -a $L
  echo "$(date -u +%T) END rc=${PIPESTATUS[0]}" | tee -a $L
  . $W/cr.env; aws s3 sync --only-show-errors $O/ $B/metrics/llm_vllm/
done
kill $SP; wait $SP 2>/dev/null
. $W/cr.env; aws s3 cp --only-show-errors $L $B/metrics/llm_vllm/logs/; aws s3 cp --only-show-errors $W/e1x/server_$J.log $B/metrics/llm_vllm/logs/
echo "$(date -u +%T) RUN_DONE $J" | tee -a $L
