# start one vLLM server per GPU (ports 8000-8003), logs /scratch/vllm_<gpu>.log
cd "$(dirname "$0")"
for g in $(seq 0 $(( ${NGPU:-4} - 1 ))); do nohup bash serve.sh $g > /scratch/vllm_$g.log 2>&1 & done
echo started
