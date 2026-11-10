# HF-vs-vLLM check on the first 50 clips of m3_ckpt440 (q box, after vllm_judge_setup.sh):
# vLLM seeds 0/1 + diagnostics (top_k -1, BOS for l2, server-side chat template), then HF scorer on the subset, seeds 0/1.
#   bash vllm_judge_first50.sh <commit>
C=$1; R=/scratch/adlm_repo/ops/e1x; CSV=/scratch/e1x/csv/m3_ckpt440.csv; O=/scratch/e1x/out_vllm
A="--csv $CSV --first_clips 50"
bash $R/vllm_judge_run.sh $C l3 "$A --variant l3_zero_lenient --tag s0;$A --variant l3_zero_lenient --seed 1 --tag s1;$A --variant l3_zero_lenient --top_k -1 --tag diag_topk-1;$A --variant l3_zero_lenient --endpoint chat --tag diag_chat" || exit 1
bash $R/vllm_judge_run.sh $C l2 "$A --variant l2 --tag s0;$A --variant l2 --seed 1 --tag s1;$A --variant l2 --top_k -1 --tag diag_topk-1;$A --variant l2 --bos --tag diag_bos;$A --variant l2 --endpoint chat --tag diag_chat" || exit 1
for s in 0 1; do for v in l3_zero_lenient l2; do [ $v = l2 ] && m=Llama-2-7b-chat-hf || m=Meta-Llama-3-8B-Instruct
  HF_HUB_OFFLINE=1 $HOME/adlm/venv/bin/python $R/llm_ad_eval_hf.py --csv /scratch/e1x/m3_ckpt440_first50.csv --variant $v \
    --model_dir /scratch/judges/$m --seed $s --out_dir $O/hf50_s$s 2>&1 | grep -v -i warn; done; done
. /scratch/cr.env; aws s3 sync --only-show-errors $O/ s3://balu-transfer-933516006154-apne1/adlm/e1x/metrics/llm_vllm/
echo CHAIN_DONE
