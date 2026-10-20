#!/bin/bash
# Pull prompt-variant outputs (20 clips) from box A and score entailment vs base Qwen + Gemini.
cd /home/dbalu/workspace/ad_gen/adlm_katha-ai_2026 || exit 1
K=$HOME/.ssh/balu-aws-test.pem; A=ubuntu@13.115.233.87
rsync -a --include=*/ --include=adlm_timeline.json --include=adlm_descriptions.json --include=timings.json --exclude=* --prune-empty-dirs \
  -e "ssh -i $K" $A:/prep/qwen_prompt/ output_qwen_prompt/
for ip in 172.31.39.100 172.31.34.103 172.31.36.221 172.31.39.93 15.168.133.218; do
  rsync -a --include=*/ --include=adlm_timeline.json --include=adlm_descriptions.json --include=timings.json --exclude=* --prune-empty-dirs \
    -e "ssh -i $K -o StrictHostKeyChecking=accept-new -o ProxyCommand='ssh -i $K -W %h:%p $A'" ubuntu@$ip:qwen_prompt/ output_qwen_prompt/
done
PY=/workspace/.common_cache/common_share_for_users/stgc/envs/lf_qwen35/bin/python
R=(--run "gemini35fl=output/{vid}/gemini-3.5-flash-lite/adlm_timeline.json" --run "qwen_base=output_qwen_dense/{vid}/Qwen3.8-27B-FP8/adlm_timeline.json")
for v in desc action desc_action desc_action_v2 desc_action_v3 desc_action_v4 desc_action_v5 desc_action_dense; do R+=(--run "$v=output_qwen_prompt/$v/{vid}/Qwen3.8-27B-FP8/adlm_timeline.json"); done
CUDA_VISIBLE_DEVICES=0 $PY eval_entailment.py score --clips output_entail/clips_prompt20.txt --device cuda:0 "${R[@]}" \
  --out output_entail/scores_prompt20.json 2>&1 | grep -v -i "warning\|loading" > logs/prompt_eval.log
tail -3 logs/prompt_eval.log
