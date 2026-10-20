#!/bin/bash
# Pull finished Qwen3.8-27B timelines from the AWS run (box A + q1-q4) and score entailment vs Gemini 3.5 FL.
# Appends one summary line per run to output_entail/periodic.log.   (cron: every 3 h)
cd /home/dbalu/workspace/ad_gen/adlm_katha-ai_2026 || exit 1
exec 9>/tmp/periodic_eval.lock; flock -n 9 || exit 0
K=$HOME/.ssh/balu-aws-test.pem; A=ubuntu@13.115.233.87
SYNC="rsync -a --include=*/ --include=adlm_timeline.json --include=adlm_descriptions.json --exclude=* --prune-empty-dirs"
$SYNC -e "ssh -i $K" $A:/prep/qwen_dense_final/ output_qwen_dense_final/
python3 - <<'PY'
import glob, json
out = []
for f in sorted(glob.glob("output_qwen_dense_final/*/Qwen3.8-27B-FP8/adlm_descriptions.json")):
    vid = f.split("/")[1]
    if glob.glob(f"output/{vid}/gemini-3.5-flash-lite/adlm_timeline.json"):
        out.append(json.load(open(f))["cmd_filename"])
open("output_entail/clips_periodic.txt", "w").write("\n".join(out) + "\n")
print(len(out), "clips")
PY
python3 eval_entailment.py claims --clips output_entail/clips_periodic.txt
PY=/workspace/.common_cache/common_share_for_users/stgc/envs/lf_qwen35/bin/python
N=$(wc -l < output_entail/clips_periodic.txt)
CUDA_VISIBLE_DEVICES=0 $PY eval_entailment.py score --clips output_entail/clips_periodic.txt --device cuda:0 \
  --run "qwen27b_v2final=output_qwen_dense_final/{vid}/Qwen3.8-27B-FP8/adlm_timeline.json" \
  --run "gemini35fl=output/{vid}/gemini-3.5-flash-lite/adlm_timeline.json" \
  --out output_entail/scores_periodic.json > logs/periodic_eval_last.log 2>&1
echo "$(date '+%F %T') clips=$N $(grep '^ALL' logs/periodic_eval_last.log)" >> output_entail/periodic.log
