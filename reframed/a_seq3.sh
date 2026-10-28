# Box A queue v3 (user, 2026-10-28): every teacher-forced run first (fixed prefix), then free-running.
for p in $(pgrep -f "bash a_seq2.sh"); do kill -9 $p; done
for p in $(pgrep -f "q_rf_nobudget654_test_free.yaml"); do kill -9 $p; done
echo "$(date +%T) STOPPED rf_nobudget654_test_free -> queue v3: all TF first, then free" >> /scratch/inf/queue.log
. ~/tfgen.sh
[ -f /scratch/inf/ckpts/gemini_nobudget_600/adapter_model.safetensors ] || aws s3 sync --quiet s3://balu-transfer-933516006154-apne1/adlm/ckpts/cmdad_gemini_nobudget/checkpoint-600/ /scratch/inf/ckpts/gemini_nobudget_600/
for r in rf_nobudget654_test_tf2 gemini_budget_696_tf2 gemini_nobudget_696_tf2 qwen38_budget_460_tf2 gemini_nobudget_600_tf2 \
         rf_nobudget654_test_free gemini_budget_696_free gemini_nobudget_696_free; do run $r; done
echo "$(date +%T) ALL_A_DONE" >> /scratch/inf/queue.log
