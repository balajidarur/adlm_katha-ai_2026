# Box C queue v3 (user, 2026-10-28): every teacher-forced run first (fixed prefix), then free-running. Replaces c_seq2 + c_cmd_free.
for p in $(pgrep -f "bash c_seq2.sh") $(pgrep -f "bash c_cmd_free.sh"); do kill -9 $p; done
for p in $(pgrep -f "q_rf_budget654_test_free.yaml"); do kill -9 $p; done
echo "$(date +%T) STOPPED rf_budget654_test_free -> queue v3: all TF first, then free" >> /scratch/inf/queue.log
. /scratch/creds.env; . ~/tfgen.sh
Q=/scratch/inf/root/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38
mk qwen38_budget_460_free   /scratch/inf/ckpts/qwen38_budget_460 1 $Q eval527 free:4096
mk qwen38_nobudget_480_free /data/cmd_ad/runs/qwen38_nobudget_frames_full/checkpoint-480 0 $Q/nobudget nobudget_eval527 free:4096
for r in rf_budget654_test_tf2 rf_budget654_validation_tf qwen38_nobudget_480_tf2 qwen38_nobudget_520_tf2 \
         rf_budget654_test_free rf_budget654_validation_free qwen38_budget_460_free qwen38_nobudget_480_free; do run $r; done
echo "$(date +%T) ALL_C_DONE" >> /scratch/inf/queue.log
