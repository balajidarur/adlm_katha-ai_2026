# Box A queue v2 (2026-10-28, after the TF prefix fix): finish REFRAMED no-budget free -> REFRAMED no-budget TF rerun
# (fixed prefix) -> CMD-AD Gemini budget/no-budget 696 free -> CMD-AD TF reruns (fixed prefix) of the box-A checkpoints.
for p in $(pgrep -f "bash a_seq.sh"); do kill -9 $p; done
. ~/tfgen.sh
D=/data/cmd_ad/datasets; Q=$D/frame_variants_448x252_full_qwen38
mk rf_nobudget654_test_tf2 /scratch/runs/reframed_qwen38_nobudget_frames/checkpoint-654 0 /scratch/rf_datasets/frame_variants_448x252 reframed_nobudget_f150_999k_test tf
mk gemini_budget_696_tf2   /data/cmd_ad/runs/bins_frames_full/checkpoint-696 1 $D/bins_frames eval_full tf
mk gemini_nobudget_696_tf2 /scratch/inf/ckpts/gemini_nobudget_696 0 $D/nobudget_frames eval_full tf
mk gemini_nobudget_600_tf2 /scratch/inf/ckpts/gemini_nobudget_600 0 $D/nobudget_frames eval_full tf
mk qwen38_budget_460_tf2   /prep/eval_ckpts/bins_ckpt460 1 $Q eval527 tf
aws s3 sync --quiet s3://balu-transfer-933516006154-apne1/adlm/ckpts/cmdad_gemini_nobudget/checkpoint-600/ /scratch/inf/ckpts/gemini_nobudget_600/ &
adopt rf_nobudget654_test_free
run rf_nobudget654_test_tf2
run gemini_budget_696_free
run gemini_nobudget_696_free
wait
for r in gemini_budget_696_tf2 gemini_nobudget_696_tf2 qwen38_budget_460_tf2 gemini_nobudget_600_tf2; do run $r; done
echo "$(date +%T) ALL_A_DONE" >> /scratch/inf/queue.log
