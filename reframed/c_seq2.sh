# Box C queue v2 (2026-10-28, after the TF prefix fix): finish REFRAMED budget test free -> REFRAMED budget test TF rerun
# (fixed prefix) -> val TF (fixed) -> val free -> ALL_C_DONE (releases c_cmd_free.sh: Qwen3.8 460/480 CMD-AD free)
# -> CMD-AD TF reruns (fixed prefix) of the Qwen3.8 no-budget checkpoints on box C /data (read-only).
for p in $(pgrep -f "bash c_seq.sh"); do kill -9 $p; done
. /scratch/creds.env; . ~/tfgen.sh
RF=/scratch/rf_datasets/frame_variants_448x252; RC=/scratch/runs/reframed_qwen38_bins_frames/checkpoint-654
Q=/scratch/inf/root/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38; QR=/data/cmd_ad/runs/qwen38_nobudget_frames_full
mk rf_budget654_test_tf2        $RC 1 $RF reframed_bins_f150_999k_test tf
mk rf_budget654_validation_tf   $RC 1 $RF reframed_bins_f150_999k_validation tf
mk rf_budget654_validation_free $RC 1 $RF reframed_bins_f150_999k_validation free:6144
mk qwen38_nobudget_480_tf2 $QR/checkpoint-480 0 $Q/nobudget nobudget_eval527 tf
mk qwen38_nobudget_520_tf2 $QR/checkpoint-520 0 $Q/nobudget nobudget_eval527 tf
adopt rf_budget654_test_free
run rf_budget654_test_tf2
run rf_budget654_validation_tf
run rf_budget654_validation_free
echo "$(date +%T) ALL_C_DONE" >> /scratch/inf/queue.log
while ! grep -q ALL_C_CMD_DONE /scratch/inf/queue.log; do sleep 60; done
for r in qwen38_nobudget_480_tf2 qwen38_nobudget_520_tf2; do run $r; done
echo "$(date +%T) ALL_C_TF2_DONE" >> /scratch/inf/queue.log
