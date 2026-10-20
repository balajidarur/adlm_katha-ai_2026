#!/bin/bash
# tmux session outlives the job; log APPENDS so a restart never destroys the
# previous crash traceback.
tmux kill-session -t qwen38 2>/dev/null
tmux new-session -d -s qwen38
tmux send-keys -t qwen38 'echo "=== launch $(date -Is) ===" >> ~/adlm/logs/qwen38_bins_frames_full.log; bash ~/adlm/run_qwen38.sh 2>&1 | tee -a ~/adlm/logs/qwen38_bins_frames_full.log; echo; echo "=== job exited; tmux shell still alive ==="' C-m
echo "tmux session qwen38 started"
