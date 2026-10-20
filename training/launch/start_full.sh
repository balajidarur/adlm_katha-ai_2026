#!/bin/bash
# tmux session outlives the job; log APPENDS so a restart never destroys the
# previous crash traceback.
tmux kill-session -t full 2>/dev/null
tmux new-session -d -s full
tmux send-keys -t full 'echo "=== launch $(date -Is) ===" >> ~/adlm/logs/bins_frames_full.log; bash ~/adlm/run_full.sh 2>&1 | tee -a ~/adlm/logs/bins_frames_full.log; echo; echo "=== job exited; tmux shell still alive ==="' C-m
echo "tmux session 'full' started"
