#!/bin/bash
# Restarts the full training run if it dies. Runs on the box itself, so it
# survives any client session ending. Resumes from the newest checkpoint.
CFG=/home/ubuntu/adlm/LLaMA-Factory/examples/budget_ad/bins_frames_full.yaml
OUT=/data/cmd_ad/runs/bins_frames_full
LOG=/home/ubuntu/adlm/logs/watchdog.log

while true; do
  sleep 300
  if pgrep -f "llamafactory/launcher.py" > /dev/null; then continue; fi
  if grep -q "FULL RUN DONE" /home/ubuntu/adlm/logs/bins_frames_full.log 2>/dev/null \
     && [ -f "$OUT/all_results.json" ]; then
    echo "$(date -Is) finished cleanly; watchdog exiting" >> $LOG
    exit 0
  fi
  CK=$(ls -d $OUT/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  if [ -n "$CK" ]; then
    sed -i '/^resume_from_checkpoint:/d' $CFG
    echo "resume_from_checkpoint: $OUT/checkpoint-$CK" >> $CFG
    sed -i 's/^overwrite_output_dir: .*/overwrite_output_dir: false/' $CFG
    echo "$(date -Is) restarting from checkpoint-$CK" >> $LOG
  else
    echo "$(date -Is) restarting from scratch (no checkpoint yet)" >> $LOG
  fi
  bash /home/ubuntu/adlm/start_full.sh >> $LOG 2>&1
done
