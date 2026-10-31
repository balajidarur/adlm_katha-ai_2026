# mirror an ablation run's status to s3 .../qwen38-dd_<name>/_status.txt every 2 min:  bash abl_status.sh <name>
NAME=$1; export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
W=/scratch; [ -d /opt/dlami/nvme ] && W=/opt/dlami/nvme
while true; do
  sleep 120; [ -f $W/creds.env ] && . $W/creds.env
  { date -u +%FT%TZ
    grep -aE "WORKDIR|COMMIT|MISMATCH|DATASET|LF_IMPORT|TRAINER_MD5|GPUS|TRAIN_|FAILED" ~/abl_$NAME.out
    tail -c 4000 $W/runs/train_$NAME.log 2>/dev/null | tr '\r' '\n' | grep -aoE '[0-9]+/[0-9]+ \[[^]]*\]' | tail -1
    tail -1 $W/runs/qwen38_$NAME/trainer_log.jsonl 2>/dev/null
    tail -3 $W/ckpt_sync.log 2>/dev/null
    grep -aiE "error|Traceback" $W/runs/train_$NAME.log 2>/dev/null | tail -2
    tail -2 ~/abl_$NAME.out
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' '; echo
  } > /tmp/status_$NAME.txt 2>&1
  aws s3 cp --only-show-errors /tmp/status_$NAME.txt s3://balu-transfer-933516006154-apne1/adlm/models/cmdad/qwen38-dd_$NAME/_status.txt
done
