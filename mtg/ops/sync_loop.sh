# usage: sync_loop.sh OUT_ROOT S3_PREFIX TOTAL  -> every 5 min: progress line to /scratch/mtg/progress.txt, s3 sync of outputs
OUT=$1; S3=$2; TOT=$3
while true; do
  N=$(ls $OUT/*/Qwen3.8-27B-FP8/adlm_descriptions.json 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) $(hostname) done $N/$TOT" >> /scratch/mtg/progress.txt
  . $HOME/.cr.env 2>/dev/null
  aws s3 sync --only-show-errors --region ap-northeast-1 $OUT $S3 --exclude "*/_logs/*" > /scratch/mtg/sync.log 2>&1 || echo "$(date -u +%TZ) SYNC_FAIL" >> /scratch/mtg/progress.txt
  [ -f /scratch/mtg/sync.stop ] && break
  sleep 300
done
