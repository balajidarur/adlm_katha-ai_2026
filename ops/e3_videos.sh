# s4: the AWS mp4s (box A's /data/cmd_ad/videos; s3 srt_for_ads/data/cmd_ad/videos/{year}.tar.gz) for the clips whose
# Qwen3-VL-8B record needs resampled frames the pool doesn't have (budget binds with the longer DD text). Streams each
# year's tar once and keeps only the listed members.   bash e3_videos.sh <need_videos.txt (year/vid lines)>
set -x
W=/opt/dlami/nvme/e3; . $W/cr.env; B=s3://balu-transfer-933516006154-apne1/srt_for_ads/data/cmd_ad/videos
L=$1; V=$W/videos; mkdir -p $V
for y in $(cut -d/ -f1 $L | sort -u); do
  grep "^$y/" $L | sed 's/$/.mp4/' > $W/need_$y.txt
  ( aws s3 cp --only-show-errors $B/$y.tar.gz - | pigz -dc | tar -x -C $V -T $W/need_$y.txt; echo "YEAR $y exit=$? got $(ls $V/$y 2>/dev/null | wc -l)/$(wc -l < $W/need_$y.txt)" | tee -a $W/progress.txt ) &
done
wait
echo "VIDEOS_DONE $(date -u +%H:%M) $(find $V -name '*.mp4' | wc -l)/$(wc -l < $L)" | tee -a $W/progress.txt
