# usage: prep_list.sh LIST.tsv [JOBS]   TSV: key video_path video_id offset clip_dur(or -)
# writes /scratch/mtg/prep/<key>/ and /scratch/mtg/srt/<key>.srt; log /scratch/mtg/prep_logs/
. /scratch/cbenv/bin/activate
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p /scratch/mtg/prep_logs
export HERE
one() {
  key=$1; vid=$2; id=$3; off=$4; dur=$5
  [ -f /scratch/mtg/prep/$key/meta.json ] && return 0
  extra=""; [ "$dur" != "-" ] && extra="--clip-dur $dur"
  python $HERE/prep_nocast.py --video="$vid" --key="$key" --video-id="$id" --offset=$off $extra \
    --out-root /scratch/mtg/prep --srt-root /scratch/mtg/srt --mtg-jsonl /scratch/mtg/all.jsonl \
    > /scratch/mtg/prep_logs/${key//\//_}.log 2>&1 || echo "PREP_FAIL $key"
}
export -f one
cat /scratch/mtg/hf/data/annotations/{test,validation}.jsonl > /scratch/mtg/all.jsonl
grep -v '^#' $1 | xargs -P ${2:-32} -L 1 bash -c 'one "$@"' _
echo PREP_DONE
