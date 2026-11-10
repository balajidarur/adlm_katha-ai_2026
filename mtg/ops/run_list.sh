# usage: run_list.sh CLIPS.txt OUT_ROOT VARIANT [PER_SERVER] [PORTS...]
. /scratch/cbenv/bin/activate
HERE=$(cd "$(dirname "$0")/.." && pwd)
CL=$1; OUT=$2; V=$3; PS=${4:-16}; shift 4; PORTS=${@:-8000 8001 8002 8003}
export WHISPERX_SRT=/scratch/mtg/srt CMD_VIDEO_BASE=/scratch/mtg/none
python $HERE/run_dense_all.py --clips $CL --prep-root /scratch/mtg/prep --out-root $OUT --ports $PORTS \
  --per-server $PS --prompt-variant $V --no-cast ${SKIP_FILE:+--skip-file $SKIP_FILE}
