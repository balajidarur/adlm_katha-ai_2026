# MADX: MAD-Eval writer dataset for one DD source, built exactly as T3 built M3's (T3 bundle/box/a_stage2.sh):
# build_frame_datasets.py (variant B: 1 FPS 448x252 frames in >= 1 s dialogue gaps, cap 200, 48k token budget) on the T3
# windows via t3/mad_build.py (frames_lib monkey-patched to the windows), then the budget-bin legend + 150-frame cap,
# no window dropped. Only the DD source changes (--dd-tag). Window mp4s are re-cut from the 480p movies with T3's exact
# ffmpeg command (a_stage1.sh step 5) because the builder extracts the 1 FPS grid from them.
#   bash build.sh <commit> <q38|4b|q3>
#   q38 = T3's own Qwen3.8 DDs (s3 mad/run/dd_all.tgz) into /scratch/rc_ds: must reproduce T3's dataset (validation).
#   4b / q3 = MADX DDs from s3 adlm/inference/mad/dd/<m>/out -> /scratch/mad_ds_<m>; JSON + frames -> s3 adlm/inference/mad/<model>/dataset/
WANT=$1; M=$2
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
R=/scratch/adlm_repo; X=$R/ops/madx; B=s3://balu-transfer-933516006154-apne1; MR=$B/mad/run
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
cr(){ set -a; . ~/cr.env; set +a; }
cr
log(){ echo "$(date -u +%T) $*" | tee -a /scratch/madx_build.log; }
case $M in
  q38) TAG=Qwen3.8-27B-FP8; OUT=/scratch/rc_ds; NAME=m3;;
  4b) TAG=Qwen3.5-4B; OUT=/scratch/mad_ds_4b; NAME=m6_qwen35_4b-dd_bins_ckpt440;;
  q3) TAG=Qwen3-VL-8B-Instruct; OUT=/scratch/mad_ds_q3; NAME=e3_qwen3vl8b-dd_budget_ckpt376;;
  *) echo BAD; exit 1;;
esac
# windows: T3 rc_win (+ rc_data annotations/dense/clips_all) and frame-accurate re-cuts of the 480p movies
[ -f /scratch/rc_win/windows.json ] || aws s3 cp --only-show-errors $MR/rc_win.tgz - | tar xz -C /scratch/
if [ ! -f /scratch/rcw/.ok ]; then
  mkdir -p /scratch/mad/videos /scratch/rcw
  aws s3 cp --recursive --only-show-errors $B/mad/MADEval/MADeval_video_480p/ /scratch/mad/videos/ --exclude "*" --include "*.mkv"
  python3 -c "
import json
for m, ws in json.load(open('/scratch/rc_win/windows.json')).items():
    for w in ws: print(m, w['wid'], w['start'], round(w['end'] - w['start'], 3))" > /scratch/cuts.txt
  cut1(){ mkdir -p /scratch/rcw/$1; [ -s /scratch/rcw/$1/$2.mp4 ] && return
    ffmpeg -nostdin -loglevel error -y -ss $3 -i /scratch/mad/videos/$1.mkv -t $4 -map 0:v:0 -c:v libx264 -preset veryfast -crf 18 \
      -threads 3 -an /scratch/rcw/$1/$2.tmp.mp4 && mv /scratch/rcw/$1/$2.tmp.mp4 /scratch/rcw/$1/$2.mp4; }
  export -f cut1
  xargs -P ${CUT_P:-3} -L 1 bash -c 'cut1 $0 $1 $2 $3' < /scratch/cuts.txt
  [ $(ls /scratch/rcw/*/*.mp4 | wc -l) = $(wc -l < /scratch/cuts.txt) ] && touch /scratch/rcw/.ok
  log "CUT $(ls /scratch/rcw/*/*.mp4 | wc -l)/$(wc -l < /scratch/cuts.txt) windows; ffmpeg $(ffmpeg -version | head -1 | cut -d' ' -f3)"
fi
# DDs
D=/scratch/dd_src_$M; mkdir -p $D
if [ $M = q38 ]; then [ -n "$(ls $D)" ] || aws s3 cp --only-show-errors $MR/dd_all.tgz - | tar xz -C $D
else aws s3 sync --only-show-errors $B/adlm/inference/mad/dd/$M/out/ $D/ --exclude "*" --include "*/adlm_descriptions.json"; fi
log "DD $M $(ls $D/*/$TAG/adlm_descriptions.json | wc -l) windows"
# dataset code = this commit's dataset/ (as a_stage2: codeA/dataset_prep + add_bins_to_gapframes.py + mk150.py)
A=/scratch/codeA_$M; rm -rf $A; mkdir -p $A && ln -sfn $R/dataset $A/dataset_prep
for f in add_bins_to_gapframes.py mk150.py; do ln -sfn $R/dataset/$f $A/$f; done
rm -rf $OUT/*.json
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ~/adlm/venv/bin/python $X/t3/mad_build.py --code-a $A --root /scratch/rc_data --win /scratch/rc_win \
  --videos /scratch/rcw --dd $D --dd-tag $TAG --out $OUT --workers ${WORKERS:-8} --cap 200 --token-budget 48000 --ffmpeg-threads 1 > /scratch/build_$M.log 2>&1
log "BUILD $M $(grep -E 'built:|post:' /scratch/build_$M.log | tr '\n' ' ') errors=$(python3 -c "
import json; s=json.load(open('$OUT/build_stats.json')); print(s.get('errors', s.get('n_errors', '?')) if isinstance(s, dict) else '?')" 2>/dev/null)"
[ $M = q38 ] && exit 0
# upload: bins JSON (+ meta/order/build_stats) and the frames it references
cd $OUT && python3 -c "
import json, os
d = json.load(open('mad_bins_f150_999k_challenge.json'))
miss = sum(not os.path.exists(p) for r in d for p in r['images'])
print('WINDOWS', len(d), 'FRAMES', sum(len(r['images']) for r in d), 'MAX', max(len(r['images']) for r in d), 'MISSING', miss)
open('$OUT/frames_list.txt', 'w').write('\n'.join(sorted({p.lstrip('/') for r in d for p in r['images']})) + '\n')" | tee -a /scratch/madx_build.log
tar cf - -C / -T $OUT/frames_list.txt | pigz -1 > /scratch/frames_$M.tgz
sha256sum mad_bins_f150_999k_challenge.json cmdad_dds_gapframes_challenge_meta.json order.json build_stats.json > SHA256SUMS
cr; P=$B/adlm/inference/mad/$NAME/dataset
for f in mad_bins_f150_999k_challenge.json cmdad_dds_gapframes_challenge_meta.json order.json build_stats.json SHA256SUMS frames_list.txt; do
  aws s3 cp --only-show-errors $OUT/$f $P/$f; done
aws s3 cp --only-show-errors /scratch/frames_$M.tgz $P/frames.tgz && log "UPLOADED $M -> $P"
