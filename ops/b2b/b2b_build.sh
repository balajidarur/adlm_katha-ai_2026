# Box A, CPU only, lowest priority (box A is training): the M3 training data rebuilt with Qwen3.5-2B dense descriptions, plus a
# full eval591. Same build as a_dd4b_build.sh (run_qwen_build.sh args: cap 200, token budget 48k, 448x252, 1 FPS frame pool) and the
# same 0.5 s budget bins; then select_2b.py keeps M3's train clips (40k drop) and writes eval527 and eval591 (150-frame cap, no drop).
# eval591 = eval527 + the 64 clips of cmd_ad_eval591/T2_RESULTS.md: 23 of them (16 no local video, 5 truncated AWS mp4, 2 no
# WhisperX SRT) are rebuilt in an isolated root (CMD_AD_ROOT) from the official videos / ASR in s3 cmd_ad_eval591 (as q4_build*.sh),
# the other 41 come from the main build, kept even when add_bins bins 0 ADs (as T2 did).
#   bash b2b_build.sh <commit> [workers]
WANT=$1; NW=${2:-20}
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH; . /scratch/creds.env
B=s3://balu-transfer-933516006154-apne1; E=$B/cmd_ad_eval591
W=/prep/b2b; R=$W/repo; DD=$W/dd2b; V=$W/V; S=$W/special; SR=$S/root; O=$W/cmdad_qwen35_2b_dd; P=$HOME/adlm/venv/bin/python
NICE="nice -n 19 ionice -c 3"
mkdir -p $W && aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle $W/adlm.bundle && rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 sync --only-show-errors $B/dd_inputs/full2b/out $DD --exclude "*" --include "*/adlm_descriptions.json"
echo "DD2B $(find $DD -name adlm_descriptions.json | wc -l) clips"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 DENSE_PATH_TMPL=$DD/{vid}/Qwen3.5-2B/adlm_descriptions.json
cd $R/dataset

# 1) main build: the standard index (train + eval), frames from / resampled into box A's pool
$NICE $P -u build_frame_datasets.py --n-per-split 0 --workers $NW --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $V 2>&1 | grep -v "it/s\]" | tail -15
for s in train eval; do
  $NICE $P -u add_bins_to_gapframes.py --src $V/cmdad_dds_gapframes_$s.json --meta $V/cmdad_dds_gapframes_${s}_meta.json --out $V/gapframes_bins_$s.json || { echo BINS_FAILED; exit 1; }
done
echo MAIN_BUILD_DONE

# 2) the 23 special eval clips in an isolated root
mkdir -p $SR/annotations $SR/datasets/data $SR/videos $SR/asr/SRT $S/in
ln -sfn /data/cmd_ad/annotations/cmd_ad_anno_v1.csv $SR/annotations/cmd_ad_anno_v1.csv
aws s3 sync --only-show-errors $E/inputs/videos/ $S/in/videos_b/
aws s3 sync --only-show-errors $E/inputs_hf/videos/ $S/in/videos_hf/
aws s3 sync --only-show-errors $E/inputs/srt/ $S/in/srt_b/
aws s3 sync --only-show-errors $E/asr_hf/SRT/ $S/in/srt_hf/
aws s3 sync --only-show-errors $E/asr_new/SRT/ $S/in/srt_new/
for f in $S/in/videos_b/*/*.mkv $S/in/videos_hf/*/*.mkv; do y=$(basename $(dirname $f)); v=$(basename $f .mkv); mkdir -p $SR/videos/$y; ln -sfn $f $SR/videos/$y/$v.mp4; done
for f in $S/in/srt_b/*/*.srt $S/in/srt_hf/*/*.srt $S/in/srt_new/*/*.srt; do y=$(basename $(dirname $f)); mkdir -p $SR/asr/SRT/$y; cp $f $SR/asr/SRT/$y/; done
for c in 2011/XdMuekkeCeU 2014/yr-HmSz421c; do mkdir -p $SR/videos/${c%/*}; ln -sfn /data/cmd_ad/videos/$c.mp4 $SR/videos/$c.mp4; done
$P - <<PY
import json, glob, os
vids = sorted({p.split("/")[-2] + "/" + os.path.basename(p)[:-4] for p in glob.glob("$SR/videos/*/*.mp4")})
json.dump([{"video_id": c.split("/")[1], "year": c.split("/")[0]} for c in vids], open("$SR/datasets/data/ad_interleave_eval_meta.json", "w"))
json.dump([], open("$SR/datasets/data/ad_interleave_train_meta.json", "w"))
print("SPECIAL", len(vids), vids)
PY
IDS=$(python3 -c "import json;print(' '.join(m['video_id'] for m in json.load(open('$SR/datasets/data/ad_interleave_eval_meta.json'))))")
CMD_AD_ROOT=$SR $NICE $P -u build_frame_datasets.py --n-per-split 0 --splits eval --ids $IDS --workers 8 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root $S/frames/1fps_448x252 --out $S/build 2>&1 | grep -v "it/s\]" | tail -8
echo SPECIAL_BUILD_DONE

# 3) assemble train / eval527 / eval591 (150-frame cap), frames needed by eval591 special clips -> s3 frames/eval591_official
aws s3 cp --only-show-errors $E/inputs/missing64.txt $W/missing64.txt
$NICE $P -u $R/ops/b2b/select_2b.py --main $V --special $S/build --m3 /data/cmd_ad/datasets/frame_variants_448x252_full_qwen38 \
  --missing64 $W/missing64.txt --special-frames $S/frames/1fps_448x252 --out $O || { echo SELECT_FAILED; exit 1; }
aws s3 sync --only-show-errors $S/frames/1fps_448x252/ $B/adlm/frames/eval591_official/
# new resampled frames written into box A's pool by step 1 (s3 sync uploads only new files)
aws s3 sync --only-show-errors /data/cmd_ad/frames/1fps_448x252/ $B/adlm/frames/1fps_448x252/
aws s3 sync --only-show-errors $O $B/adlm/datasets/cmdad_qwen35_2b_dd/ && echo DATASET_UPLOADED
