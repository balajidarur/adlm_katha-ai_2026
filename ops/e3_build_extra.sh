# s4, after e3_build.sh: eval rows for the 23 eval591 clips the full build cannot make, exactly as T2 made them for M3-591
# (t2/ops/q4_build.sh, q4_build_hf.sh), with only the dense descriptions swapped to Qwen3-VL-8B:
#   16 A (official HF .mkv, asr_hf), 5 B (/storage .mkv, existing SRT), 2 C (XdMuekkeCeU, yr-HmSz421c: new WhisperX SRT).
# Real videos here, so the plain builder (frames are extracted if the longer DD text needs new resampled frames).
# A/B DDs: s1a run (cmd_ad_eval591/q3vl8b_dense_e3/); C DDs: the full run.   bash e3_build_extra.sh <commit>
WANT=$1; set -x
W=/opt/dlami/nvme/e3; cd $W; . $W/cr.env; B=s3://balu-transfer-933516006154-apne1; S3=$B/cmd_ad_eval591
P(){ echo "$(date -u +%H:%M) $*" | tee -a $W/progress.txt; }
R=$W/adlm_repo; PY=$HOME/adlm/venv/bin/python
HAVE=$(git -C $R rev-parse --short HEAD); [ "$HAVE" = "$WANT" ] || { P COMMIT_MISMATCH; exit 1; }
X=$W/extra; mkdir -p $X/in $X/dense $X/asr/SRT $X/frames
aws s3 sync --only-show-errors $S3/inputs/videos/ $X/in/videos/
aws s3 sync --only-show-errors $S3/inputs/asr_videos/ $X/in/asr_videos/
aws s3 sync --only-show-errors $S3/inputs_hf/videos/ $X/in/hf_videos/
aws s3 sync --only-show-errors $S3/inputs/srt/ $X/asr/SRT/; aws s3 sync --only-show-errors $S3/asr_new/SRT/ $X/asr/SRT/; aws s3 sync --only-show-errors $S3/asr_hf/SRT/ $X/asr/SRT/
aws s3 sync --only-show-errors $S3/build_inputs/ $X/bin/
aws s3 cp --only-show-errors $S3/inputs_hf/shots_cache_gemini-3.5-flash-lite.json $X/shots.json
aws s3 sync --only-show-errors $S3/q3vl8b_dense_e3/ $X/dense/ --exclude "*" --include "*/adlm_descriptions.json"
for v in XdMuekkeCeU yr-HmSz421c; do mkdir -p $X/dense/$v/Qwen3-VL-8B-Instruct; cp $W/dd_q3vl/$v/Qwen3-VL-8B-Instruct/adlm_descriptions.json $X/dense/$v/Qwen3-VL-8B-Instruct/; done
IDS="$(cd $X/dense && ls -d */ | tr -d /)"; echo "IDS ($(echo $IDS | wc -w)): $IDS"
# frames: T2's pools for these clips (same frames as the M3-591 rows); anything new is extracted from the video
for v in $IDS; do
  aws s3 sync --only-show-errors $S3/frames_hf_1fps_448x252/$v/ $X/frames/$v/
  aws s3 sync --only-show-errors $S3/frames_1fps_448x252/$v/ $X/frames/$v/
  [ -d $X/frames/$v ] || aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/$v/ $X/frames/$v/
done
# /data/cmd_ad switched to these inputs for this build (restored afterwards)
C=/data/cmd_ad; sudo mv $C $W/data_cmd_ad_main; sudo mkdir -p $C && sudo chown ubuntu:ubuntu $C; mkdir -p $C/annotations $C/datasets/data $C/dense $C/asr $C/frames
cp -r $W/data_cmd_ad_main/annotations/* $C/annotations/
cp $X/bin/ad_interleave_*_meta.json $C/datasets/data/; cp $X/shots.json $C/dense/shots_cache_gemini-3.5-flash-lite.json
ln -sfn $X/asr/SRT $C/asr/SRT; ln -sfn $X/frames $C/frames/1fps_448x252
for f in $X/in/videos/*/* $X/in/asr_videos/*/* $X/in/hf_videos/*/*; do y=$(basename $(dirname $f)); v=$(basename $f); v=${v%.*}; mkdir -p $C/videos/$y; ln -sfn $f $C/videos/$y/$v.mp4; done
python3 -c "import json;m=json.load(open('$C/datasets/data/ad_interleave_eval_meta.json'));print('index eval', len(m))"
cd $R/dataset && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 \
  DENSE_PATH_TMPL=$X/dense/{vid}/Qwen3-VL-8B-Instruct/adlm_descriptions.json
$PY -u build_frame_datasets.py --n-per-split 0 --splits eval --ids $IDS --workers 23 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $X/build 2>&1 | grep -v "clips:.*%" | tail -12
$PY - <<PY | tee -a $W/progress.txt
import json, sys
sys.path.insert(0, "$R/dataset")
import add_bins_to_gapframes as ab
b = ab.load_budget(); D = "$X/build"
data = json.load(open(f"{D}/cmdad_dds_gapframes_eval.json")); meta = json.load(open(f"{D}/cmdad_dds_gapframes_eval_meta.json"))
out = []
for rec, mt in zip(data, meta):
    n, w = ab.convert(rec, mt["video_id"], b)
    assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])
    out.append(rec)
json.dump(out, open(f"{D}/extra_eval.json", "w"), ensure_ascii=False)
print("EXTRA_BUILT", len(out), "of", len("$IDS".split()), json.load(open(f"{D}/build_stats.json"))["failed"])
PY
sudo rm -rf $C; sudo mv $W/data_cmd_ad_main $C
# frames for these rows live under $X/frames; the records point at /data/cmd_ad/frames/1fps_448x252/<vid>/: merge them in
for v in $IDS; do mkdir -p $W/frames/1fps_448x252/$v; cp -n $X/frames/$v/* $W/frames/1fps_448x252/$v/; done
P EXTRA_DONE
