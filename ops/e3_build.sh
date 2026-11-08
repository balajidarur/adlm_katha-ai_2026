# s4: E3 dataset = M3's build (build_frame_datasets: cap 200, 48k budget, 448x252 1 FPS gap frames, same pool) + add_bins
# + M3's clip selection, with only the dense descriptions swapped to Qwen3-VL-8B (DENSE_PATH_TMPL), like the 4B-DD run.
# No videos on s4: build_frame_datasets_durs.py feeds M3's recorded durations. Validation = the same build with the
# Qwen3.8 DDs must reproduce M3's records exactly.     bash e3_build.sh <commit>
WANT=$1; set -x
W=/opt/dlami/nvme/e3; cd $W; . $W/cr.env; B=s3://balu-transfer-933516006154-apne1
P(){ echo "$(date -u +%H:%M) $*" | tee -a $W/progress.txt; }
R=$W/adlm_repo; PY=$HOME/adlm/venv/bin/python
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { P COMMIT_MISMATCH; exit 1; }
M3=$W/m3/frame_variants_448x252_full_qwen38
# /data/cmd_ad from box A's own copies (evaldata.tgz annotations), the index metas, the DD-run SRT set, the frame pool
[ -f $W/ref/data/cmd_ad/annotations/cmd_ad_anno_v1.csv ] || tar -xzf $W/ref/evaldata.tgz -C $W/ref --wildcards "data/cmd_ad/annotations/*" "data/cmd_ad/datasets/*"
sudo mkdir -p /data/cmd_ad && sudo chown -R ubuntu:ubuntu /data/cmd_ad; C=/data/cmd_ad; sudo rm -f $C/frames/1fps_448x252  # stale link from an earlier run
mkdir -p $C/annotations $C/datasets/data $C/dense $C/asr $C/frames
cp -r $W/ref/data/cmd_ad/annotations/* $C/annotations/
cp $W/ad_interleave_train_meta.json $W/ad_interleave_eval_meta.json $C/datasets/data/
aws s3 cp --only-show-errors $B/cmd_ad_eval591/inputs/shots_cache_gemini-3.5-flash-lite.json $C/dense/
[ -d $W/asr/SRT ] || { mkdir -p $W/asr && aws s3 cp --only-show-errors $B/dd_inputs/entail4b/asr_srt.tar.gz $W/ && tar -xzf $W/asr_srt.tar.gz -C $W/asr; }
ln -sfn $W/asr/SRT $C/asr/SRT; ln -sfn $W/frames/1fps_448x252 $C/frames/1fps_448x252
ls $C/asr/SRT | head -3; ls $C/frames/1fps_448x252 | wc -l
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export BUILD_VARIANTS=B DUR_META=$M3/cmdad_dds_gapframes_train_meta.json,$M3/cmdad_dds_gapframes_eval_meta.json
cd $R/dataset
build(){ # $1 = DENSE_PATH_TMPL, $2 = out dir
  DENSE_PATH_TMPL=$1 $PY -u build_frame_datasets_durs.py --n-per-split 0 --workers 46 --ffmpeg-threads 1 \
    --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $2 > $2.log 2>&1
  tail -12 $2.log | grep -v "clips:.*%"
  n=$(python3 -c "import json;print(len(json.load(open('$2/build_stats.json'))['failed']))"); [ "$n" -lt 300 ] || { P "BUILD_FAILED $2 ($n failed)"; exit 1; }; }
P "BUILD_Q38_START (validation)"
build $W/dd_q38/qwen_dense_final/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json $W/v_q38 || true
$PY - <<PY | tee -a $W/progress.txt
import json
for s in ("train", "eval"):
    a = json.load(open("$M3/cmdad_dds_gapframes_%s.json" % s)); b = json.load(open("$W/v_q38/cmdad_dds_gapframes_%s.json" % s))
    am = json.load(open("$M3/cmdad_dds_gapframes_%s_meta.json" % s)); bm = json.load(open("$W/v_q38/cmdad_dds_gapframes_%s_meta.json" % s))
    ka = {m["video_id"]: r for r, m in zip(a, am)}; kb = {m["video_id"]: r for r, m in zip(b, bm)}
    same = sum(ka[v] == kb.get(v) for v in ka)
    print(f"VALIDATE {s}: M3 {len(a)} records, rebuilt {len(b)}, identical {same}, only-M3 {len(set(ka)-set(kb))}, only-rebuilt {len(set(kb)-set(ka))}")
PY
P "BUILD_Q3VL_START"
build $W/dd_q3vl/{vid}/Qwen3-VL-8B-Instruct/adlm_descriptions.json $W/v_q3vl
P "BUILD_Q3VL_DONE $(python3 -c "import json;print(json.load(open('$W/v_q3vl/build_stats.json'))['failed'][:40])")"
for s in train eval; do
  $PY -u add_bins_to_gapframes.py --src $W/v_q3vl/cmdad_dds_gapframes_$s.json --meta $W/v_q3vl/cmdad_dds_gapframes_${s}_meta.json \
    --out $W/v_q3vl/gapframes_bins_$s.json | tail -1 | tee -a $W/progress.txt
done
P BINS_DONE
