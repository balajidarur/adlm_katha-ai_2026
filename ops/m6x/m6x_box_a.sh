# Box A: M6 (Qwen3.5-4B DDs + budget writer, ckpt-440) teacher-forced on the full CMD-AD eval591.
# 1) the 64 extra eval591 clips built with 4B DDs (main eval build for the ones in the standard index, isolated build with the
#    official videos / ASR for the 23 special ones, as ops/b2b/b2b_build.sh steps 1-2 but eval split only); 2) select_m6x.py
#    (rows 0-526 = M6's exact eval527 records); 3) TF on 4 GPUs with ops/xeval/xe_infer.sh (TF2 settings, bins).
#   bash m6x_box_a.sh <commit>          -> s3 adlm/datasets/cmdad_qwen35_4b_dd/dd4b_eval591.json, s3 adlm/xeval/preds/m6_cmd591_s*of4/
WANT=$1; set -x; exec >> $HOME/m6x.log 2>&1; echo "START $(date -u)"
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH AWS_DEFAULT_REGION=ap-northeast-1; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
B=s3://balu-transfer-933516006154-apne1; E=$B/cmd_ad_eval591
W=/prep/m6x; R=/scratch/adlm_repo; DD=$W/dd4b; V=$W/V; S=$W/special; SR=$S/root; O=$W/out; P=$HOME/adlm/venv/bin/python
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p /scratch && sudo mount /dev/$DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
sudo mkdir -p $W && sudo chown ubuntu:ubuntu $W /prep
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle /scratch/m6x.bundle && rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/m6x.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 sync --only-show-errors $B/dd_inputs/full4b/out $DD --exclude "*" --include "*/adlm_descriptions.json"
aws s3 sync --only-show-errors $B/dd_inputs/full4b/out21 $DD --exclude "*" --include "*/adlm_descriptions.json"
echo "DD4B $(find $DD -name adlm_descriptions.json | wc -l) clips"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 DENSE_PATH_TMPL=$DD/{vid}/Qwen3.5-4B/adlm_descriptions.json
aws s3 cp --only-show-errors $E/inputs/missing64.txt $W/missing64.txt
IDS64=$(sed 's#.*/##' $W/missing64.txt | tr '\n' ' ')
cd $R/dataset
# 1a) main eval build for the 64 (clips not in the standard index fail there and come from the isolated build)
$P -u build_frame_datasets.py --n-per-split 0 --splits eval --ids $IDS64 --workers 32 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $V 2>&1 | grep -v "it/s\]" | tail -6
$P -u add_bins_to_gapframes.py --src $V/cmdad_dds_gapframes_eval.json --meta $V/cmdad_dds_gapframes_eval_meta.json --out $V/gapframes_bins_eval.json || echo BINS_WARN
# 1b) the 23 special clips (copied from b2b_build.sh step 2)
mkdir -p $SR/annotations $SR/datasets/data $SR/videos $SR/asr/SRT $SR/dense $S/in
aws s3 cp --only-show-errors $E/inputs/shots_cache_gemini-3.5-flash-lite.json $S/in/shots_b.json
aws s3 cp --only-show-errors $E/inputs_hf/shots_cache_gemini-3.5-flash-lite.json $S/in/shots_hf.json
$P - <<PY
import json
c = json.load(open("/data/cmd_ad/dense/shots_cache_gemini-3.5-flash-lite.json"))
for f in ("$S/in/shots_b.json", "$S/in/shots_hf.json"): c.update(json.load(open(f)))
json.dump(c, open("$SR/dense/shots_cache_gemini-3.5-flash-lite.json", "w")); print("SHOTS", len(c))
PY
ln -sfn /data/cmd_ad/annotations/cmd_ad_anno_v1.csv $SR/annotations/cmd_ad_anno_v1.csv
aws s3 sync --only-show-errors $E/inputs/videos/ $S/in/videos_b/; aws s3 sync --only-show-errors $E/inputs_hf/videos/ $S/in/videos_hf/
aws s3 sync --only-show-errors $E/inputs/srt/ $S/in/srt_b/; aws s3 sync --only-show-errors $E/asr_hf/SRT/ $S/in/srt_hf/; aws s3 sync --only-show-errors $E/asr_new/SRT/ $S/in/srt_new/
for f in $S/in/videos_b/*/*.mkv $S/in/videos_hf/*/*.mkv; do y=$(basename $(dirname $f)); v=$(basename $f .mkv); mkdir -p $SR/videos/$y; ln -sfn $f $SR/videos/$y/$v.mp4; done
for f in $S/in/srt_b/*/*.srt $S/in/srt_hf/*/*.srt $S/in/srt_new/*/*.srt; do y=$(basename $(dirname $f)); mkdir -p $SR/asr/SRT/$y; cp $f $SR/asr/SRT/$y/; done
for c in 2011/XdMuekkeCeU 2014/yr-HmSz421c; do mkdir -p $SR/videos/${c%/*}; ln -sfn /data/cmd_ad/videos/$c.mp4 $SR/videos/$c.mp4; done
$P - <<PY
import json, glob, os
vids = sorted({p.split("/")[-2] + "/" + os.path.basename(p)[:-4] for p in glob.glob("$SR/videos/*/*.mp4")})
json.dump([{"video_id": c.split("/")[1], "year": c.split("/")[0]} for c in vids], open("$SR/datasets/data/ad_interleave_eval_meta.json", "w"))
json.dump([], open("$SR/datasets/data/ad_interleave_train_meta.json", "w")); print("SPECIAL", len(vids))
PY
IDS=$(python3 -c "import json;print(' '.join(m['video_id'] for m in json.load(open('$SR/datasets/data/ad_interleave_eval_meta.json'))))")
CMD_AD_ROOT=$SR $P -u build_frame_datasets.py --n-per-split 0 --splits eval --ids $IDS --workers 8 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root $S/frames/1fps_448x252 --out $S/build 2>&1 | grep -v "it/s\]" | tail -6
[ -s $S/build/cmdad_dds_gapframes_eval.json ] || { echo SPECIAL_BUILD_FAILED; exit 1; }
# 2) assemble eval591
aws s3 cp --only-show-errors $B/adlm/datasets/cmdad_qwen35_4b_dd/dd4b_eval527.json $W/dd4b_eval527.json
$P -u $R/ops/m6x/select_m6x.py --eval527 $W/dd4b_eval527.json --main $V --special $S/build --missing64 $W/missing64.txt \
  --special-frames $S/frames/1fps_448x252 --out $O || { echo SELECT_FAILED; exit 1; }
sudo mkdir -p /data/cmd_ad/frames/eval591_official && sudo chown ubuntu:ubuntu /data/cmd_ad/frames/eval591_official
rsync -a $S/frames/1fps_448x252/ /data/cmd_ad/frames/eval591_official/
$P -c "import json,os;d=json.load(open('$O/dd4b_eval591.json'));m=[p for r in d for p in r['images'] if not os.path.exists(p)];print('EVAL591',len(d),'MISSING',len(m),m[:3])"
. $HOME/.cr.env; aws s3 sync --only-show-errors $O $B/adlm/datasets/cmdad_qwen35_4b_dd/ && echo DATASET_UPLOADED
aws s3 sync --only-show-errors $S/frames/1fps_448x252/ $B/adlm/frames/eval591_official_dd4b/
# 3) TF on all GPUs (env from s3 tf_rerun if missing)
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null || [ ! -d $HOME/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B ]; then
  mkdir -p /scratch/pack; aws s3 cp --only-show-errors --recursive $B/adlm/tf_rerun/ /scratch/pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc /scratch/pack/$f.tgz | sudo tar -x -C / ; done; sudo chown -R ubuntu:ubuntu $HOME/adlm
fi
MD5=$(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10); echo "TRAINER_MD5 $MD5"; [ "$MD5" = 955de66d8a ] || { echo TRAINER_MISMATCH; exit 1; }
CKD=/scratch/ckpt/m6_440; aws s3 cp --only-show-errors --recursive $B/adlm/models/cmdad/qwen35_4b-dd_bins/checkpoint-440/ $CKD/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
NG=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((NG - 1))); do bash $R/ops/xeval/xe_infer.sh $WANT $CKD m6_cmd591 $O/dd4b_eval591.json $g $g $NG bins tf > /scratch/m6x_g$g.log 2>&1 & done; wait
grep -h -E "^END|SYNCED|MISMATCH" /scratch/m6x_g*.log; echo "ALL_DONE $(date -u)"
