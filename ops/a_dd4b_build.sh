# Box A: the M3 training data rebuilt with Qwen3.5-4B dense descriptions instead of Qwen3.8 - same build
# (run_qwen_build.sh: cap 200, token budget 48k, 1 FPS 448x252, same frame pool), same 0.5 s budget bins, same
# 150-frame cap, and M3's exact clips (select_like_m3.py). Locked in s3 adlm/datasets/cmdad_qwen35_4b_dd (SHA256SUMS).
#   bash a_dd4b_build.sh <commit>
WANT=$1
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH; mv /tmp/cr.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
B=s3://balu-transfer-933516006154-apne1
R=/scratch/adlm_repo_dd4b; DD=/scratch/dd4b_out; V=/scratch/datasets/frame_variants_448x252_full_qwen35_4b
O=/scratch/datasets/cmdad_qwen35_4b_dd; M3=/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle /scratch/adlm_dd4b.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm_dd4b.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 sync --only-show-errors $B/dd_inputs/full4b/out $DD --exclude "*" --include "*/adlm_descriptions.json"
echo "DD4B $(find $DD -name adlm_descriptions.json | wc -l) clips"
cd $R/dataset
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export DENSE_PATH_TMPL=$DD/{vid}/Qwen3.5-4B/adlm_descriptions.json
~/adlm/venv/bin/python -u build_frame_datasets.py --n-per-split 0 --workers 44 --ffmpeg-threads 1 \
  --cap 200 --token-budget 48000 --frame-root /data/cmd_ad/frames/1fps_448x252 --out $V || { echo BUILD_FAILED; exit 1; }
for s in train eval; do
  ~/adlm/venv/bin/python -u add_bins_to_gapframes.py --src $V/cmdad_dds_gapframes_$s.json \
    --meta $V/cmdad_dds_gapframes_${s}_meta.json --out $V/gapframes_bins_$s.json || { echo BINS_FAILED; exit 1; }
done
~/adlm/venv/bin/python -u select_like_m3.py --name dd4b --bins-dir $V --m3-dir $M3 --out $O || { echo SELECT_FAILED; exit 1; }
aws s3 sync --only-show-errors $O $B/adlm/datasets/cmdad_qwen35_4b_dd/ && echo DATASET_UPLOADED
