# Box A: build G1 (1 s budget bins) and A6 (no DDs, whole-clip frames <= 150) from the M3 sources on /data + /prep,
# lock them in S3 (SHA256SUMS). T1 (plain-text budget tags) is built off-box from the locked M3 JSONs.
#   bash a_new3_build.sh <commit>
WANT=$1
set -x
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme3n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mv /tmp/cr.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
B=s3://balu-transfer-933516006154-apne1/adlm; R=/scratch/adlm_repo_new3; M3=/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38
G=/scratch/datasets/cmdad_qwen38_bins1s; V=/scratch/datasets/frame_variants_448x252_full_qwen38_c; A6=/scratch/datasets/cmdad_qwen38_a6_fullframes
PY=$HOME/adlm/venv/bin/python
aws s3 cp --only-show-errors $B/code/adlm_$WANT.bundle /scratch/adlm_new3.bundle
rm -rf $R && git clone -q -b qwen38-dense-dataset-training /scratch/adlm_new3.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
cd $R/dataset
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
# G1
$PY -u add_bins1s.py --q $M3 --out $G || { echo G1_FAILED; exit 1; }
(cd $G && sha256sum bins1s_f150_40k.json bins1s_eval527.json dataset_info.json > SHA256SUMS)
aws s3 sync --only-show-errors $G $B/datasets/cmdad_qwen38_bins1s/ && echo G1_UPLOADED
# A6: variant C (DDs + every 1 fps frame) with the Qwen3.8 DDs, same build args as M3 (run_qwen_build.sh)
ls /prep/qwen_dense_final | head -2 || { echo QWEN38_DD_MISSING; exit 1; }
export DENSE_PATH_TMPL=/prep/qwen_dense_final/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json
$PY -u build_frame_datasets.py --n-per-split 0 --workers 44 --ffmpeg-threads 1 --cap 200 --token-budget 48000 \
  --frame-root /data/cmd_ad/frames/1fps_448x252 --out $V || { echo BUILD_FAILED; exit 1; }
for s in train eval; do
  $PY -u add_bins_to_gapframes.py --src $V/cmdad_dds_frames_$s.json --meta $V/cmdad_dds_frames_${s}_meta.json \
    --out $V/cmdad_dds_frames_bins_$s.json || { echo BINS_FAILED; exit 1; }
done
$PY -u make_a6_fullframes.py --bins-dir $V --m3-dir $M3 --out $A6 || { echo A6_FAILED; exit 1; }
aws s3 sync --only-show-errors $A6 $B/datasets/cmdad_qwen38_a6_fullframes/ && echo A6_UPLOADED
echo BUILD_ALL_DONE
