# Box A: build the 0.2 s bins datasets from commit 7fe3455 (git bundle), CPU only, -> /prep/bins02 -> S3
set -e
. /scratch/creds.env
B=s3://balu-transfer-933516006154-apne1/adlm
aws s3 cp --only-show-errors $B/code/adlm_7fe3455.bundle /prep/adlm_7fe3455.bundle
rm -rf /prep/adlm_repo && git clone -q -b qwen38-dense-dataset-training /prep/adlm_7fe3455.bundle /prep/adlm_repo
cd /prep/adlm_repo && echo "COMMIT $(git rev-parse --short HEAD) (want 7fe3455)"; [ "$(git rev-parse --short HEAD)" = 7fe3455 ]
~/adlm/venv/bin/python -u dataset/add_bins02.py --q /data/cmd_ad/datasets/frame_variants_448x252_full_qwen38 --out /prep/bins02
ls -la /prep/bins02
aws s3 cp --only-show-errors --recursive /prep/bins02/ $B/datasets/cmdad_qwen38_bins02/ && echo UPLOADED
echo BUILD_DONE
