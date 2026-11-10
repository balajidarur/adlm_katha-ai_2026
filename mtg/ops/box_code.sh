#!/bin/bash
# usage: box_code.sh HASH  -> /scratch/code_HASH from the S3 bundle (asserts the hash); /scratch/code symlinks to it
. $HOME/.cr.env; export AWS_DEFAULT_REGION=ap-northeast-1
H=$1
aws s3 cp --only-show-errors s3://balu-transfer-933516006154-apne1/mtg/code/adlm_$H.bundle /scratch/adlm_$H.bundle
rm -rf /scratch/code_$H && git clone -q -b qwen38-dense-dataset-training /scratch/adlm_$H.bundle /scratch/code_$H
GOT=$(git -C /scratch/code_$H rev-parse --short HEAD)
echo "COMMIT $GOT"
[ "$GOT" = "$H" ] || { echo "COMMIT MISMATCH want $H"; exit 1; }
[ -d /scratch/code ] && [ ! -L /scratch/code ] && mv /scratch/code /scratch/code_old_$(date +%s)
ln -sfn /scratch/code_$H /scratch/code
