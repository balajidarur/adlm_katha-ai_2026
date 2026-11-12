#!/bin/bash
# causaltf box runner: build causal + std records from the first N test records, then TF on 4 GPUs (gx5_infer.sh settings).
#   bash run_box.sh <commit> <tag> <ckpt_dir> <test json> <bins|nobins>
WANT=$1; TAG=$2; CKD=$3; SRC=$4; TOK=$5; R=/scratch/adlm_repo; W=/scratch/causaltf/$TAG; mkdir -p $W
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
python3 -I $R/ops/causaltf/build_causal.py $SRC $W 5 || exit 1
# causal records split over GPUs 0-2, std (5 full clips) on GPU 3; gx5_infer.sh shards by i % nshard
for g in 0 1 2; do bash $R/ops/causaltf/ct_infer.sh $WANT $CKD causal_$TAG $W/causal.json $g $g 3 $TOK > $W/causal_g$g.log 2>&1 & done
bash $R/ops/causaltf/ct_infer.sh $WANT $CKD std_$TAG $W/std.json 3 0 1 $TOK > $W/std_g3.log 2>&1 &
wait; echo "ALL_DONE $TAG $(date -u)"
