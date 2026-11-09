# q box: OFFICIAL unmodified Action + CRITIC (same commands as t2/ops/q4_metrics.sh) for runs whose inputs are in
# s3 adlm/e1x/ac_inputs/<RUN>/{action_input,critic_input}.csv  -> s3 adlm/e1x/ac_out/
#   bash e1x_ac.sh <commit> "<RUN ...>"
WANT=$1; RUNS=$2; W=/scratch; R=$W/adlm_repo; B=s3://balu-transfer-933516006154-apne1/adlm/e1x
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
. $W/cr.env; M=$W/ac/stage/metrics; OUT=$W/ac/out; mkdir -p $OUT/logs $OUT/action_detail $W/ac/preds
unset HF_HOME; export HF_HOME=$W/hf_ac HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for r in $RUNS; do
  aws s3 sync --only-show-errors $B/ac_inputs/$r/ $W/ac/preds/$r/
  ( cd $M && CUDA_VISIBLE_DEVICES=0 $W/critic_env/bin/python critic_metric.py --path $W/ac/preds/$r/critic_input.csv ) > $OUT/logs/critic_$r.log 2>&1
  ( cd $M/action_score && CUDA_VISIBLE_DEVICES=0 $W/ac_env/bin/python get_action_score.py --dataset cmdad --mode ad \
      --pred_path $W/ac/preds/$r/action_input.csv --save_path $OUT/action_detail/$r.csv ) > $OUT/logs/action_$r.log 2>&1
  echo "$r action: $(grep 'action score is' $OUT/logs/action_$r.log) | critic: $(grep 'avg IoU' $OUT/logs/critic_$r.log)" >> $OUT/summary.txt
  . $W/cr.env; aws s3 sync --only-show-errors $OUT/ $B/ac_out/
done
echo AC_DONE >> $OUT/summary.txt; aws s3 sync --only-show-errors $OUT/ $B/ac_out/
