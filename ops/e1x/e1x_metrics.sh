# E1X metrics on a q box (1 GPU): LLM-AD-Eval variants and/or R@k/N for the system CSVs in s3 adlm/e1x/csv/.
#   bash e1x_metrics.sh <commit> "<variants or 'recall'>" "<csv names without .csv>"
#   outputs -> s3 adlm/e1x/metrics/{llm,recall}/
WANT=$1; JOBS=$2; SYS=$3; W=/scratch; R=$W/adlm_repo; E=$R/ops/e1x; B=s3://balu-transfer-933516006154-apne1/adlm/e1x
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
. $W/cr.env; mkdir -p $W/e1x/csv $W/e1x/out/llm $W/e1x/out/recall
aws s3 sync --only-show-errors $B/csv/ $W/e1x/csv/
export HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PY=$HOME/adlm/venv/bin/python
CSVS=$(for s in $SYS; do echo $W/e1x/csv/$s.csv; done)
for j in $JOBS; do
  echo "$(date -u +%T) START $j [$SYS]" >> $W/e1x/metrics.log
  if [ $j = recall ]; then
    for c in $CSVS; do o=$W/e1x/out/recall/$(basename $c .csv).json; [ -f $o ] || $PY $E/recall_cmdad.py --csv $c --out $o >> $W/e1x/metrics.log 2>&1; done
  else
    CUDA_VISIBLE_DEVICES=0 $PY $E/llm_ad_eval_hf.py --csv $CSVS --variant $j --out_dir $W/e1x/out/llm >> $W/e1x/metrics.log 2>&1
  fi
  echo "$(date -u +%T) END $j rc=$?" >> $W/e1x/metrics.log
  . $W/cr.env; aws s3 sync --only-show-errors $W/e1x/out/ $B/metrics/; aws s3 cp --only-show-errors $W/e1x/metrics.log $B/metrics/logs/metrics_$(hostname).log
done
echo "$(date -u +%T) ALL_DONE [$JOBS] [$SYS]" >> $W/e1x/metrics.log
. $W/cr.env; aws s3 cp --only-show-errors $W/e1x/metrics.log $B/metrics/logs/metrics_$(hostname).log
