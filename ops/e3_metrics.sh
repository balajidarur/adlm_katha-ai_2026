# s4, after the free run: TF metrics of an E3 checkpoint with the official / E1X-calibrated tools, one GPU each in parallel:
#   GPU0 official CRITIC then Action (t2/q4_metrics.sh commands), GPU1 LLM-AD-Eval l2 (official AutoAD-III prompt),
#   GPU2 LLM-AD-Eval l3_zero_lenient (AutoAD-Zero Alg. 3 + "Be generous..."), GPU3 R@k/N (recall_cmdad.py, per movie).
# Inputs (from e3_post.py on garuda): s3 adlm/e3/ac_inputs/<RUN>/{action_input,critic_input}.csv, adlm/e3/csv/<RUN>.csv
#   bash e3_metrics.sh <commit> <RUN>          outputs -> s3 adlm/e3/{ac_out,metrics}/
WANT=$1; RUN=$2; W=/opt/dlami/nvme/e3; R=$W/adlm_repo_eval; E=$R/ops/e1x; B=s3://balu-transfer-933516006154-apne1/adlm/e3
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
. $W/cr.env; M=$W/ac/stage/metrics; O=$W/e3m; mkdir -p $O/ac/logs $O/ac/action_detail $O/in $O/llm $O/recall
aws s3 sync --only-show-errors $B/ac_inputs/$RUN/ $O/in/$RUN/; aws s3 cp --only-show-errors $B/csv/$RUN.csv $O/in/$RUN.csv
ls $O/in/$RUN/action_input.csv $O/in/$RUN/critic_input.csv $O/in/$RUN.csv || { echo INPUTS_MISSING; exit 1; }
PY=$HOME/adlm/venv/bin/python
( export HF_HOME=$W/hf_ac HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  ( cd $M && CUDA_VISIBLE_DEVICES=0 $W/critic_env/bin/python critic_metric.py --path $O/in/$RUN/critic_input.csv ) > $O/ac/logs/critic_$RUN.log 2>&1
  ( cd $M/action_score && CUDA_VISIBLE_DEVICES=0 $W/ac_env/bin/python get_action_score.py --dataset cmdad --mode ad \
      --pred_path $O/in/$RUN/action_input.csv --save_path $O/ac/action_detail/$RUN.csv ) > $O/ac/logs/action_$RUN.log 2>&1
  echo "$RUN action: $(grep 'action score is' $O/ac/logs/action_$RUN.log) | critic: $(grep 'avg IoU' $O/ac/logs/critic_$RUN.log)" >> $O/ac/summary.txt ) &
( export HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=1 $PY $E/llm_ad_eval_hf.py --csv $O/in/$RUN.csv --variant l2 --out_dir $O/llm > $O/llm_l2.log 2>&1 ) &
( export HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=2 $PY $E/llm_ad_eval_hf.py --csv $O/in/$RUN.csv --variant l3_zero_lenient --out_dir $O/llm > $O/llm_l3.log 2>&1 ) &
( export HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  CUDA_VISIBLE_DEVICES=3 $PY $E/recall_cmdad.py --csv $O/in/$RUN.csv --out $O/recall/$RUN.json > $O/recall.log 2>&1 ) &
wait
. $W/cr.env; aws s3 sync --only-show-errors $O/ac/ $B/ac_out/; aws s3 sync --only-show-errors $O/llm/ $B/metrics/llm/
aws s3 sync --only-show-errors $O/recall/ $B/metrics/recall/; for f in $O/*.log; do aws s3 cp --only-show-errors $f $B/metrics/logs/; done
cat $O/ac/summary.txt; echo "E3_METRICS_DONE $RUN"
