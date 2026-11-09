# garuda (light): no-budget ckpt-480 TF shards -> merged 591 predictions, TF seg CIDEr (591 and eval527 subset), ADQA TF CSV,
# Action/CRITIC inputs, judge/recall CSV. Same tools as E1 (e1_score.sh): merge_e1.py, eval_adqa_dense.py, to_adqa_csv.py,
# build_ac_inputs_e1.py.
#   bash e1x_post.sh
. /home/dbalu/.claude/jobs/50f46b5f/tmp/cr.env
B=s3://balu-transfer-933516006154-apne1/adlm; RAW=/storage/users/dbalu/e1x/raw/nobudget_ckpt480/tf
D=/storage/users/dbalu/preds/inference/cmdad/eval591/qwen38-dd_nobudget_ckpt480/teacher_forced; L=/storage/users/dbalu/e1x
E=/storage/users/dbalu/preds/inference/cmdad/eval527/eval_adqa_dense.py; H=$(dirname $0)
mkdir -p $RAW $D $L/adqa $L/ac
aws s3 sync --only-show-errors $B/inference/cmdad/eval591/_e1x_raw/nobudget_ckpt480/tf/ $RAW/
python3 /storage/users/dbalu/e1/bin/merge_e1.py $RAW $D/predictions.jsonl || exit 1
cd $D; python3.11 $E predictions.jsonl --tious 0.5 --json-out metrics.json > /dev/null 2>&1
head -527 predictions.jsonl > /tmp/e1x_nb527.jsonl; python3.11 $E /tmp/e1x_nb527.jsonl --tious 0.5 --json-out metrics_eval527subset.json > /dev/null 2>&1
A=/workspace/dbalu/adqa; python3 $A/to_adqa_csv.py --template $A/cmd_ad_public_template.csv --pred predictions.jsonl --mode tf --n 1000 --out $L/adqa/tf_nobudget_ckpt480.csv
python3 /storage/users/dbalu/e1/bin/build_ac_inputs_e1.py /workspace/dbalu/action_critic/stage $D/predictions.jsonl $L/ac A1_nobudget_ckpt480 > $L/ac/build.log 2>&1; tail -c 400 $L/ac/build.log
python3 $H/make_e1x_csvs.py /workspace/dbalu/action_critic/stage/baselines/autoad0_cmdad_prediction_public.csv $L/csv a1_nobudget_ckpt480=$L/ac/A1_nobudget_ckpt480__n591/action_input.csv
aws s3 sync --only-show-errors $L/csv/ $B/e1x/csv/; aws s3 sync --only-show-errors $L/ac/A1_nobudget_ckpt480__n591/ $B/e1x/ac_inputs/A1_nobudget_ckpt480__n591/
aws s3 cp --only-show-errors $D/ $B/inference/cmdad/eval591/qwen38-dd_nobudget_ckpt480/teacher_forced/ --recursive --exclude "*" --include "predictions.jsonl" --include "metrics*.json"
python3 -c "
import json
for f in ['metrics.json','metrics_eval527subset.json']:
    r=json.load(open('$D/'+f)); print(f, r['clips'], r['gt_ads'], r['pred_ads'], r['adqa_cider'], 'unique', r['adqa_unique_frac'])"
