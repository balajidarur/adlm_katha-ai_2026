#!/bin/bash
# extra.sh RUN... : (1) LLM-AD-Eval Llama-3 with AutoAD-Zero lenient line (ops/e1x l3_zero_lenient, the E1X calibrated protocol)
#                   (2) R@k/16 in ac_env (transformers 4.44 + bert-score, = T3's recall env)
# pushed code only (commit-checked clone); outputs -> s3 adlm/inference/mad/metrics_out/<run>/
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
R=/scratch/adlm_repo; B=s3://balu-transfer-933516006154-apne1; W=/scratch/mx
[ "$(git -C $R rev-parse --short HEAD)" = a2b8d86 ] || { echo COMMIT_MISMATCH; exit 1; }
set -a; . ~/cr.env; set +a
log(){ echo "$(date -u +%T) $*" | tee -a /scratch/madx_extra.log; }
$W/ac_env/bin/python -c "import bert_score" 2>/dev/null || UV_CACHE_DIR=$W/uvcache uv pip install -q --python $W/ac_env/bin/python "bert-score==0.3.13" >> /scratch/madx_extra.log 2>&1
log "ENV ac_env transformers=$($W/ac_env/bin/python -c 'import transformers;print(transformers.__version__)') bert_score=$($W/ac_env/bin/python -c 'import bert_score;print(bert_score.__version__)')"
for RUN in "$@"; do
  O=$W/out/$RUN; mkdir -p $O
  [ -f $O/in.csv ] || aws s3 cp --only-show-errors $B/adlm/inference/mad/metrics_in/$RUN.csv $O/in.csv
  ( HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 $HOME/adlm/venv/bin/python $R/ops/e1x/llm_ad_eval_hf.py \
      --csv $O/in.csv --variant l3_zero_lenient --bs ${BS:-32} --out_dir $O/l3len ) > $O/l3len.log 2>&1
  ( HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 $W/ac_env/bin/python $R/ops/madx/t3/recall_mad.py \
      --csv $O/in.csv --out $O/recall_acenv.json ) > $O/recall_acenv.log 2>&1
  log "EXTRA $RUN l3len $(tail -1 $O/l3len.log) | recall_acenv $(python3 -c "import json;d=json.load(open('$O/recall_acenv.json'));print('macro',d['macro'],'micro',d['R@5/16'],d['R@1/16'])" 2>&1 | tail -1)"
  set -a; . ~/cr.env; set +a; aws s3 sync --only-show-errors $O $B/adlm/inference/mad/metrics_out/$RUN/ --exclude "critic/*"
done
log EXTRA_DONE
