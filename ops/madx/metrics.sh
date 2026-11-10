# MADX: model-based MAD-Eval metrics with T3's code and settings (T3 bundle/box/metrics_box.sh + llm_s4.sh), one GPU:
#   Action  : shot-by-shot official get_action_score.py --dataset madeval (gte-Qwen2-7B), ac_env as ops/e3_ac_setup.sh
#   CRITIC  : AutoAD-III official critic_metric.py, data files extended with the 10 MAD-Eval casts (t3/critic_inputs.py)
#   R@5/16  : t3/recall_mad.py (AutoAD-II recall_within_neighbours, BERTScore roberta-large rescaled; macro over movies reported)
#   SPICE   : pycocoevalcap SPICE (t3/spice_only.py); Stanford CoreNLP 3.6.0 jars from Maven Central (as T3)
#   LLM-AD-Eval: t3/llm_ad_eval_hf.py, llama2 | llama3, --bs 32, seed 0 (T3's s4 run: official prompt; Llama-3 via chat template)
# Input CSVs (score_mad_seg.py preds_tf.csv format: anno_idx, imdbid, movie, start, end, text_gt, text_gen):
#   s3 adlm/inference/mad/metrics_in/<run>.csv
#   bash metrics.sh <commit> setup            (envs + judge models; CPU/network, can overlap a GPU job)
#   bash metrics.sh <commit> run <run> [...]  -> s3 adlm/inference/mad/metrics_out/<run>/
WANT=$1; MODE=$2; shift 2
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
R=/scratch/adlm_repo; X=$R/ops/madx/t3; B=s3://balu-transfer-933516006154-apne1; W=/scratch/mx
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
cr(){ set -a; . ~/cr.env; set +a; }
cr; mkdir -p $W; export UV_CACHE_DIR=$W/uvcache
log(){ echo "$(date -u +%T) $*" | tee -a /scratch/madx_metrics.log; }
if [ $MODE = setup ]; then
  # Action / CRITIC envs (ops/e3_ac_setup.sh incl. its 2026-11-09 pins)
  mkdir -p $W/ac && cd $W/ac && aws s3 cp --only-show-errors $B/cmd_ad_eval591/action_critic/stage.tgz . && tar xzf stage.tgz
  uv venv -q -p 3.11 $W/ac_env
  uv pip install -q --python $W/ac_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
  uv pip install -q --python $W/ac_env/bin/python "transformers==4.44.2" "sentence-transformers==3.0.1" "spacy==3.8.2" "pandas==2.2.3" "numpy==1.26.4" "huggingface_hub<1" tqdm accelerate "click<8.2" "pyarrow==17.0.0"
  uv pip install -q --python $W/ac_env/bin/python "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" || log FLASH_WHEEL_FAIL
  uv pip install -q --python $W/ac_env/bin/python "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
  uv pip install -q --python $W/ac_env/bin/python pycocoevalcap
  uv venv -q -p 3.11 $W/critic_env
  uv pip install -q --python $W/critic_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
  uv pip install -q --python $W/critic_env/bin/python "transformers==4.35.2" "fastcoref==2.1.6" "spacy==3.8.2" "pandas==2.2.3" "numpy==1.26.4" "huggingface_hub<1" "pyarrow==17.0.0" "datasets==2.21.0" "click<8.2"
  uv pip install -q --python $W/critic_env/bin/python "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
  REV=a8d08b36ada9cacfe34c4d6f80957772a025daf2
  HF_HOME=$W/hf_ac $W/ac_env/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Alibaba-NLP/gte-Qwen2-7B-instruct', revision='$REV'))"
  d=$W/hf_ac/hub/models--Alibaba-NLP--gte-Qwen2-7B-instruct; mkdir -p $d/refs; echo -n $REV > $d/refs/main
  HF_HOME=$W/hf_ac $W/critic_env/bin/python -c "from fastcoref import FCoref; m=FCoref(device='cpu'); print('FCOREF_OK')"
  HF_HOME=$W/hf_ac $W/ac_env/bin/python -c "import flash_attn, sentence_transformers, spacy; spacy.load('en_core_web_lg'); print('AC_OK')"
  # SPICE: java + CoreNLP 3.6.0 jars into pycocoevalcap/spice/lib (stanford.edu is unreachable; Maven Central has the same jars)
  command -v java > /dev/null || sudo apt-get install -y -q default-jre-headless > $W/java.log 2>&1
  L=$($W/ac_env/bin/python -c "import pycocoevalcap.spice as s, os; print(os.path.dirname(s.__file__))")/lib; mkdir -p $L
  for j in stanford-corenlp-3.6.0.jar stanford-corenlp-3.6.0-models.jar; do
    [ -s $L/$j ] || curl -sSfL -o $L/$j https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/$j; done
  ls -la $L | grep corenlp
  # judges: ~/adlm/venv (transformers 5.8, as T3's s4 LLM-AD-Eval env) + bert_score; Llama-2/3 + roberta-large
  uv pip install -q --python $HOME/adlm/venv/bin/python --no-deps "bert_score==0.3.13"
  HF_HOME=$W/hf_judge $HOME/adlm/venv/bin/python -c "
from huggingface_hub import snapshot_download as s
for m in ['NousResearch/Llama-2-7b-chat-hf','NousResearch/Meta-Llama-3-8B-Instruct','FacebookAI/roberta-large','roberta-large']:
    print(s(m, allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','merges.txt','vocab.json']))
from bert_score import BERTScorer; BERTScorer(lang='en', rescale_with_baseline=True); print('BERTSCORE_OK')"
  log METRICS_SETUP_DONE; exit 0
fi
# run: one GPU; Action alone (gte-Qwen2-7B fp32), then CRITIC + R@5/16 + LLM llama2 + llama3 + SPICE together
for RUN in "$@"; do
  O=$W/out/$RUN; mkdir -p $O; cr
  aws s3 cp --only-show-errors $B/adlm/inference/mad/metrics_in/$RUN.csv $O/in.csv || { log "NO_INPUT $RUN"; continue; }
  log "RUN $RUN rows=$(($(wc -l < $O/in.csv) - 1))"
  rm -rf $O/critic && cp -r $W/ac/stage/metrics $O/critic
  python3 $X/critic_inputs.py --metrics $O/critic --roles $X/madeval_charbank_roles.json --anno $X/madeval_anno.csv --preds $O/in.csv \
    --out $O/critic_in.csv > $O/critic_in.log 2>&1
  ( cd $W/ac/stage/metrics/action_score && HF_HOME=$W/hf_ac HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
      $W/ac_env/bin/python get_action_score.py --dataset madeval --mode ad --pred_path $O/in.csv --save_path $O/action.csv ) > $O/action.log 2>&1
  log "ACTION $RUN $(grep -i 'action score' $O/action.log | tail -1)"
  ( cd $O/critic && HF_HOME=$W/hf_ac HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 $W/critic_env/bin/python critic_metric.py --path $O/critic_in.csv ) > $O/critic.log 2>&1 &
  ( HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 $HOME/adlm/venv/bin/python $X/recall_mad.py --csv $O/in.csv --out $O/recall.json ) > $O/recall.log 2>&1 &
  for m in llama2 llama3; do
    ( HF_HOME=$W/hf_judge HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 $HOME/adlm/venv/bin/python $X/llm_ad_eval_hf.py \
        --csv $O/in.csv --model $m --bs 32 --out $O/llm_$m.json ) > $O/llm_$m.log 2>&1 &
  done
  ( $W/ac_env/bin/python $X/spice_only.py $O/in.csv $O/spice.json ) > $O/spice.log 2>&1 &
  wait
  log "DONE $RUN critic: $(grep -i 'avg IoU' $O/critic.log | tail -1) | recall: $(python3 -c "import json;d=json.load(open('$O/recall.json'));print(d['macro'], d['R@5/16'])" 2>&1 | tail -1) | llm2 $(tail -1 $O/llm_llama2.log) | llm3 $(tail -1 $O/llm_llama3.log) | $(tail -1 $O/spice.log)"
  cr; aws s3 sync --only-show-errors $O $B/adlm/inference/mad/metrics_out/$RUN/ --exclude "critic/*"
done
log METRICS_ALL_DONE
