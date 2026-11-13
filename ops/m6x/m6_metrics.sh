# Box A after m6x_box_a.sh: M6-591 teacher-forced R@1/5 (ops/e1x/recall_cmdad.py, per movie, BERTScore roberta-large) and the
# official CRITIC (AutoAD-II critic_metric.py, fastcoref env exactly as ops/e3_ac_setup.sh). Inputs from garuda (merge +
# build_ac_inputs_591 + make_e1x_csvs): s3 adlm/xeval/m6/{m6_ckpt440.csv, critic_input.csv}. Outputs -> s3 adlm/xeval/m6/out/.
#   bash m6_metrics.sh <commit>
WANT=$1; set -x; exec >> $HOME/m6m.log 2>&1
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH; . $HOME/.cr.env; B=s3://balu-transfer-933516006154-apne1; X=$B/adlm/xeval/m6
W=/scratch/m6m; R=/scratch/adlm_repo; mkdir -p $W/out && cd $W
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
aws s3 cp --only-show-errors $X/m6_ckpt440.csv . && aws s3 cp --only-show-errors $X/critic_input.csv .
export UV_CACHE_DIR=$W/uvcache HF_HOME=$W/hf
uv venv -q -p 3.11 $W/bs_env && uv pip install -q --python $W/bs_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121 \
  && uv pip install -q --python $W/bs_env/bin/python "bert-score==0.3.13" "transformers==4.44.2" "pandas==2.2.3" "numpy==1.26.4"
( CUDA_VISIBLE_DEVICES=0 $W/bs_env/bin/python $R/ops/e1x/recall_cmdad.py --csv m6_ckpt440.csv --out out/recall_m6_ckpt440.json > out/recall.log 2>&1; echo "RECALL_EXIT $?" ) &
aws s3 cp --only-show-errors $B/cmd_ad_eval591/action_critic/stage.tgz . && tar xzf stage.tgz
uv venv -q -p 3.11 $W/critic_env
uv pip install -q --python $W/critic_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python $W/critic_env/bin/python "transformers==4.35.2" "fastcoref==2.1.6" "spacy==3.8.2" "pandas==2.2.3" "numpy==1.26.4" "huggingface_hub<1" "pyarrow==17.0.0" "datasets==2.21.0" "click<8.2"
uv pip install -q --python $W/critic_env/bin/python "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
M=$(dirname $(find $W -name critic_metric.py | head -1))
( cd $M && CUDA_VISIBLE_DEVICES=1 $W/critic_env/bin/python critic_metric.py --path $W/critic_input.csv ) > out/critic.log 2>&1; echo "CRITIC_EXIT $? $(grep 'avg IoU' out/critic.log)"
wait; . $HOME/.cr.env; aws s3 sync --only-show-errors out $X/out/ && echo M6M_SYNCED
