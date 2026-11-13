# REFRAMED challenge QEval / QEval-T for the CMD-AD-only model (ours-m3, raw + realistic), on one 4-GPU box,
# same setup as challenge_eval/box/qa_box.sh: reframedeval's own qa_based scorer, OLMo-3-1125-32B (base) under vLLM 0.31,
# tensor parallel 2, two lanes (GPUs 0,1 = raw; 2,3 = realistic), prompt-order patch (qa_sorted_patch.py, scores unchanged).
# Inputs: s3 reframed/challenge_eval/r2/qa_m3/qa_in_m3.tgz (patched reframedeval + preprocessed references/generations).
#   bash qa_m3.sh        -> s3 reframed/challenge_eval/r2/qa_m3/out/
set -x; exec >> $HOME/qa_m3.log 2>&1; echo "START $(date -u)"
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH; . $HOME/.cr.env; S=s3://balu-transfer-933516006154-apne1/reframed/challenge_eval/r2/qa_m3
W=/scratch; cd $W && sudo touch /etc/gpu-idle-shutdown.disable
export HF_HOME=$W/hf_cache UV_CACHE_DIR=$W/uvcache VLLM_USE_FLASHINFER_SAMPLER=0
[ -x $W/qaenv/bin/python ] || { uv venv -q -p 3.12 $W/qaenv && uv pip install -q --python $W/qaenv/bin/python vllm==0.31.0 "huggingface_hub[hf_xet]>=1.5,<2" \
  datasets numpy omegaconf pycocoevalcap pysrt tqdm wtpsplit; }
$W/qaenv/bin/python -c "import vllm; print('VLLM', vllm.__version__)" || { echo ENV_FAILED; exit 1; }
[ -f $W/olmo.done ] || { $W/qaenv/bin/hf download allenai/Olmo-3-1125-32B > $W/olmo_dl.log 2>&1 && touch $W/olmo.done; } || { echo OLMO_DL_FAILED; exit 1; }
echo "OLMO_READY $(date -u)"
mkdir -p $W/qa && aws s3 cp --only-show-errors $S/qa_in_m3.tgz - | tar xz -C $W/qa
cd $W/qa/reframedeval && export PYTHONPATH=$PWD/src
lane(){ gpus=$1; cfg=$2
  echo "QA_RUN $cfg gpus=$gpus $(date -u +%T)"
  yes yes | CUDA_VISIBLE_DEVICES=$gpus $W/qaenv/bin/python src/scorer.py --config configs/$cfg.yaml --metrics qa_based > $W/qa_$cfg.log 2>&1; rc=$?
  echo "QA_END $cfg exit=$rc $(date -u +%T)"
  (cd $W/qa && tar czf $W/qa_out_$cfg.tgz $(find rfeval_work/preprocessed_data/generations -path '*/qa/*.json'))
  . $HOME/.cr.env; aws s3 cp --only-show-errors $W/qa_out_$cfg.tgz $S/out/qa_out_$cfg.tgz && aws s3 cp --only-show-errors $W/qa_$cfg.log $S/out/qa_$cfg.log && echo "SYNCED $cfg"; }
lane 0,1 qa_m3_raw & sleep 120; lane 2,3 qa_m3_real & wait
echo "QA_ALL_DONE $(date -u)"
