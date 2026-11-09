# s4: official Action (shot-by-shot) + CRITIC (AutoAD-II) envs (copy of t2/ops/q4_metrics_setup.sh, W=/opt/dlami/nvme/e3), per /workspace/dbalu/action_critic/METHOD.md
set -x
W=/opt/dlami/nvme/e3; . $W/cr.env; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$W/uvcache HF_HOME=$W/hf_ac
mkdir -p $W/ac && cd $W/ac && aws s3 cp --only-show-errors s3://balu-transfer-933516006154-apne1/cmd_ad_eval591/action_critic/stage.tgz . && tar xzf stage.tgz
uv venv -q -p 3.11 $W/ac_env
uv pip install -q --python $W/ac_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python $W/ac_env/bin/python "transformers==4.44.2" "sentence-transformers==3.0.1" "spacy==3.8.2" "pandas==2.2.3" "numpy==1.26.4" "huggingface_hub<1" tqdm accelerate
uv pip install -q --python $W/ac_env/bin/python "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" || echo FLASH_WHEEL_FAIL
uv pip install -q --python $W/ac_env/bin/python "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
uv venv -q -p 3.11 $W/critic_env
uv pip install -q --python $W/critic_env/bin/python "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python $W/critic_env/bin/python "transformers==4.35.2" "fastcoref==2.1.6" "spacy==3.8.2" "pandas==2.2.3" "numpy==1.26.4" "huggingface_hub<1"
uv pip install -q --python $W/critic_env/bin/python "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
REV=a8d08b36ada9cacfe34c4d6f80957772a025daf2
$W/ac_env/bin/python -c "
from huggingface_hub import snapshot_download; p=snapshot_download('Alibaba-NLP/gte-Qwen2-7B-instruct', revision='$REV'); print(p)"
d=$HF_HOME/hub/models--Alibaba-NLP--gte-Qwen2-7B-instruct; mkdir -p $d/refs; echo -n $REV > $d/refs/main
$W/critic_env/bin/python -c "from fastcoref import FCoref; m=FCoref(device='cpu'); print('FCOREF_OK')"
$W/ac_env/bin/python -c "import flash_attn, sentence_transformers, spacy; spacy.load('en_core_web_lg'); print('AC_OK')"
echo METRICS_SETUP_DONE
# 2026-11-09: unpinned deps drifted since T2's setup (pyarrow now needs NumPy 2; spaCy CLI missing click). Pin them back.
uv pip install -q --python $W/critic_env/bin/python "pyarrow==17.0.0" "datasets==2.21.0" "click<8.2"
uv pip install -q --python $W/ac_env/bin/python "click<8.2" "pyarrow==17.0.0"
$W/critic_env/bin/python -c "from fastcoref import FCoref; m=FCoref(device='cpu'); print('FCOREF_OK')"
$W/ac_env/bin/python -c "import flash_attn, sentence_transformers, spacy; spacy.load('en_core_web_lg'); print('AC_OK')"
echo METRICS_SETUP_FIXED
