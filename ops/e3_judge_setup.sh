# s4: E1X's judge setup (ops/e1x/e1x_setup.sh "judge" part) under W=/opt/dlami/nvme/e3: bert_score into ~/adlm/venv
# (--no-deps), NousResearch Llama-2-7b-chat-hf + Meta-Llama-3-8B-Instruct + roberta-large into $W/hf_judge.
W=/opt/dlami/nvme/e3; export PATH=$HOME/.local/bin:$PATH
uv pip install -q --python $HOME/adlm/venv/bin/python --no-deps "bert_score==0.3.13" || $HOME/adlm/venv/bin/pip install -q --no-deps "bert_score==0.3.13"
HF_HOME=$W/hf_judge HF_HUB_OFFLINE=0 $HOME/adlm/venv/bin/python -c "
from huggingface_hub import snapshot_download as s
for m in ['NousResearch/Llama-2-7b-chat-hf','NousResearch/Meta-Llama-3-8B-Instruct','FacebookAI/roberta-large']:
    print(s(m, allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','merges.txt','vocab.json']))
from bert_score import BERTScorer; BERTScorer(lang='en', rescale_with_baseline=True); print('BERTSCORE_OK')"
echo JUDGE_SETUP_DONE
