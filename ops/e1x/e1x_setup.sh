# E1X q-box setup (1xL40S): /scratch, commit-checked clone, no-budget ckpt-480 adapter, eval591_nobudget + frames for all
# 591 clips (per-source as E1: 16 HF clips and 7 T2 rebuilds from cmd_ad_eval591/), judge models (NousResearch Llama-2-7b-chat,
# Meta-Llama-3-8B-Instruct; roberta-large for BERTScore), bert_score into ~/adlm/venv (--no-deps).
#   bash e1x_setup.sh <commit> [judge] [ac]
WANT=$1; W=/scratch; B=s3://balu-transfer-933516006154-apne1
sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q $W; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p $W && sudo mount /dev/$DEV $W && sudo chown ubuntu:ubuntu $W
fi
cp ~/cr.env $W/cr.env; . $W/cr.env; export PATH=$HOME/.local/bin:$PATH
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle $W/adlm.bundle
R=$W/adlm_repo; rm -rf $R && git clone -q -b qwen38-dense-dataset-training $W/adlm.bundle $R
HAVE=$(git -C $R rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
echo "TRAINER_MD5 $(md5sum $R/training/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10)"
E=$R/ops/e1x
aws s3 cp --only-show-errors --recursive $B/adlm/models/cmdad/qwen38-dd_nobudget/checkpoint-480/ $W/ckpt/qwen38-dd_nobudget_ckpt480/ \
  --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt" &
mkdir -p $W/e1x; aws s3 sync --only-show-errors $B/adlm/datasets/e1x_eval591_nobudget/ $W/e1x/ds/
( cd $W/e1x/ds && sha256sum -c SHA256SUMS ) || { echo SHA_MISMATCH; exit 1; }
cat $E/vids591.txt | xargs -P 32 -I{} aws s3 sync --only-show-errors $B/adlm/frames/1fps_448x252/{}/ $W/frames/1fps_448x252/{}/
F=$W/frames/1fps_448x252
for v in $(cat $E/vids_hf.txt); do rm -rf $F/$v; aws s3 sync --only-show-errors $B/cmd_ad_eval591/frames_hf_1fps_448x252/$v/ $F/$v/; done
for v in $(cat $E/vids_t2.txt); do rm -rf $F/$v; aws s3 sync --only-show-errors $B/cmd_ad_eval591/frames_1fps_448x252/$v/ $F/$v/; done
sudo mkdir -p "$(readlink -m /data/cmd_ad)/frames"; [ -e /data/cmd_ad/frames/1fps_448x252 ] && [ ! -L /data/cmd_ad/frames/1fps_448x252 ] && echo "NOTE real /data frames dir exists"
[ -L /data/cmd_ad/frames/1fps_448x252 ] && sudo rm /data/cmd_ad/frames/1fps_448x252
[ -e /data/cmd_ad/frames/1fps_448x252 ] || sudo ln -sfn $F /data/cmd_ad/frames/1fps_448x252
python3 -c "
import json,os; d=json.load(open('$W/e1x/ds/eval591_nobudget.json')); m=sum(not os.path.exists(p) for r in d for p in r['images']); print('EVAL591_NB',len(d),'missing',m); assert m==0" || { echo FRAMES_MISSING; exit 1; }
wait; ls -la $W/ckpt/qwen38-dd_nobudget_ckpt480/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
if [ "$2" = judge ] || [ "$3" = judge ]; then
  uv pip install -q --python $HOME/adlm/venv/bin/python --no-deps "bert_score==0.3.13" || $HOME/adlm/venv/bin/pip install -q --no-deps "bert_score==0.3.13"
  HF_HOME=$W/hf_judge HF_HUB_OFFLINE=0 $HOME/adlm/venv/bin/python -c "
from huggingface_hub import snapshot_download as s
for m in ['NousResearch/Llama-2-7b-chat-hf','NousResearch/Meta-Llama-3-8B-Instruct','FacebookAI/roberta-large']:
    print(s(m, allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt','merges.txt','vocab.json']))
from bert_score import BERTScorer; BERTScorer(lang='en', rescale_with_baseline=True); print('BERTSCORE_OK')"
fi
if [ "$2" = ac ] || [ "$3" = ac ]; then bash $E/e1x_ac_setup.sh; fi
echo E1X_SETUP_DONE
