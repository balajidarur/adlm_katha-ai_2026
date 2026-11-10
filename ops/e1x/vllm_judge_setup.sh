# q-box setup for llm_ad_eval_vllm.py (1xL40S): /scratch, commit-checked clone from the S3 bundle, vLLM venv,
# NousResearch judge weights (S3 cache adlm/e1x/judges/ if present, else HF), system CSVs.
#   bash vllm_judge_setup.sh <commit>
WANT=$1; W=/scratch; B=s3://balu-transfer-933516006154-apne1; VV=${VLLM_VERSION:-0.28.0}
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
mkdir -p $W/e1x/csv; aws s3 sync --only-show-errors $B/adlm/e1x/csv/ $W/e1x/csv/
[ -x $W/vllm/bin/vllm ] || { uv venv -q -p 3.12 $W/vllm && uv pip install -q --python $W/vllm/bin/python vllm==$VV pandas requests; } || exit 1
$W/vllm/bin/python -c "import vllm; print('VLLM', vllm.__version__)"
mkdir -p $W/judges
for m in Llama-2-7b-chat-hf Meta-Llama-3-8B-Instruct; do
  aws s3 sync --only-show-errors $B/adlm/e1x/judges/$m/ $W/judges/$m/
  if ! ls $W/judges/$m/*.safetensors >/dev/null 2>&1; then
    HF_HUB_ENABLE_HF_TRANSFER=0 $W/vllm/bin/python -c "
from huggingface_hub import snapshot_download as s
print(s('NousResearch/$m', local_dir='$W/judges/$m', allow_patterns=['*.json','*.safetensors','tokenizer*','*.model']))"
    aws s3 sync --only-show-errors $W/judges/$m/ $B/adlm/e1x/judges/$m/ --exclude ".cache/*"
  fi
done
ls -la $W/judges/*/
# first-50-clip subset CSV (CSV order) for the HF re-runs used as the noise estimate
$W/vllm/bin/python -c "
import pandas as pd; d=pd.read_csv('$W/e1x/csv/m3_ckpt440.csv', keep_default_na=False); k=list(dict.fromkeys(d.cmd_filename))[:50]
d[d.cmd_filename.isin(k)].to_csv('$W/e1x/m3_ckpt440_first50.csv', index=False); print('FIRST50', d.cmd_filename.isin(k).sum())"
echo VLLM_SETUP_DONE
