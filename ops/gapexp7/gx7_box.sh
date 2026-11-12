#!/bin/bash
# gapexp7 box (box C): REFRAMED test, AD-length-dependent budget change (halve above / double below the median predicted AD TTS), teacher-forced, budget and no-budget ckpt-432.
# Repo at <commit>, env/model if missing, REFRAMED frames + test records (rf_frames_datasets.tar.gz -> /scratch), build the
# variant records for the first N test clips, then two runs side by side: budget on GPUs 0-1, no-budget on GPUs 2-3.
#   bash gx7_box.sh <commit>
WANT=$1; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
exec > $HOME/gx7.log 2>&1; echo "START $(date -u)"; . $HOME/.cr.env; sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dn -o NAME,MODEL | awk '/Instance Storage/{print $1}' | while read d; do lsblk -n -o MOUNTPOINT /dev/$d | grep -q . || echo $d; done | head -1)
  sudo mkfs.ext4 -q -F /dev/$DEV && sudo mkdir -p /scratch && sudo mount /dev/$DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
cd /scratch; [ "$(git -C adlm_repo rev-parse --short HEAD 2>/dev/null)" = "$WANT" ] || { aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle . && rm -rf adlm_repo && git clone -q -b qwen38-dense-dataset-training adlm_$WANT.bundle adlm_repo; }
HAVE=$(git -C adlm_repo rev-parse --short HEAD); echo "COMMIT $HAVE (want $WANT)"; [ "$HAVE" = "$WANT" ] || { echo COMMIT_MISMATCH; exit 1; }
if [ ! -x $HOME/adlm/venv/bin/python ] || ! $HOME/adlm/venv/bin/python -c "import liger_kernel" 2>/dev/null || [ ! -d $HOME/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B ]; then
  mkdir -p pack; aws s3 cp --only-show-errors --recursive $B/adlm/tf_rerun/ pack/ --exclude "*" --include "uvpython.tgz" --include "venv.tgz" --include "qwen35_4b.tgz"
  for f in uvpython venv qwen35_4b; do pigz -dc pack/$f.tgz | sudo tar -x -C / ; done; rm -rf pack; sudo chown -R ubuntu:ubuntu $HOME/adlm
fi
[ -f /scratch/rf_datasets/frame_variants_448x252/reframed_bins_f150_999k_test.json ] || \
  aws s3 cp --only-show-errors $B/reframed/train_data/rf_frames_datasets.tar.gz - | pigz -dc | tar -x -C /scratch
D=/scratch/rf_datasets/frame_variants_448x252; mkdir -p /scratch/gx7
aws s3 cp --only-show-errors $B/gapexp7/in/pred_tts_x1.json /scratch/gx7/
python3 adlm_repo/ops/gapexp7/build_gapexp7.py $D/reframed_bins_f150_999k_test.json /scratch/gx7/pred_tts_x1.json budget /scratch/gx7/gx7_budget.json /scratch/gx7/manifest_budget.json
python3 adlm_repo/ops/gapexp7/build_gapexp7.py $D/reframed_nobudget_f150_999k_test.json /scratch/gx7/pred_tts_x1.json nobudget /scratch/gx7/gx7_nobudget.json /scratch/gx7/manifest_nobudget.json
python3 -c "import json,os; d=json.load(open('/scratch/gx7/gx7_budget.json')); print('records', len(d), 'missing frames', sum(not os.path.exists(p) for r in d for p in r['images']))"
aws s3 sync --only-show-errors /scratch/gx7 $B/gapexp7/in/
for t in budget nobudget; do aws s3 cp --only-show-errors --recursive $B/adlm/models/reframed/qwen38-dd_$t/checkpoint-432/ /scratch/ckpt/rf_$t/ --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"; done
ls /scratch/ckpt/rf_budget/adapter_model.safetensors /scratch/ckpt/rf_nobudget/adapter_model.safetensors || { echo CKPT_MISSING; exit 1; }
I=/scratch/adlm_repo/ops/gapexp5/gx5_infer.sh
for g in 0 1; do bash $I $WANT /scratch/ckpt/rf_budget gx7_rfbudget432 /scratch/gx7/gx7_budget.json $g $g 2 bins > $HOME/gx7_b_g$g.log 2>&1 & done
for g in 2 3; do bash $I $WANT /scratch/ckpt/rf_nobudget gx7_rfnobudget432 /scratch/gx7/gx7_nobudget.json $g $((g - 2)) 2 nobins > $HOME/gx7_n_g$g.log 2>&1 & done
wait; echo "ALL_DONE $(date -u)"
