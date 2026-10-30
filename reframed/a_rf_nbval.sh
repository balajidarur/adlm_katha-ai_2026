# Box A (2026-10-31): REFRAMED no-budget ckpt-654 on the validation split - teacher-forced (fixed trainer) then free-running.
# Raw outputs -> s3 adlm/_runs/2026-10-31/<run>/ ; scored + filed into adlm/inference/ from garuda.
sudo touch /etc/gpu-idle-shutdown.disable
mountpoint -q /scratch || { sudo mkfs.ext4 -q -F /dev/nvme3n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme3n1 /scratch && sudo chown ubuntu:ubuntu /scratch; }
mkdir -p /scratch/inf/out; mv /tmp/crA.env /scratch/creds.env 2>/dev/null; chmod 600 /scratch/creds.env; . /scratch/creds.env
aws configure set default.s3.max_concurrent_requests 64
B=s3://balu-transfer-933516006154-apne1
[ -d /scratch/rf_datasets ] || aws s3 cp --only-show-errors $B/reframed/train_data/rf_frames_datasets.tar.gz - | pigz -dc | tar -x -C /scratch
CK=/scratch/ckpt/reframed_qwen38-dd_nobudget/checkpoint-654
NB=$B/adlm/models/reframed/qwen38-dd_nobudget/checkpoint-654/
until [ "$(aws s3 ls --recursive $NB | wc -l)" -ge 15 ] && ! aws s3 ls $B/adlm/ckpts/reframed_qwen38_nobudget/ >/dev/null 2>&1; do sleep 60; . /scratch/creds.env; done   # wait for the S3 reorg move
[ -f $CK/adapter_model.safetensors ] || aws s3 cp --only-show-errors --recursive $B/adlm/models/reframed/qwen38-dd_nobudget/checkpoint-654/ $CK/ \
  --exclude "optimizer.pt" --exclude "rng_state*" --exclude "scheduler.pt"
echo "trainer md5 $(md5sum ~/adlm/LLaMA-Factory/src/llamafactory/train/sft/trainer.py | cut -c1-10) (want 955de66d8a)"
ls $CK; ls /scratch/rf_datasets/frame_variants_448x252/ | grep -i "nobudget.*validation"
. /home/ubuntu/tfgen.sh
S=$B/adlm/_runs/2026-10-31
RF=/scratch/rf_datasets/frame_variants_448x252
mk rf_nobudget654_validation_tf2  $CK 0 $RF reframed_nobudget_f150_999k_validation tf
mk rf_nobudget654_validation_free $CK 0 $RF reframed_nobudget_f150_999k_validation free:6144
run rf_nobudget654_validation_tf2
run rf_nobudget654_validation_free
echo ALL_DONE >> /scratch/inf/queue.log
