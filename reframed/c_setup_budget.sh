# box C: instance NVMe -> /scratch,/scratch2 (never /data); back up C's key checkpoints to S3;
# pull REFRAMED videos + GT ADs, Qwen3.8 DDs, dialogue SRTs, train AD word times, CMD-AD build code
set -x
export PATH=$HOME/.local/bin:$PATH
sudo touch /etc/gpu-idle-shutdown.disable
for p in "nvme2n1 /scratch" "nvme3n1 /scratch2"; do set -- $p
  mountpoint -q $2 || { sudo mkfs.ext4 -q -F /dev/$1 && sudo mkdir -p $2 && sudo mount /dev/$1 $2 && sudo chown ubuntu:ubuntu $2; }; done
[ -f /tmp/credsC.env ] && mv /tmp/credsC.env /scratch2/creds.env; chmod 600 /scratch2/creds.env; . /scratch2/creds.env
aws configure set default.s3.max_concurrent_requests 64; aws configure set default.s3.multipart_chunksize 64MB
B=s3://balu-transfer-933516006154-apne1
# 1) key checkpoints from box C (full dirs, resumable)
( up(){ aws s3 cp --recursive --only-show-errors /data/cmd_ad/runs/$1/checkpoint-$2 $B/adlm/ckpts/$3/checkpoint-$2/ && echo "CKPT_OK $3/$2"; }
  up nobudget_frames_full 480 cmdad_gemini_nobudget; up nobudget_frames_full 600 cmdad_gemini_nobudget; up nobudget_frames_full 696 cmdad_gemini_nobudget
  up qwen38_nobudget_frames_full 480 cmdad_qwen38_nobudget; up qwen38_nobudget_frames_full 520 cmdad_qwen38_nobudget
  for r in nobudget_frames_full qwen38_nobudget_frames_full; do aws s3 cp --only-show-errors /data/cmd_ad/runs/$r/trainer_log.jsonl $B/adlm/ckpts/${r}_trainer_log.jsonl; done
  echo CKPTS_DONE ) > /scratch2/ckpt_upload.log 2>&1 &
# 2) inputs
mkdir -p /scratch/reframed /scratch2/in && cd /scratch2/in
aws s3 cp --only-show-errors $B/reframed/dd_qwen38_scenes.tar.gz . && tar xzf dd_qwen38_scenes.tar.gz -C /scratch2
aws s3 cp --only-show-errors $B/reframed/inputs/dialogue_srt_all_splits.tgz . && mkdir -p /scratch2/rf_srt && tar xzf dialogue_srt_all_splits.tgz -C /scratch2/rf_srt
aws s3 cp --only-show-errors $B/reframed/inputs/reframed_train_srt_adtimes.tgz . && mkdir -p /scratch2/rf_train && tar xzf reframed_train_srt_adtimes.tgz -C /scratch2/rf_train
aws s3 cp --only-show-errors $B/adlm/code/adlm_code_boxA_20261027.tgz . && mkdir -p /scratch2/codeA && tar xzf adlm_code_boxA_20261027.tgz -C /scratch2/codeA
for s in test validation train; do
  aws s3 cp --only-show-errors $B/reframed/reframed-$s.tar.gz - | pigz -dc | tar -x -C /scratch/reframed --wildcards '*excerpt/video/*' '*excerpt/audio_description_us/*' '*/meta.json'
  echo "$(date +%T) $s videos: $(find /scratch/reframed/$s -name '*.mp4' | wc -l)"
done
echo C_SETUP_DONE
