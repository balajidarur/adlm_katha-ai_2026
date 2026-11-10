#!/bin/bash
# T5 box setup (4xL40S): /scratch on a free instance-store nvme, repo at $WANT from the S3 bundle, MTG data, then the
# writer datasets (frames 1 FPS 448x252, cap 250, token budget $TB) + budget bins.   bash a_setup.sh <commit> [TB]
WANT=$1; TB=${2:-56000}; B=s3://balu-transfer-933516006154-apne1; export AWS_DEFAULT_REGION=ap-northeast-1
exec > $HOME/t5_setup.log 2>&1; echo "START $(date -u)"; . $HOME/.cr.env
sudo touch /etc/gpu-idle-shutdown.disable
if ! mountpoint -q /scratch; then
  DEV=$(lsblk -dpbno NAME,SIZE,TYPE | awk '$3=="disk"' | sort -k2 -n -r | while read d sz t; do
          [ -z "$(lsblk -no MOUNTPOINT $d | tr -d '[:space:]')" ] && [ $(lsblk -no NAME $d | wc -l) = 1 ] && { echo $d; break; }; done)
  echo "scratch dev $DEV"; sudo mkfs.ext4 -q -F $DEV && sudo mkdir -p /scratch && sudo mount $DEV /scratch && sudo chown ubuntu:ubuntu /scratch
fi
M=/scratch/mtg; mkdir -p $M/dd
aws s3 cp --only-show-errors $B/adlm/code/adlm_$WANT.bundle /scratch/ && rm -rf /scratch/adlm_repo && git clone -q /scratch/adlm_$WANT.bundle /scratch/adlm_repo \
  && git -C /scratch/adlm_repo checkout -q $WANT; echo "COMMIT $(git -C /scratch/adlm_repo rev-parse --short HEAD)"
aws s3 cp --only-show-errors $B/mtg/ft/split.json $M/split.json; aws s3 cp --only-show-errors $B/mtg/ft/gt_tts.json $M/gt_tts.json
(aws s3 cp --recursive --only-show-errors $B/mtg/hf_mind-the-gap-inline-ad/ $M/hf/ && echo HF_DONE) &
(aws s3 cp --recursive --only-show-errors $B/mtg/videos_480p/ $M/videos_480p/ && echo VIDEOS_DONE) &
(aws s3 cp --recursive --only-show-errors $B/mtg/dd_full_nocast_mtg_b/ $M/dd/ --exclude "*" --include "*/adlm_descriptions.json" && echo DD_DONE) &
wait; ls $M/hf/data | head; echo "videos $(ls $M/videos_480p/*/*.mp4 | wc -l) dds $(ls $M/dd | wc -l)"
export PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache HF_HUB_OFFLINE=1 MTG=$M
cd /scratch/adlm_repo/mtg/ft && python mtg_build.py --out $M/ds --cap 250 --token-budget $TB --workers 44 --frame-root $M/frames \
  && python mtg_bins.py --dir $M/ds --max-frames 250 --budget $TB && echo BUILD_DONE
aws s3 sync --only-show-errors $M/ds $B/mtg/ft/data/ds/ && echo DS_SYNCED
echo "END $(date -u)"
