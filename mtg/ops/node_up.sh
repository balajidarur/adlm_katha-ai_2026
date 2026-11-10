#!/bin/bash
# Bring up an extra MTG DD node and run its shard.  usage: node_up.sh HASH NGPU SHARD.txt [PER_SERVER]
# (box_setup.sh + box_code.sh must sit in $HOME; shard keys are mtg/<id>; videos are fetched per clip from S3)
HASH=$1; NGPU=$2; SHARD=$3; PS=${4:-8}
NO_VIDEOS=1 bash ~/box_setup.sh
bash ~/box_code.sh $HASH >> /scratch_setup.log 2>&1 || exit 1
NGPU=$NGPU bash /scratch/code/mtg/ops/start_servers.sh >> /scratch_setup.log
. $HOME/.cr.env; export AWS_DEFAULT_REGION=ap-northeast-1
B=s3://balu-transfer-933516006154-apne1/mtg/videos_480p
mkdir -p /scratch/mtg/videos_480p/test /scratch/mtg/videos_480p/validation
cp $SHARD /scratch/mtg/shard.txt
: > /scratch/mtg/prep_shard.tsv
for k in $(cat $SHARD); do v=${k#mtg/}
  for sp in test validation; do
    aws s3 cp --only-show-errors "$B/$sp/$v.mp4" "/scratch/mtg/videos_480p/$sp/$v.mp4" 2>/dev/null && { echo "$k /scratch/mtg/videos_480p/$sp/$v.mp4 $v 0 -" >> /scratch/mtg/prep_shard.tsv; break; }
  done
done
bash /scratch/code/mtg/ops/prep_list.sh /scratch/mtg/prep_shard.tsv $(( $(nproc) - 2 )) > /scratch/mtg/prep_shard.log 2>&1
for g in $(seq 0 $((NGPU-1))); do until curl -sf localhost:$((8000+g))/v1/models >/dev/null; do sleep 10; done; done
echo "SERVERS_UP $(date -u)" >> /scratch_setup.log
PORTS="$(seq -s ' ' 8000 $((8000+NGPU-1)))" PER_SERVER=$PS bash /scratch/code/mtg/ops/box_full.sh nocast_mtg_b /scratch/mtg/shard.txt
