# usage: box_full.sh VARIANT CLIPS.txt   (on a box with servers up and prep done)
# outputs /scratch/mtg/dd_full_<VARIANT>/, synced to s3 mtg/dd_full_<VARIANT>/ every 5 min; skip file /scratch/mtg/skip.txt
V=$1; CL=$2
OUT=/scratch/mtg/dd_full_$V
mkdir -p $OUT; touch /scratch/mtg/skip.txt
nohup bash /scratch/code/mtg/ops/sync_loop.sh $OUT s3://balu-transfer-933516006154-apne1/mtg/dd_full_$V/ $(grep -c . $CL) > /dev/null 2>&1 &
SKIP_FILE=/scratch/mtg/skip.txt bash /scratch/code/mtg/ops/run_list.sh $CL $OUT $V ${PER_SERVER:-16} ${PORTS:-8000 8001 8002 8003} > /scratch/mtg/run_full.log 2>&1
echo "FULL_DONE $(date -u)" >> /scratch/mtg/run_full.log
