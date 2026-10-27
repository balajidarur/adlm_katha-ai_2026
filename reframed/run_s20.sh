# usage: run_box.sh NGPU — on a fleet box: wait for its servers, then run its clip list (/tmp/list_s20.txt) -> /scratch2/rf_dd
N=$1
until grep -qE "SERVERS_READY|SERVERS_FAILED|SETUP_FAILED" /tmp/box_setup.log; do sleep 20; done
grep -q SERVERS_READY /tmp/box_setup.log || { echo "RUN_ABORT setup failed"; exit 1; }
. /scratch2/envQ.sh; cd ~/rf_code
PORTS=$(seq -s " " 8000 $((8000+N-1)))
for i in $(seq 1 120); do n=0; for p in $PORTS; do curl -s localhost:$p/v1/models > /dev/null && n=$((n+1)); done; [ $n -eq $N ] && break; sleep 10; done
echo "RUN_START $(date +%T) $(wc -l < /tmp/list_s20.txt) clips ports $PORTS"
python -u run_dense_all.py --clips /tmp/list_s20.txt --prep-root /scratch/rf_prep_360p --out-root /scratch2/rf_dd_s20 \
  --ports $PORTS --per-server 4 --prompt-variant desc_action_v2 > /scratch2/rf_dd_s20_driver.log 2>&1
echo "RUN_DONE $(date +%T) $(ls /scratch2/rf_dd_s20/*/Qwen3.8-27B-FP8/adlm_descriptions.json | wc -l) clips"
