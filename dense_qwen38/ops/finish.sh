# usage: finish.sh IP LIST — verify every clip in LIST exists, final sync to box A, then shut the box down
IP=$1; L=$2; K="-o ConnectTimeout=10 -i /home/ubuntu/.ssh/c.pem"
[ "$(ssh $K ubuntu@$IP "ps -eo args | grep -c \"[c]harbank_pipeline.py --video\"")" = 0 ] || { echo "$IP still running"; exit 1; }
rsync -a -e "ssh $K" --include="*/" --include="adlm_descriptions.json" --include="adlm_timeline.json" --include="timings.json" --exclude="*" --prune-empty-dirs ubuntu@$IP:qwen_dense_final/ /prep/qwen_dense_final/
miss=0; for c in $(cat ~/prep_code/$L); do [ -f /prep/qwen_dense_final/${c#*/}/Qwen3.8-27B-FP8/adlm_descriptions.json ] || { miss=$((miss+1)); echo "MISSING $c"; }; done
echo "$IP $L: $(wc -l < ~/prep_code/$L) clips, missing $miss"
[ $miss = 0 ] && ssh $K ubuntu@$IP "sudo shutdown -h +0" >/dev/null 2>&1 && echo "$IP shut down"
