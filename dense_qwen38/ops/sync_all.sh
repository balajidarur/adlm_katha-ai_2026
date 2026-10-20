# every 10 min: pull every machine's outputs into /prep/qwen_dense_final on box A + combined status
M="172.31.39.100 172.31.34.103 172.31.36.221 172.31.39.93 15.168.133.218 15.164.241.27 15.164.177.204 3.36.13.160 172.31.33.99"
while true; do
  for ip in $M; do
    rsync -a -e "ssh -o ConnectTimeout=10 -i /home/ubuntu/.ssh/c.pem" --include="*/" --include="adlm_descriptions.json" \
      --include="adlm_timeline.json" --include="timings.json" --exclude="*" --prune-empty-dirs \
      ubuntu@$ip:qwen_dense_final/ /prep/qwen_dense_final/ 2>/dev/null &
  done; wait
  python3 /home/ubuntu/prep_code/status_all.py > /prep/qwen_status.tmp && mv /prep/qwen_status.tmp /prep/qwen_status.txt
  cat /prep/qwen_status.txt >> /prep/qwen_status.log
  sleep 600
done
