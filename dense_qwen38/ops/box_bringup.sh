# runs on box A: bring up one new box end to end.  usage: box_bringup.sh IP SHARD(q1..q4)
IP=$1; SH=$2; K="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i /home/ubuntu/.ssh/c.pem"
L=/prep/bringup_$SH.log; exec > $L 2>&1
until ssh $K ubuntu@$IP true; do sleep 5; done; echo "$(date +%T) SSH_UP"
scp $K /scratch/envA.sh /scratch/serveA.sh /scratch/setupA.sh /home/ubuntu/prep_code/remote_setup.sh ubuntu@$IP:/tmp/
ssh $K ubuntu@$IP "bash /tmp/remote_setup.sh > /tmp/remote_setup.log 2>&1 &"
until ssh $K ubuntu@$IP "grep -q MOUNTED /tmp/remote_setup.log"; do sleep 5; done; echo "$(date +%T) MOUNTED"
# data: 3 parallel tar streams of the slim file list + SRT + code
cd /prep/prep_360p
split -n l/3 -d /prep/files_$SH.txt /prep/files_$SH.part
for p in /prep/files_$SH.part0*; do (tar -cf - -T $p | ssh $K ubuntu@$IP "tar -xf - -C /scratch/prep_360p") & done
(tar -cf - -C /data/cmd_ad/asr SRT | ssh $K ubuntu@$IP "tar -xf - -C /scratch/asr") &
(tar -cf - -C /home/ubuntu prep_code | ssh $K ubuntu@$IP "tar -xf - -C /home/ubuntu") &
wait; echo "$(date +%T) DATA_DONE files=$(ssh $K ubuntu@$IP 'find /scratch/prep_360p -type f | wc -l') expected=$(wc -l < /prep/files_$SH.txt)"
until ssh $K ubuntu@$IP "grep -q -E 'SERVER_READY|SERVER_FAILED' /tmp/remote_setup.log"; do sleep 10; done
ssh $K ubuntu@$IP "tail -1 /tmp/remote_setup.log"
ssh $K ubuntu@$IP "cd ~/prep_code && . /scratch/envA.sh && PYTHONUNBUFFERED=1 nohup python -u run_dense_all.py --clips clips_$SH.txt --prep-root /scratch/prep_360p --out-root /home/ubuntu/qwen_dense_v2 --ports 8000 --per-server 4 --prompt-variant desc_action_v2 > /home/ubuntu/qwen_dense_v2/driver.log 2>&1 & nohup python3 qprog.py /home/ubuntu/qwen_dense_v2 clips_$SH.txt /scratch/prep_360p > /dev/null 2>&1 & sleep 5; head -1 /home/ubuntu/qwen_dense_v2/driver.log"
echo "$(date +%T) RUNNING"
