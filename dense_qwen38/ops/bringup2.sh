# usage: bringup2.sh IP SHARD NGPU — setup + data only (driver started later)
IP=$1; SH=$2; NG=$3; K="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i /home/ubuntu/.ssh/c.pem"
exec > /prep/bringup_$SH.log 2>&1
until ssh $K ubuntu@$IP true; do sleep 5; done; echo "$(date +%T) SSH_UP"
scp $K /scratch/envA.sh /scratch/serveA.sh /home/ubuntu/prep_code/setupA2.sh /home/ubuntu/prep_code/remote_setup2.sh ubuntu@$IP:/tmp/
ssh $K ubuntu@$IP "bash /tmp/remote_setup2.sh $NG > /tmp/remote_setup.log 2>&1 < /dev/null &"
until ssh $K ubuntu@$IP "grep -q MOUNTED /tmp/remote_setup.log"; do sleep 5; done; echo "$(date +%T) MOUNTED"
cd /prep/prep_360p; split -n l/4 -d /prep/files_$SH.txt /prep/files_$SH.part
for p in /prep/files_$SH.part0*; do (tar -cf - -T $p | ssh $K ubuntu@$IP "tar -xf - -C /scratch/prep_360p") & done
(tar -cf - -C /data/cmd_ad/asr SRT | ssh $K ubuntu@$IP "tar -xf - -C /scratch/asr") &
(tar -cf - -C /home/ubuntu prep_code | ssh $K ubuntu@$IP "tar -xf - -C /home/ubuntu") &
wait; echo "$(date +%T) DATA_DONE files=$(ssh $K ubuntu@$IP "find /scratch/prep_360p -type f | wc -l") expected=$(wc -l < /prep/files_$SH.txt)"
until ssh $K ubuntu@$IP "grep -q -E \"SERVER_READY|SERVER_FAILED\" /tmp/remote_setup.log"; do sleep 10; done
echo "$(date +%T) $(ssh $K ubuntu@$IP "tail -1 /tmp/remote_setup.log")"
