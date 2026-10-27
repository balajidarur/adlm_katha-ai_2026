# usage: bringup.sh IP NGPU NAME — copy bundle + setup script to a box and start setup (log /tmp/box_setup.log there)
IP=$1; N=$2; NAME=$3; K=~/.ssh/balu-aws-test.pem; O="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
for t in $(seq 1 40); do ssh $O -i $K ubuntu@$IP true 2>/dev/null && break; sleep 10; done
scp -q $O -i $K $CLAUDE_JOB_DIR/tmp/rf_bundle.tgz /workspace/dbalu/reframed/code/box_setup.sh ubuntu@$IP:/tmp/ && \
ssh $O -i $K ubuntu@$IP "nohup bash /tmp/box_setup.sh $N > /tmp/box_setup.log 2>&1 < /dev/null & echo $NAME setup started"
