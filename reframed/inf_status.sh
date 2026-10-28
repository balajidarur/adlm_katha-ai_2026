# 5-min inference tick: queue log + progress bar of every running job on box A and box C
K=~/.ssh/balu-aws-test.pem; O="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR -i $K"
for b in A:13.115.233.87 C:54.168.178.88; do echo "== box ${b%%:*}"
ssh $O ubuntu@${b#*:} 'tail -n 4 /scratch/inf/queue.log; for f in /scratch/inf/out/*.log; do r=$(basename $f .log); grep -q "END $r" /scratch/inf/queue.log && continue
  echo "$r: $(tail -c 400 $f | tr "\r" "\n" | grep -a "it/s\|s/it" | tail -1 | sed "s/.*| //")"; done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr "\n" " "; echo'
done
