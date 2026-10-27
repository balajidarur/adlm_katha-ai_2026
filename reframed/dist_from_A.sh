# on box A: send each box only its clips (lists in /tmp/lists/<box>.txt) + portraits/crops; becomes /scratch/rf_prep_360p there
S=/scratch/rf_prep_scenes; K=~/.ssh/c.pem
RECV='set -e; rm -rf /scratch/rf_prep_new; mkdir -p /scratch/rf_prep_new; tar xf - -C /scratch/rf_prep_new
if [ -d /scratch/rf_prep_360p ]; then mv /scratch/rf_prep_360p /scratch/rf_prep_360p_10s; fi
mv /scratch/rf_prep_new /scratch/rf_prep_360p; ls /scratch/rf_prep_360p/*/*/meta.json | wc -l'
send(){ name=$1; dest=$2; via=$3
  O=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $K)
  [ -n "$via" ] && O+=(-o "ProxyCommand=ssh -i $K -o StrictHostKeyChecking=no -W %h:%p ubuntu@$via")
  n=$(cd $S && tar cf - _portraits _crops $(cat /tmp/lists/$name.txt) | ssh "${O[@]}" ubuntu@$dest "bash -c '$RECV'")
  echo "$(date +%T) $name got $n clips (list $(wc -l < /tmp/lists/$name.txt))"; }
send cb 172.31.42.17 & send ev1 172.31.42.106 & send ev2 172.31.38.147 & send q1 172.31.39.100 &
send q2 172.31.34.103 & send q3 172.31.36.221 & send q4 172.31.39.93 & send s1b 54.117.65.59 &
send s4 172.31.18.241 54.117.65.59 & send s1a 172.31.20.249 54.117.65.59 &
( if [ -d /scratch/rf_prep_360p ] && [ ! -L /scratch/rf_prep_360p ]; then mv /scratch/rf_prep_360p /scratch/rf_prep_360p_10s; fi
  ln -sfn $S /scratch/rf_prep_360p; echo "$(date +%T) A linked" ) &
wait; echo DIST_DONE
