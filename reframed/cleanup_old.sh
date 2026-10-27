# delete the old fixed-10 s prep (and aborted 10 s DD outputs) on the given boxes
D=/workspace/dbalu/reframed/code
for n in "$@"; do
  ( bash $D/onbox.sh $n 'rm -rf /scratch/rf_prep_360p_10s /scratch/rf_prep_360p_old /scratch/rf_prep_new /scratch2/rf_dd_10s_aborted_* /scratch2/shards /scratch/rf_data /scratch/rf_pilot 2>/dev/null
     [ -d /scratch/rf_prep_360p_10s ] || echo "$(hostname) clean, free $(df -h /scratch | tail -1 | awk "{print \$4}")"' 2>&1 | tail -1 | sed "s/^/$n: /" ) &
done; wait
