# stop the DD driver + pipeline workers on every box (vLLM servers stay up), keep partial outputs aside
D=/workspace/dbalu/reframed/code
for n in $(grep -v '^#' $D/boxes.txt | awk '{print $1}'); do
  ( bash $D/onbox.sh $n 'for p in $(pgrep -f "run_box.sh|run_dense_all.py|charbank_pipeline.py"); do kill $p; done; sleep 2
     [ -d /scratch2/rf_dd ] && mv /scratch2/rf_dd /scratch2/rf_dd_10s_aborted_$(date +%H%M); echo "$(hostname) stopped"' 2>&1 | tail -1 | sed "s/^/$n: /" ) &
done; wait
