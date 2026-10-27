# full-run progress per box: done clips / list, failures
D=/workspace/dbalu/reframed/code
for n in $(grep -v '^#' $D/boxes.txt | awk '{print $1}'); do
  ( r=$(bash $D/onbox.sh $n 'd=$(ls /scratch2/rf_dd/*/Qwen3.8-27B-FP8/adlm_descriptions.json 2>/dev/null | wc -l); f=$(grep -c "\"rc\": [1-9]" /scratch2/rf_dd/run_log.jsonl 2>/dev/null); echo "$d $(wc -l < /tmp/list.txt) ${f:-0} $(grep -c RUN_DONE /tmp/run_box.log)"' 2>&1 | tail -1); echo "$n $r" ) &
done; wait
