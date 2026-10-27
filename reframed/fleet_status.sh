# one line per box: setup stage / prep / servers / DD progress
D=/workspace/dbalu/reframed/code
for n in $(grep -v '^#' $D/boxes.txt | awk '{print $1}'); do
  ( s=$(timeout 25 bash $D/onbox.sh $n 'l=/tmp/box_setup.log; st=$(grep -oE "PREP_READY [0-9]+|SERVERS_READY [0-9]+|SERVERS_FAILED|SETUP_FAILED" $l | tr "\n" " "); dd=$(ls /scratch2/rf_dd/*/Qwen3.8-27B-FP8/adlm_descriptions.json 2>/dev/null | wc -l); echo "$st dd=$dd"' 2>&1 | tail -1)
    echo "$n: $s" ) &
done; wait
