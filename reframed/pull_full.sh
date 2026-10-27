# usage: pull_full.sh BOX... — copy finished DD outputs (no logs) into dd_full/
D=/workspace/dbalu/reframed/code; O=/workspace/dbalu/reframed/dd_full; mkdir -p $O
for b in "$@"; do
  ( bash $D/onbox.sh $b 'cd /scratch2/rf_dd && tar cf - $(ls -d */Qwen3.8-27B-FP8 | grep -v _logs) run_log.jsonl 2>/dev/null | pigz -p 4' | pigz -dc | tar xf - -C $O --transform "s#^run_log.jsonl#_runlogs/$b.jsonl#" && echo "$b pulled" ) &
done; wait
echo "local: $(ls $O/*/Qwen3.8-27B-FP8/adlm_descriptions.json | wc -l)"
