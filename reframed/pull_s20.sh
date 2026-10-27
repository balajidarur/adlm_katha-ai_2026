# pull finished 20-sample outputs from the owning boxes into dd20_scenes/
D=/workspace/dbalu/reframed/code; O=/workspace/dbalu/reframed/dd20_scenes; mkdir -p $O
for b in A s4 s1b q3 ev1 cb q4; do
  bash $D/onbox.sh $b 'cd /scratch2/rf_dd_s20 && tar cf - $(ls -d */Qwen3.8-27B-FP8 2>/dev/null | grep -v _logs) 2>/dev/null' | tar xf - -C $O &
done; wait
ls $O/*/Qwen3.8-27B-FP8/adlm_timeline.json | wc -l
