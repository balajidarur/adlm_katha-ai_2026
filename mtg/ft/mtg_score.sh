#!/bin/bash
# One-command MTG official scorer (garuda): prep (Piper 1.2x TTS) + run_ad_eval.py (default Gemini judge) for one system,
# parallel over P id-chunks (one run_ad_eval process per chunk, same output dir; per-video files).
#   P=16 bash mtg_score.sh <name> <split: test|val10> (--preds P.jsonl | --auto-ad DIR)
# -> /storage/users/dbalu/mtg_eval/sys/<name>_<split>/{auto_ad,tts,visual_info,eval/metrics/output_*.json}
NAME=$1; SPLIT=$2; shift 2; P=${P:-16}; E=/storage/users/dbalu/mtg_eval; O=$E/sys/${NAME}_$SPLIT; R=/workspace/dbalu/tasks_nov8/T4/mtg-ad
python3 -c "import json; s=json.load(open('/workspace/dbalu/tasks_nov8/T5/split.json')); json.dump(s['$SPLIT'], open('$E/ids_$SPLIT.json','w'))"
python3 $(dirname $(readlink -f $0))/mtg_eval_prep.py "$@" --ids $E/ids_$SPLIT.json --anno-dir /workspace/dbalu/tasks_nov8/T4/data --out $O || exit 1
SRC=$([ $SPLIT = test ] && echo test || echo validation); VID=$([ $SPLIT = test ] && echo /workspace/dbalu/tasks_nov8/T4/videos || echo /workspace/dbalu/tasks_nov8/T4/videos_val)
mkdir -p $O/chunks; split -n l/$P -d $O/ids.txt $O/chunks/ids_
cd $R
for c in $O/chunks/ids_*; do
  python3 -u run_ad_eval.py --api_key_path $E/gemini_key.txt --video_ids_file $c --output_dir $O/eval \
    --input_transcript_dir $E/raw/$SRC/transcript --input_gt_ad_dir $E/raw/$SRC/gt_ad --input_visual_info_dir $O/visual_info \
    --input_auto_ad_dir $O/auto_ad --input_tts_dir $O/tts --input_video_dir $VID --no_generate_missing_inputs > $c.log 2>&1 &
done; wait
echo "EVAL_DONE $NAME $SPLIT $(ls $O/eval/metrics | wc -l)/$(wc -l < $O/ids.txt) metrics files"
