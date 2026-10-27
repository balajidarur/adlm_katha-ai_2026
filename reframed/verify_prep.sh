# per box: clips in /scratch/rf_prep_360p that are in its list, and how many carry the scene-shot source
D=/workspace/dbalu/reframed/code
for n in $(grep -v '^#' $D/boxes.txt | awk '{print $1}'); do
  ( r=$(bash $D/onbox.sh $n "cat > /tmp/vl.txt <<'L'
$(cat $D/lists/$n.txt)
L
python3 - <<'P'
import json,os
L=[l.strip() for l in open('/tmp/vl.txt') if l.strip()]
ok=sc=0
for c in L:
    p=f'/scratch/rf_prep_360p/{c}/meta.json'
    if os.path.exists(p):
        ok+=1; sc+=json.load(open(p)).get('shots_source')=='ffmpeg_scene_cmdad_params'
print(f'{ok}/{len(L)} present, {sc} scene-shot')
P" 2>&1 | tail -1); echo "$n: $r" ) &
done; wait
