# s4, after e3_build.sh (+ e3_videos.sh) and the s1a DDs: extra eval rows, selection, checks, probe sets, upload.
#   bash e3_finalize.sh <commit>
WANT=$1; set -x
W=/opt/dlami/nvme/e3; cd $W; . $W/cr.env; B=s3://balu-transfer-933516006154-apne1
P(){ echo "$(date -u +%H:%M) $*" | tee -a $W/progress.txt; }
R=$W/adlm_repo; PY=$HOME/adlm/venv/bin/python
HAVE=$(git -C $R rev-parse --short HEAD); [ "$HAVE" = "$WANT" ] || { P COMMIT_MISMATCH; exit 1; }
bash $R/ops/e3_build_extra.sh $WANT > $W/e3_build_extra.log 2>&1 || { P EXTRA_FAILED; exit 1; }
O=$W/ds; rm -rf $O
cd $R/dataset && $PY -u select_e3.py --name q3vl8b --v $W/v_q3vl --m3 $W/m3/frame_variants_448x252_full_qwen38 \
  --eval527 $W/ref/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38/eval527.json --eval-ids $W/eval_ids.txt \
  --extra $W/extra/build/extra_eval.json --out $O 2>&1 | tee $W/select.log | grep -v "^[0-9a-f]\{64\}" | tee -a $W/progress.txt
[ -f $O/q3vl8b_eval591.json ] || { P SELECT_INCOMPLETE; exit 1; }
$PY - <<PY | tee -a $W/progress.txt
import json, os
for n in ("q3vl8b_f150_40k", "q3vl8b_eval591"):
    d = json.load(open(f"$O/{n}.json"))
    miss = sum(not os.path.exists(p) for r in d for p in r["images"])
    bad = sum(r["messages"][0]["content"].count("<image>") != len(r["images"]) for r in d)
    print(f"CHECK {n}: {len(d)} records | {sum(len(r['images']) for r in d):,} frames | missing frames {miss} | image/tag mismatch {bad}")
PY
$PY -u e3_probe_ds.py --model $W/models/Qwen3-VL-8B-Instruct --train $O/q3vl8b_f150_40k.json --out $W/probe 2>&1 | tail -20 | tee -a $W/progress.txt
# eval591 lengths too (cutoff check)
$PY - <<PY | tee -a $W/progress.txt
import json, sys
sys.argv = ["x"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$W/models/Qwen3-VL-8B-Instruct")
tok.add_tokens([f"<|budget_{i}|>" for i in range(9)], special_tokens=True)
vis = "<|vision_start|>" + "<|image_pad|>" * 112 + "<|vision_end|>"
d = json.load(open("$O/q3vl8b_eval591.json"))
L = sorted(len(tok(r["messages"][0]["content"].replace("<image>", vis) + r["messages"][1]["content"], add_special_tokens=False)["input_ids"]) + 20 for r in d)
print(f"EVAL591 Qwen3-VL lengths: max {L[-1]}, p50 {L[len(L)//2]}, over 49152: {sum(x > 49152 for x in L)}, over 65536: {sum(x > 65536 for x in L)}")
PY
# frames the E3 builds added to the pool (resampled frames for the longer DD text, and the extra clips' frames)
cd $W/frames && find 1fps_448x252 -newermt "2026-11-08 19:00" -type f > $W/frames_new.txt; P "NEW_FRAMES $(wc -l < $W/frames_new.txt)"
tar -cf $W/frames_new.tar -C $W/frames -T $W/frames_new.txt
. $W/cr.env; D=$B/cmd_ad_dense/datasets/cmdad_q3vl8b_dd
aws s3 sync --only-show-errors $O/ $D/ && aws s3 cp --only-show-errors $W/frames_new.tar $D/frames_new_1fps_448x252.tar && \
  aws s3 cp --only-show-errors $W/frames_new.txt $D/frames_new.txt && aws s3 sync --only-show-errors $W/probe/ $B/adlm/datasets/cmdad_q3vl8b_probe/ && P "UPLOADED $D"
aws s3 ls $D/ | tee -a $W/progress.txt
P FINALIZE_DONE
