#!/usr/bin/env bash
# Experiment 3 naming: A -> frames_only · B -> dds_gapframes · C -> dds_frames
# Renames the dataset files, their dataset_info.json keys, the meta variant_name field,
# the tokenized cache and the configs. Run only when no training job is reading the data.
set -euo pipefail
declare -A NEW=( [A_uniform1fps_dialogue]=frames_only [B_dense_dialogue_gapframes]=dds_gapframes [C_dense_dialogue_uniform1fps]=dds_frames )

for D in /data/cmd_ad/datasets/frame_variants_448x252_full /data/cmd_ad/datasets/frame_variants_448x252_pilot5; do
  [ -d "$D" ] || continue
  echo "== $D"
  for OLD in "${!NEW[@]}"; do
    for SP in train eval; do
      for SUF in "" _meta; do
        SRC="$D/cmdad_${OLD}_${SP}${SUF}.json"; DST="$D/cmdad_${NEW[$OLD]}_${SP}${SUF}.json"
        [ -f "$SRC" ] && mv -n "$SRC" "$DST" && echo "   $(basename $SRC) -> $(basename $DST)"
      done
    done
  done
  ~/adlm/venv/bin/python - "$D" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
new = {"A_uniform1fps_dialogue": "frames_only", "B_dense_dialogue_gapframes": "dds_gapframes", "C_dense_dialogue_uniform1fps": "dds_frames"}
info = json.load(open(d / "dataset_info.json"))
out = {}
for k, v in info.items():
    for o, n in new.items():
        k = k.replace(o, n); v["file_name"] = v["file_name"].replace(o, n)
    out[k] = v
json.dump(out, open(d / "dataset_info.json", "w"), indent=2)
print("   dataset_info.json keys:", list(out))
for m in d.glob("*_meta.json"):
    rows = json.load(open(m))
    for r in rows:
        for o, n in new.items():
            if r.get("variant_name") == o: r["variant_name"] = n
    json.dump(rows, open(m, "w"), indent=1)
print(f"   variant_name updated in {len(list(d.glob('*_meta.json')))} meta files")
PY
done

# tokenized cache: keep the tokens, change the name
[ -d /data/cmd_ad/tokcache/cmdad_A_frames_4b ] && mv -n /data/cmd_ad/tokcache/cmdad_A_frames_4b /data/cmd_ad/tokcache/exp3_frames_only && echo "== tokcache -> /data/cmd_ad/tokcache/exp3_frames_only"

# configs + runners
cd ~/adlm
for f in configs/adA_frames_4b.yaml configs/adA_frames_4b_probe.yaml configs/evalA_frames_4b.yaml; do
  [ -f "$f" ] || continue
  sed -i -e 's|cmdad_A_uniform1fps_dialogue|cmdad_frames_only|g' \
         -e 's|/data/cmd_ad/tokcache/cmdad_A_frames_4b|/data/cmd_ad/tokcache/exp3_frames_only|g' \
         -e 's|/data/cmd_ad/runs/qwen35-4b/lora/adA_frames_8ep|/data/cmd_ad/runs/exp3_frames_only_4b/lora/8ep|g' \
         -e 's|/data/cmd_ad/runs/qwen35-4b/lora/adA_frames_probe|/data/cmd_ad/runs/exp3_frames_only_4b/lora/probe|g' \
         -e 's|/data/cmd_ad/runs/eval_adA|/data/cmd_ad/runs/exp3_frames_only_4b/test|g' \
         -e 's|^run_name: qwen35-4b-adA-frames-8ep|run_name: exp3-frames-only-4b-8ep|' \
         -e 's|^run_name: adA-probe|run_name: exp3-frames-only-probe|' \
         -e 's|variant A|experiment 3, frames_only|g' "$f"
  new=$(echo "$f" | sed -e 's|adA_frames_4b_probe|exp3_frames_only_4b_probe|' -e 's|adA_frames_4b|exp3_frames_only_4b|' -e 's|evalA_frames_4b|exp3_frames_only_4b_test|')
  [ "$f" != "$new" ] && mv -n "$f" "$new" && echo "== $f -> $new"
done
for s in run_adA.sh run_eval_adA.sh; do
  [ -f "$s" ] || continue
  sed -i -e 's|configs/adA_frames_4b.yaml|configs/exp3_frames_only_4b.yaml|g' \
         -e 's|configs/evalA_frames_4b.yaml|configs/exp3_frames_only_4b_test.yaml|g' \
         -e 's|/data/cmd_ad/runs/eval_adA|/data/cmd_ad/runs/exp3_frames_only_4b/test|g' \
         -e 's|train_adA.log|train_exp3_frames_only.log|g' \
         -e 's|eval_adA_|test_exp3_frames_only_|g' \
         -e 's|evalA_|exp3_test_|g' -e 's|variant A|experiment 3 (frames_only)|g' "$s"
  new=$(echo "$s" | sed -e 's|run_adA.sh|run_exp3_frames_only.sh|' -e 's|run_eval_adA.sh|run_exp3_frames_only_test.sh|')
  mv -n "$s" "$new" && echo "== $s -> $new"
done
echo "done."
