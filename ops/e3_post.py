#!/usr/bin/env python3
"""E3 / M8 eval591 post-processing on garuda: raw LLaMA-Factory outputs -> merged predictions (M3-591 schema) -> metrics,
ADQA CSVs and Action/CRITIC inputs, the same way T2 scored M3-591.

  python e3_post.py <run dir with teacher_forced/ and free_running_max2048/ (each with stream/*.rank*.jsonl)>
                    <q3vl8b_eval591.json | q3vl8b_eval591_order.json> <out dir>
Writes <out>/{teacher_forced,free_running_max2048}/predictions.jsonl ({idx, video_id, predict, label}, eval591 order =
eval527 then the 64), metrics.json (all 591) and metrics_527.json (first 527 rows, the eval527 set), the ADQA CSVs
<out>/adqa/{tf,free}_591.csv (+ _videos.json) and Action/CRITIC inputs <out>/ac/ (build_ac_inputs_591.py).
"""
import csv, glob, json, os, re, subprocess, sys

run, evalj, out = sys.argv[1:4]
EVAL = "/workspace/dbalu/aws_tokyo_preds/eval_adqa_dense.py"
PY311 = "/usr/bin/python3.11"   # the scorer imports numpy 2.4 (cp311) from /storage/users/dbalu/pylibs
STAGE = "/workspace/dbalu/action_critic/stage"
ANNO = "/storage/users/dbalu/adqa/data/cmd_ad_anno_v1.csv"
recs = json.load(open(evalj))      # q3vl8b_eval591.json or q3vl8b_eval591_order.json
vids = [r if isinstance(r, str) else r["images"][0].split("/")[-2] for r in recs]
assert len(vids) == 591 and len(set(vids)) == 591
year = {}
for r in csv.DictReader(open(ANNO)):
    y, v = r["cmd_filename"].split("/")
    year[v] = y

ns = {}
src = open(EVAL).read()
exec(src.split("sys.path.insert")[0] + "\n" + src[src.index("ANNO_CSV"):src.index("def tokenize")], ns)
pred_ads = ns["pred_ads"]

for sub, kind in (("teacher_forced", "teacher_forced"), ("free_running_max2048", "generated")):
    rows = {}
    for p in glob.glob(f"{run}/{sub}/stream/{kind}.rank*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            rows.setdefault(r["idx"], r)
    miss = [i for i in range(591) if i not in rows]
    print(f"{sub}: {len(rows)} streamed rows, missing idx {miss[:10]}")
    assert not miss, sub
    os.makedirs(f"{out}/{sub}", exist_ok=True)
    with open(f"{out}/{sub}/predictions.jsonl", "w") as f:
        for i, v in enumerate(vids):
            f.write(json.dumps({"idx": i, "video_id": v, "predict": rows[i]["predict"], "label": rows[i].get("label", "")},
                               ensure_ascii=False) + "\n")
    with open(f"{out}/{sub}/predictions_527.jsonl", "w") as f:
        for l in open(f"{out}/{sub}/predictions.jsonl").readlines()[:527]:
            f.write(l)
    tious = "0.5" if sub == "teacher_forced" else "0.1,0.3,0.5,0.7,0.9"
    for name, pf in (("metrics.json", "predictions.jsonl"), ("metrics_527.json", "predictions_527.jsonl")):
        subprocess.run([PY311, EVAL, f"{out}/{sub}/{pf}", "--tious", tious, "--json-out", f"{out}/{sub}/{name}"],
                       check=True, stdout=subprocess.DEVNULL)
    m = json.load(open(f"{out}/{sub}/metrics.json")); m5 = json.load(open(f"{out}/{sub}/metrics_527.json"))
    print(f"{sub} 591: adqa CIDEr {m['adqa_cider']} dense {[(d['tiou'], d['cider'], d['f1']) for d in m['dense']]}")
    print(f"{sub} 527: adqa CIDEr {m5['adqa_cider']} dense {[(d['tiou'], d['cider'], d['f1']) for d in m5['dense']]}")
    # ADQA CSV: the canonical converter (reproduces t2/adqa591 byte-for-byte from M3-591's predictions)
    os.makedirs(f"{out}/adqa", exist_ok=True)
    tag = "tf" if sub == "teacher_forced" else "free"
    subprocess.run([sys.executable, "/workspace/dbalu/adqa/to_adqa_csv.py", "--template",
                    "/workspace/dbalu/adqa/cmd_ad_public_template.csv", "--pred", f"{out}/{sub}/predictions.jsonl",
                    "--mode", tag, "--n", "591", "--out", f"{out}/adqa/{tag}_591.csv"], check=True)

subprocess.run([sys.executable, "/workspace/dbalu/t2/ops/build_ac_inputs_591.py", STAGE,
                f"{out}/teacher_forced/predictions.jsonl", f"{out}/ac"], check=True)
print("POST_DONE", out)
