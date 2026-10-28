#!/usr/bin/env python3
"""Score REFRAMED test/val predictions (teacher-forced or free-running) with the CMD-AD metrics.

Same metrics and code as the CMD-AD scorers, only the ground truth and IDF change:
  GT   = US audio descriptions of the excerpt (gt_us.json: rf_build.join_pieces sentences with start/end,
         built from /storage/datasets/Reframed/{split}/excerpt/audio_description_us; counts = meta n_ad)
  IDF  = all val + test GT ADs (rf_valtest); CMD-AD eval-CSV IDF is reported alongside for reference
  segment CIDEr : k-th predicted AD vs k-th GT AD (eval_adqa_dense / ADQA convention)
  dense tIoU    : eval_adqa_dense dense CIDEr over tIoU 0.1-0.9 (pred span = start + words / median GT rate)
  dense window  : eval_ad.dense greedy nearest-in-time matching at 1 / 2 / 5 s (eval_windows_fixed)
Rows are mapped to excerpts by `idx` when present, else by matching the label's AD texts to the GT.

  python score_rf.py <run> <predictions.jsonl> <test|validation>
"""
import contextlib, io, json, statistics, sys
from collections import defaultdict
sys.path.insert(0, "/home/dbalu/.claude/skills/eval-ad/scripts")
sys.path.insert(0, "/workspace/dbalu/aws_tokyo_preds")
import eval_ad as A
import eval_adqa_dense as E

D = "/workspace/dbalu/aws_tokyo_preds/reframed"
run, src, split = sys.argv[1:4]
GT = json.load(open(f"{D}/gt_us.json"))
order = GT[f"_order_{split}"]
by_text = {tuple(A.norm(t) for _, _, t in GT[v]): v for v in order}

rows, unmapped = {}, 0
for line in open(src):
    if not line.strip():
        continue
    r = json.loads(line)
    vid = order[r["idx"]] if "idx" in r else by_text.get(tuple(t for _, t in A.ads_of(r["label"])))
    if vid is None:
        unmapped += 1
        continue
    rows[vid] = r
clips = [(v, rows[v]) for v in order if v in rows]

all_gt = [t for v in GT if not v.startswith("_") for _, _, t in GT[v]]
rate = statistics.median(len(t.split()) / (e - s) for v in order for s, e, t in GT[v] if e - s > 0.3)
idf = {"rf_valtest": E.make_scorer(all_gt)}
with contextlib.redirect_stdout(io.StringIO()):
    import csv
    idf["cmdad_eval_csv"] = E.make_scorer([x["text"].strip() for x in csv.DictReader(open(E.ANNO_CSV)) if x["split"] == "eval"])

P = {v: [(s, s + max(len(t.split()), 1) / rate, t) for s, t in E.pred_ads(r["predict"])] for v, r in clips}
G = {v: sorted(GT[v]) for v, _ in clips}
res = {"run": run, "split": split, "clips": len(clips), "of": len(order), "unmapped_rows": unmapped,
       "gt_ads": sum(len(G[v]) for v in G), "pred_ads": sum(len(P[v]) for v in P),
       "pred_words_per_ad": round(statistics.mean(len(t.split()) for v in P for _, _, t in P[v]), 2) if any(P.values()) else 0,
       "gt_words_per_ad": round(statistics.mean(len(t.split()) for v in G for _, _, t in G[v]), 2),
       "no_im_end": sum("<|im_end|>" not in r["predict"] for _, r in clips)}

# segment CIDEr, index pairing
pairs, missing = [], 0
for v in G:
    for k, (_, _, gt) in enumerate(G[v]):
        pt = P[v][k][2] if k < len(P[v]) else ""
        missing += not pt
        pairs.append((gt, pt))
res["segment_cider"] = {n: round(E.cider_pairs(sc, pairs)[0] * 100, 2) for n, sc in idf.items()}
res["segment_missing_frac"] = round(missing / max(len(pairs), 1), 3)

# dense CIDEr over tIoU (eval_adqa_dense convention, rf_valtest IDF)
sc = idf["rf_valtest"]
dense = []
for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
    pl, owner, mg, mp = [], [], 0, 0
    for ci, v in enumerate(G):
        hit = set()
        for ps, pe, pt in P[v]:
            h = [gi for gi, (gs, ge, _) in enumerate(G[v]) if E.tiou((ps, pe), (gs, ge)) >= thr]
            mp += bool(h); hit.update(h)
            for gi in h or [None]:
                pl.append((G[v][gi][2] if gi is not None else E.DUMMY, pt)); owner.append(ci)
        mg += len(hit)
    per = E.cider_pairs(sc, pl)[1] if pl else []
    bc = defaultdict(list)
    for ci, x in zip(owner, per):
        bc[ci].append(float(x))
    p, r_ = mp / max(res["pred_ads"], 1), mg / max(res["gt_ads"], 1)
    dense.append({"tiou": thr, "cider": round(statistics.mean(statistics.mean(bc[ci]) if bc[ci] else 0.0 for ci in range(len(G))) * 100, 2),
                  "precision": round(p, 3), "recall": round(r_, 3), "f1": round(2 * p * r_ / max(p + r_, 1e-9), 3)})
res["dense_tiou"] = dense
res["dense_tiou_mean"] = {k: round(statistics.mean(d[k] for d in dense), 3 if k != "cider" else 2) for k in ("cider", "precision", "recall", "f1")}

# window-matched dense CIDEr (eval_windows_fixed), GT text from the label
A.cider_score = lambda pr, _idf: ((lambda s: (s[0] * 100.0, s[1]))(E.cider_pairs(sc, pr) if pr else (0.0, [])))
lr = [(r.get("predict", ""), r.get("label", "")) for _, r in clips]
with contextlib.redirect_stdout(io.StringIO()):
    res["dense_window"] = [A.dense(lr, None, w) for w in (1.0, 2.0, 5.0)]

json.dump(res, open(f"{D}/{run}_metrics.json", "w"), indent=1)
w2 = res["dense_window"][1]
print(f"{run} {split} {len(clips)}/{len(order)} (unmapped {unmapped}): seg CIDEr {res['segment_cider']['rf_valtest']} "
      f"(cmdad-IDF {res['segment_cider']['cmdad_eval_csv']}) | dense tIoU-mean {res['dense_tiou_mean']['cider']} F1 {res['dense_tiou_mean']['f1']} "
      f"| @2s CIDEr {w2['cider']} R {w2['recall']} | ADs {res['pred_ads']}/{res['gt_ads']} words/AD {res['pred_words_per_ad']} vs {res['gt_words_per_ad']} | no_im_end {res['no_im_end']}")
