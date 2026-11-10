#!/usr/bin/env python3
"""R@k/N (AutoAD-II) on MAD-Eval: R@5/16 (and R@1/16).

Official function: AutoAD/autoad_ii/recall_within_neighbours.py (BERTScorer(lang="en", rescale_with_baseline=True),
F1 of BERTScore; per movie, GT and predictions in time order; windows of N=16 with stride 8; top-k accuracy of the
diagonal). Only the 16x16 blocks the official loop reads are scored (the full gen x GT matrix is never used elsewhere),
so the numbers are identical to the official code at a fraction of the cost.
Aggregation: micro-average over all windows of all movies (StrAD D.3); per-movie macro also reported.

  python recall_mad.py --csv preds_tf.csv --out recall.json
"""
import argparse, json, sys

import pandas as pd
import torch
from bert_score import BERTScorer

sys.path.insert(0, "/workspace/dbalu/tasks_nov8/T3/ext/AutoAD/autoad_ii")


def calc_topk_accuracy(output, target, topk=(1,)):   # copied from the official file
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return torch.stack(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=16)
    a = ap.parse_args()
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    df = pd.read_csv(a.csv, keep_default_na=False)
    df["start"] = df["start"].astype(float)
    N, topk = a.N, (1, 5)
    allw, per_movie = [], {}
    for imdb, g in df.groupby("imdbid", sort=True):
        g = g.sort_values(["start", "anno_idx"])
        gt, gen = g.text_gt.tolist(), g.text_gen.tolist()
        outs = []
        for i in range(0, len(gen) - N + 1, N // 2):
            c = [gen[i + r] for r in range(N) for _ in range(N)]
            rf = [gt[i + q] for _ in range(N) for q in range(N)]
            F = scorer.score(rf, c)[-1].view(N, N)          # F[r, q] = BERTScore(gen_r, gt_q)
            outs.append(calc_topk_accuracy(F, torch.arange(N), topk=topk))
        o = torch.stack(outs, 0)
        per_movie[imdb] = o.mean(0).tolist()
        allw.append(o)
    allw = torch.cat(allw, 0).mean(0).tolist()
    res = {f"R@{k}/{N}": round(100 * v, 2) for k, v in zip(topk, allw)}
    res["per_movie"] = {m: {f"R@{k}/{N}": round(100 * v, 2) for k, v in zip(topk, x)} for m, x in per_movie.items()}
    res["macro"] = {f"R@{k}/{N}": round(100 * sum(x[j] for x in per_movie.values()) / len(per_movie), 2) for j, k in enumerate(topk)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
