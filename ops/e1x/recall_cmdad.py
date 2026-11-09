#!/usr/bin/env python3
"""R@k/N (AutoAD-II recall_within_neighbours.py) on CMD-AD, with the grouping/ordering variants found in papers.

Official function: BERTScorer(lang="en", rescale_with_baseline=True) F1 between every generated AD and every GT AD
of a window; windows of N with stride N//2 over a sequence (tail beyond the last full window ignored); top-k accuracy of
the diagonal. Only the N x N blocks the official loop reads are scored (identical result).

Groupings (all ordered by imdbid, then YouTube id, then GT start, as StrAD D.3 prescribes):
  clip    windows inside each clip; clips with fewer than N ADs skipped; micro-average over windows
  clipmac same windows, macro-average over clips
  movie   clips of one movie concatenated; windows per movie; micro over windows
  global  one sequence over the whole split (StrAD's "single, correctly-ordered sequence ... across all videos")
Reports R@1/5, R@5/5 is trivial so k=(1,) for N=5 plus R@1/16 and R@5/16 for N=16.

  python recall_cmdad.py --csv in.csv --out out.json      (columns: anno_idx, imdbid, cmd_filename, start, text_gt, text_gen)
"""
import argparse, json

import pandas as pd
import torch
from bert_score import BERTScorer


def topk_acc(F, topk):               # = calc_topk_accuracy(F, arange(N), topk) of the official file
    N = F.shape[0]
    _, pred = F.topk(max(topk), 1, True, True)
    correct = pred.t().eq(torch.arange(N).view(1, -1).expand(max(topk), N))
    return [correct[:k].reshape(-1).float().sum().item() / N for k in topk]


def windows(seq, N):
    return [seq[i:i + N] for i in range(0, len(seq) - N + 1, N // 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    df = pd.read_csv(a.csv, keep_default_na=False)
    df["yt"] = df.cmd_filename.str.split("/").str[-1]
    df["start"] = df.start.astype(float)
    df = df.sort_values(["imdbid", "yt", "start", "anno_idx"]).reset_index(drop=True)
    gen, gt = df.text_gen.tolist(), df.text_gt.tolist()
    seqs = {"clip": [g.index.tolist() for _, g in df.groupby(["imdbid", "yt"], sort=True)],
            "movie": [g.index.tolist() for _, g in df.groupby("imdbid", sort=True)],
            "global": [df.index.tolist()]}
    cfg = {5: (1,), 16: (1, 5)}
    blocks = {}                      # (grouping, N) -> list of (seq_id, window row ids)
    need = set()
    for gname, ss in seqs.items():
        for N in cfg:
            bl = []
            for sid, s in enumerate(ss):
                for w in windows(s, N):
                    bl.append((sid, w)); need.update((r, q) for r in w for q in w)
            blocks[(gname, N)] = bl
    need = sorted(need)
    scorer = BERTScorer(lang="en", rescale_with_baseline=True, batch_size=a.bs, device="cuda" if torch.cuda.is_available() else "cpu")
    F = scorer.score([gen[r] for r, _ in need], [gt[q] for _, q in need])[-1].tolist()   # F1(gen_r, gt_q)
    S = dict(zip(need, F))
    res = {"csv": a.csv, "n_ads": len(df), "pairs_scored": len(need)}
    for (gname, N), bl in blocks.items():
        ks = cfg[N]
        accs = [topk_acc(torch.tensor([[S[(r, q)] for q in w] for r in w]), ks) for _, w in bl]
        for j, k in enumerate(ks):
            res[f"R@{k}/{N}_{gname}"] = round(100 * sum(x[j] for x in accs) / max(len(accs), 1), 2)
            if gname == "clip":
                per = {}
                for (sid, _), x in zip(bl, accs):
                    per.setdefault(sid, []).append(x[j])
                res[f"R@{k}/{N}_clipmac"] = round(100 * sum(sum(v) / len(v) for v in per.values()) / max(len(per), 1), 2)
        res[f"windows_N{N}_{gname}"] = len(bl)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
