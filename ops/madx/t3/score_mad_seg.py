#!/usr/bin/env python3
"""[MADX: T3 copy + --preds-json]
MAD-Eval segment-level scoring of M3 teacher-forced output (community protocol: one AD per GT AD interval).

Mapping: the k-th [AD] line the model wrote in a window fills the k-th GT slot of that window, GT slots sorted by
(start, end, text) exactly as the dataset builder rendered them (frames_lib.render_target); each slot carries its
madeval_anno.csv row index (anno_idx). Budget tokens are stripped, whitespace collapsed.

Metrics over all 6,520 GT ADs (corpus level, pycocoevalcap with PTB tokenization):
  CIDEr   ADQA-vendored Cider with IDF over the 6,520 MAD-Eval GT ADs (the scorer used for all our CMD-AD numbers);
          also pycocoevalcap CiderD (IDF from the same refs) as a check
  BLEU-1..4, METEOR, ROUGE-L, SPICE   pycocoevalcap

  python score_mad_seg.py --run DIR --win rc_win --anno madeval_anno.csv --out OUT
     DIR: <box>/teacher_forced_predictions.jsonl + order_<box>.json for every box (rc_ds/order_*.json)
Out: OUT/preds_tf.csv (anno_idx, imdbid, start, end, text_gt, text_gen), OUT/seg_metrics.json
"""
import argparse, contextlib, csv, glob, io, json, os, re, sys

sys.path.insert(0, "/storage/users/dbalu/adqa")
sys.path.insert(0, "/storage/users/dbalu/pylibs")

AD_RE = re.compile(r"^\s*(\d+):(\d\d(?:\.\d+)?)\s*\[AD\]\s*(.*)$")
SPECIAL_RE = re.compile(r"<\|[^|]*\|>")


def clean(s):
    return re.sub(r"\s+", " ", SPECIAL_RE.sub(" ", s)).strip()


def pred_ads(text):
    text = text.split("<|im_end|>")[0].split("<|endoftext|>")[0]
    out = []
    for line in text.split("\n"):
        m = AD_RE.match(line)
        if m:
            out.append((int(m.group(1)) * 60 + float(m.group(2)), clean(m.group(3))))
    return out


def load_anno(path):
    raw = open(path, "rb").read()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("cp1252")
    return list(csv.DictReader(txt.splitlines()))


def load_preds(run_dir, ds_dir, kind):
    """{wid: predict} from every box shard; rows are in dataset order = order_<box>.json."""
    fn = {"tf": "teacher_forced_predictions.jsonl", "free": "generated_predictions.jsonl"}[kind]
    out, warn = {}, []
    for op in sorted(glob.glob(f"{ds_dir}/order_*.json")):
        box = os.path.basename(op)[6:-5]
        order = json.load(open(op))
        p = f"{run_dir}/m3_{kind}_{box}/{fn}"
        rows = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        if len(rows) != len(order):
            warn.append(f"{kind} {box}: {len(rows)} rows for {len(order)} windows")
        for wid, r in zip(order, rows):
            out[wid] = r["predict"]
    return out, warn


def tokenize(caps):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    d = {str(i): [{"caption": c}] for i, c in enumerate(caps)}
    with contextlib.redirect_stdout(io.StringIO()):
        t = PTBTokenizer().tokenize(d)
    return [t[str(i)][0] for i in range(len(caps))]


def caption_metrics(refs, cands):
    from cider.cider import Cider as AdqaCider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.spice.spice import Spice
    R, C = tokenize(refs), tokenize([c if c else "" for c in cands])
    gts = {str(i): [r] for i, r in enumerate(R)}
    res = {str(i): [c] for i, c in enumerate(C)}
    out = {}
    ac = AdqaCider(); ac.init_vocabulary(list_of_refs=[[t] for t in R]); ac.reset_cider_scorer(persist_vocabulary=True)
    out["CIDEr"] = ac.compute_score(gts, res)[0] * 100
    out["CIDEr_pycoco"] = Cider().compute_score(gts, res)[0] * 100
    b = Bleu(4).compute_score(gts, res, verbose=0)[0]
    out.update({f"BLEU-{i + 1}": b[i] * 100 for i in range(4)})
    out["METEOR"] = Meteor().compute_score(gts, res)[0] * 100
    out["ROUGE-L"] = Rouge().compute_score(gts, res)[0] * 100
    try:
        out["SPICE"] = Spice().compute_score(gts, res)[0] * 100
    except Exception as e:  # SPICE needs its Stanford CoreNLP jars
        out["SPICE"] = None; out["SPICE_error"] = str(e)[:300]
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--ds"); ap.add_argument("--preds-json", help="MADX: {wid: predict} merged from shards"); ap.add_argument("--win", required=True)
    ap.add_argument("--anno", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    anno = load_anno(a.anno)
    W = json.load(open(f"{a.win}/windows.json"))
    preds, warn = (json.load(open(a.preds_json)), []) if a.preds_json else load_preds(a.run, a.ds, "tf")
    gen = {}
    stats = {"windows": sum(len(v) for v in W.values()), "windows_with_pred": 0, "count_mismatch_windows": 0,
             "slots_without_pred": 0, "extra_preds": 0, "empty_preds": 0, "warnings": warn}
    for m, ws in W.items():
        for w in ws:
            gt = sorted((g[0], g[1], g[2], g[3]) for g in json.load(open(f"{a.win}/gt/{w['wid']}.json")))
            if w["wid"] not in preds:
                continue
            stats["windows_with_pred"] += 1
            P = pred_ads(preds[w["wid"]])
            if len(P) != len(gt):
                stats["count_mismatch_windows"] += 1
                stats["extra_preds"] += max(0, len(P) - len(gt))
            for k, (s, e, t, idx) in enumerate(gt):
                if k < len(P):
                    if abs(P[k][0] - round(s, 1)) > 0.11:
                        stats.setdefault("time_mismatch", 0); stats["time_mismatch"] += 1
                    gen[idx] = P[k][1]
                    stats["empty_preds"] += not P[k][1]
                else:
                    stats["slots_without_pred"] += 1
    rows = []
    for i, r in enumerate(anno):
        rows.append({"anno_idx": i, "imdbid": r["imdb"], "movie": r["movie"], "start": r["start"], "end": r["end"],
                     "text_gt": r["sentence"].strip(), "text_gen": gen.get(i, "")})
    with open(f"{a.out}/preds_tf.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    stats["gt_ads"] = len(rows); stats["filled"] = sum(1 for r in rows if r["text_gen"])
    stats["mean_words_gen"] = round(sum(len(r["text_gen"].split()) for r in rows) / len(rows), 2)
    stats["mean_words_gt"] = round(sum(len(r["text_gt"].split()) for r in rows) / len(rows), 2)
    res = {"stats": stats, "metrics": caption_metrics([r["text_gt"] for r in rows], [r["text_gen"] for r in rows])}
    json.dump(res, open(f"{a.out}/seg_metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
