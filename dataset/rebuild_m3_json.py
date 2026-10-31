#!/usr/bin/env python3
"""Rebuild the exact M3 JSONs (Qwen3.8 DDs, 0.5 s budget bins) where box A's /data is out of reach.

  gapframes_bins_train_f150_40k.json : mk40k_qwen38 (mk150.cap + 40k est filter) on gapframes_bins_train.json,
                                       then checked clip by clip against bins02_f150_40k.json (same clips/frames/AD words).
  eval527.json                       : bins02_eval527 prompts with the 0.2 s legend swapped back to the 0.5 s legend,
                                       targets = the 0.5 s labels of the M3 eval run (qwen38_budget_460_tf2, by idx).

  python rebuild_m3_json.py --src gapframes_bins_train.json --b02-train bins02_f150_40k.json \
      --b02-eval bins02_eval527.json --m3-labels qwen38_budget_460_tf2.jsonl --out <dir>
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ab = {}
exec(open(f"{HERE}/add_bins_to_gapframes.py").read().split('if __name__ == "__main__":')[0], ab)
mk = {}
exec(open(f"{HERE}/mk150.py").read().split("d = json.load")[0], mk)


def strip(t):
    return re.sub(r"<\|budget_\d+\|>", "", t)


def body(u):
    return u.split("\n\n", 1)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("src", "b02-train", "b02-eval", "m3-labels", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # train: same build as mk40k_qwen38.py
    d = json.load(open(a.src))
    for r in d:
        mk["cap"](r)
    train = [r for r in d if mk["est"](r) <= 40000]
    ref = json.load(open(a.b02_train))
    assert len(train) == len(ref) == 7763, (len(train), len(ref))
    for x, y in zip(train, ref):
        assert x["images"] == y["images"]
        assert body(x["messages"][0]["content"]) == body(y["messages"][0]["content"])
        assert strip(x["messages"][1]["content"]).split() == strip(y["messages"][1]["content"]).split()
        assert x["messages"][0]["content"].startswith(ab["BIN_LEGEND"])
    json.dump(train, open(f"{a.out}/gapframes_bins_train_f150_40k.json", "w"), ensure_ascii=False)
    print(f"train: {len(train)} clips, identical frames/prompt/AD words to bins02_f150_40k")

    # eval527
    lab = {}
    for l in open(a.m3_labels):
        r = json.loads(l)
        lab[r["idx"]] = r["label"].split("<|im_end|>")[0].strip()
    ev = json.load(open(a.b02_eval))
    assert len(ev) == 527 and sorted(lab) == list(range(527))
    out = []
    for i, r in enumerate(ev):
        u = ab["BIN_LEGEND"] + "\n\n" + body(r["messages"][0]["content"])
        assert strip(lab[i]).split() == strip(r["messages"][1]["content"]).split(), i
        out.append(dict(r, messages=[dict(r["messages"][0], content=u), dict(r["messages"][1], content=lab[i])]))
    json.dump(out, open(f"{a.out}/eval527.json", "w"), ensure_ascii=True)   # box A wrote eval527 ASCII-escaped: byte-identical
    print(f"eval527: {len(out)} clips, prompts = bins02 bodies + 0.5 s legend, targets = M3 eval labels")
