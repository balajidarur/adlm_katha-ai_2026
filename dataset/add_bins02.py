#!/usr/bin/env python3
"""0.2 s budget bins: rebuild the Qwen3.8 bins datasets with 26 tokens instead of 9.

Same clips, frames and prompt as bins_f150_40k / eval527 - only the bin grid changes:
  <|budget_0|>  : 0.2 s or less left
  <|budget_k|>  : 0.2k to 0.2(k+1) s left, k = 1..24 (up to 5 s)
  <|budget_25|> : more than 5 s left
The per-word budget is computed exactly as in add_bins_to_gapframes.convert() (same parquet, same lookup);
only BIN_EDGES / BIN_TOKENS / BIN_LEGEND are swapped. The clip set is not re-filtered: every clip of the
0.5 s dataset is rebuilt from its unbinned source record and capped with the same mk150.cap(), so clips and
frames match one for one (asserted).

  python add_bins02.py --q /data/cmd_ad/datasets/frame_variants_448x252_full_qwen38 --out /prep/bins02
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ab = {}
exec(open(f"{HERE}/add_bins_to_gapframes.py").read().split('if __name__ == "__main__":')[0], ab)
mk = {}
exec(open(f"{HERE}/mk150.py").read().split("d = json.load")[0], mk)

N = 25
EDGES = [round(0.2 * (k + 1), 1) for k in range(N)]                 # 0.2 .. 5.0
TOKENS = [f"<|budget_{k}|>" for k in range(N + 1)]
LABELS = ["0.2 or less"] + [f"{EDGES[k - 1]:.1f} to {EDGES[k]:.1f}" for k in range(1, N)] + ["over 5"]
LEGEND = (
    "## Time budget tokens\n"
    "Each word of an audio description is preceded by one token saying how many "
    "seconds of silence remain before the next line of dialogue begins:\n"
    + "\n".join(f"- `{t}` : {l} seconds left" for t, l in zip(TOKENS, LABELS))
    + "\n`<|budget_0|>` means there is almost no time left - finish the sentence now.\n"
      "Emit the token that matches the time remaining, then the word."
)
ab.update(BIN_EDGES=EDGES, BIN_TOKENS=TOKENS, BIN_LEGEND=LEGEND)   # convert()/bin_of() read these globals


def strip(t):
    import re
    return re.sub(r"<\|budget_\d+\|>", "", t)


def key(r):
    """Frames + unbinned assistant text: frames alone are not unique (two clips share a frame list)."""
    return tuple(r["images"]), strip(r["messages"][1]["content"])


def build(src, meta, ref, budget):
    """Rebin every source clip, cap frames, keep exactly the clips of the 0.5 s reference file."""
    data, mt = json.load(open(src)), json.load(open(meta))
    assert len(data) == len(mt)
    ref = json.load(open(ref))
    want = {}
    for i, r in enumerate(ref):
        want.setdefault(key(r), []).append(i)
    out = [None] * len(ref)
    for rec, m in zip(data, mt):
        ab["convert"](rec, m.get("video_id") or m.get("vid"), budget)
        mk["cap"](rec)
        for i in want.get(key(rec), []):
            out[i] = rec
    miss = sum(r is None for r in out)
    assert miss == 0, f"{miss} reference clips not rebuilt"
    for a, b in zip(out, ref):     # same frames, same prompt apart from the legend, same AD words
        assert a["images"] == b["images"]
        assert a["messages"][0]["content"].split("\n\n", 1)[1] == b["messages"][0]["content"].split("\n\n", 1)[1]
        assert strip(a["messages"][1]["content"]) == strip(b["messages"][1]["content"])
    return out


def bin_stats(recs):
    import re
    c = [0] * (N + 1)
    for r in recs:
        for k in re.findall(r"<\|budget_(\d+)\|>", r["messages"][1]["content"]):
            c[int(k)] += 1
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = ab["load_budget"]()
    for name, src, ref in (("bins02_f150_40k", "cmdad_dds_gapframes_train", "gapframes_bins_train_f150_40k.json"),
                           ("bins02_eval527", "cmdad_dds_gapframes_eval", "eval527.json")):
        recs = build(f"{a.q}/{src}.json", f"{a.q}/{src}_meta.json", f"{a.q}/{ref}", budget)
        json.dump(recs, open(f"{a.out}/{name}.json", "w"), ensure_ascii=False)
        print(name, len(recs), "clips | tokens per bin:", bin_stats(recs), flush=True)
    info = json.load(open(f"{a.q}/dataset_info.json"))
    json.dump({"bins02_f150_40k": dict(info["bins_f150_40k"], file_name="bins02_f150_40k.json"),
               "bins02_eval527": dict(info["eval527"], file_name="bins02_eval527.json")},
              open(f"{a.out}/dataset_info.json", "w"), indent=2)
