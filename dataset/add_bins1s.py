#!/usr/bin/env python3
"""G1: 1 s budget bins - rebuild the Qwen3.8 bins datasets (M3) with 5 tokens instead of 9.

Same clips, frames and prompt as bins_f150_40k / eval527 - only the bin grid changes:
  <|budget_0|> : under 1 s left,  <|budget_1|> : 1 to 2 s,  <|budget_2|> : 2 to 3 s,  <|budget_3|> : 3 to 4 s,
  <|budget_4|> : over 4 s left
Reuses add_bins02.py unchanged (same parquet, same per-word lookup, same mk150.cap(), same one-to-one clip/frame asserts);
only its grid globals are swapped.

  python add_bins1s.py --q /data/cmd_ad/datasets/frame_variants_448x252_full_qwen38 --out /prep/bins1s
Writes bins1s_f150_40k.json, bins1s_eval527.json, dataset_info.json.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
g = {"__file__": f"{HERE}/add_bins02.py"}
exec(open(f"{HERE}/add_bins02.py").read().split('\nif __name__ == "__main__":')[0], g)

N = 4
EDGES = [1.0, 2.0, 3.0, 4.0]
TOKENS = [f"<|budget_{k}|>" for k in range(N + 1)]
LABELS = ["under 1", "1 to 2", "2 to 3", "3 to 4", "over 4"]
LEGEND = ("## Time budget tokens\n"
          "Each word of an audio description is preceded by one token saying how many "
          "seconds of silence remain before the next line of dialogue begins:\n"
          + "\n".join(f"- `{t}` : {l} seconds left" for t, l in zip(TOKENS, LABELS))
          + "\n`<|budget_0|>` means there is almost no time left - finish the sentence now.\n"
            "Emit the token that matches the time remaining, then the word.")
g.update(N=N, EDGES=EDGES, TOKENS=TOKENS, LABELS=LABELS, LEGEND=LEGEND)
g["ab"].update(BIN_EDGES=EDGES, BIN_TOKENS=TOKENS, BIN_LEGEND=LEGEND)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g["BUDGET"] = budget = g["ab"]["load_budget"]()
    for name, src, ref in (("bins1s_f150_40k", "cmdad_dds_gapframes_train", "gapframes_bins_train_f150_40k.json"),
                           ("bins1s_eval527", None, "eval527.json")):
        if os.path.exists(f"{a.out}/{name}.json"):
            print(name, "exists, skipped"); continue
        recs = (g["build"](f"{a.q}/{src}.json", f"{a.q}/{src}_meta.json", f"{a.q}/{ref}", budget) if src
                else g["rebin_ref"](f"{a.q}/{ref}"))
        json.dump(recs, open(f"{a.out}/{name}.json", "w"), ensure_ascii=False)
        print(name, len(recs), "clips | tokens per bin:", g["bin_stats"](recs), flush=True)
    info = json.load(open(f"{a.q}/dataset_info.json"))
    json.dump({"bins1s_f150_40k": dict(info["bins_f150_40k"], file_name="bins1s_f150_40k.json"),
               "bins1s_eval527": dict(info["eval527"], file_name="bins1s_eval527.json")},
              open(f"{a.out}/dataset_info.json", "w"), indent=2)
