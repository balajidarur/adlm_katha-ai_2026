#!/usr/bin/env python3
"""T1: M3 with the budget written as plain text instead of special tokens.

Same clips, frames, prompt and AD words as M3 (bins_f150_40k / eval527); only the budget marker changes:
  <|budget_k|> (0.5 s grid)  ->  `b{0.5(k+1)}` for k = 0..7 (b0.5 = under 0.5 s left ... b4 = 3.5 to 4 s left),
                                 `b4+` for k = 8 (over 4 s left)
followed by a space, so "[AD] <|budget_8|>George <|budget_7|>gets" becomes "[AD] b4+ George b4 gets".
The tags are ordinary text for the tokenizer (no new special tokens); the legend at the top of the prompt explains them.

  python make_textbins.py --m3 <dir with gapframes_bins_train_f150_40k.json, eval527.json, dataset_info.json> --out <dir>
Writes textbins_f150_40k.json, textbins_eval527.json, dataset_info.json and SHA256SUMS.
"""
import argparse, hashlib, json, os, re

TOK = re.compile(r"<\|budget_(\d)\|>")
TAGS = [f"b{0.5 * (k + 1):g}" for k in range(8)] + ["b4+"]
LABELS = ["under 0.5"] + [f"{0.5 * k:g} to {0.5 * (k + 1):g}" for k in range(1, 8)] + ["over 4"]
LEGEND = ("## Time budget tags\n"
          "Each word of an audio description is preceded by a tag saying how many seconds of silence remain before the "
          "next line of dialogue begins:\n"
          + "\n".join(f"- `{t}` : {l} seconds left" for t, l in zip(TAGS, LABELS))
          + "\n`b0.5` means there is almost no time left - finish the sentence now.\n"
            "Write the tag that matches the time remaining, a space, then the word.")


def convert(r):
    u, a = r["messages"][0]["content"], r["messages"][1]["content"]
    assert u.startswith("## Time budget tokens\n")
    r["messages"][0]["content"] = LEGEND + "\n\n" + u.split("\n\n", 1)[1]
    r["messages"][1]["content"] = TOK.sub(lambda m: TAGS[int(m.group(1))] + " ", a)
    assert "<|budget" not in r["messages"][0]["content"] + r["messages"][1]["content"]
    return r


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(f"{a.m3}/dataset_info.json"))
    tmpl = next(iter(info.values()))
    out_info = {}
    for name, src in (("textbins_f150_40k", "gapframes_bins_train_f150_40k.json"), ("textbins_eval527", "eval527.json")):
        recs = [convert(r) for r in json.load(open(f"{a.m3}/{src}"))]
        json.dump(recs, open(f"{a.out}/{name}.json", "w"), ensure_ascii=False)
        out_info[name] = dict(tmpl, file_name=f"{name}.json")
        print(name, len(recs), "clips", flush=True)
    json.dump(out_info, open(f"{a.out}/dataset_info.json", "w"), indent=2)
    with open(f"{a.out}/SHA256SUMS", "w") as f:
        for fn in ["textbins_f150_40k.json", "textbins_eval527.json", "dataset_info.json"]:
            f.write(f"{sha(f'{a.out}/{fn}')}  {fn}\n")
