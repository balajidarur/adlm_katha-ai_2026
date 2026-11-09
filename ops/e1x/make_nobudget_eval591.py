#!/usr/bin/env python3
"""eval591 for the no-budget model (qwen38-dd_nobudget ckpt-480): the exact transform of dataset/make_nobudget_qwen38.py
(drop the "## Time budget tokens" legend before "You are an audio description writer." in the user turn; delete every
<|budget_N|> in the assistant turn; images untouched) applied to s3 adlm/datasets/e1_eval591/eval591.json (md5 2ea61c05).
Check: the assistant targets of rows 0-526 must equal, position for position, the labels of the original eval527 no-budget
TF run (s3 adlm/inference/cmdad/eval527/qwen38-dd_nobudget_ckpt480/teacher_forced/raw_predictions.jsonl, trailing newline
stripped).
  python make_nobudget_eval591.py eval591.json raw_predictions.jsonl out.json
"""
import json, re, sys

BUDGET_RE = re.compile(r"<\|budget_\d+\|>")
PROMPT_START = "You are an audio description writer."
src, raw, out = sys.argv[1:4]
d = json.load(open(src))
nb, legend = [], None
for r in d:
    ms = []
    for m in r["messages"]:
        c = m["content"]
        if m["role"] == "user":
            i = c.index(PROMPT_START); head = c[:i]
            legend = legend or head
            assert head == legend and "<|budget_" not in c[i:]
            c = c[i:]
        else:
            c = BUDGET_RE.sub("", c); assert "<|" not in c
        ms.append({**m, "content": c})
    nb.append({**r, "messages": ms})
lab = {int(x["idx"]): x["label"].rstrip("\n") for x in map(json.loads, open(raw))}
A = [[m["content"] for m in r["messages"] if m["role"] == "assistant"][0].rstrip("\n") for r in nb]
same = sum(A[i] == lab[i] for i in range(527))
print(f"rows {len(nb)} | eval527 targets identical to original no-budget run: {same}/527")
assert same == 527
json.dump(nb, open(out, "w"), ensure_ascii=False)
