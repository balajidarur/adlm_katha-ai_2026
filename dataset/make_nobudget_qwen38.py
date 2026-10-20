"""Build no-budget copies of the bins_frames datasets and verify them.

Removes (1) the "## Time budget tokens" legend at the top of every user prompt and
(2) every <|budget_N|> token in the assistant targets. Nothing else may change;
verify() checks that field by field and writes a JSON report for the HTML page.
"""
import json, os, re, sys, random

SRC = "/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38"
DST = "/data/cmd_ad/datasets/frame_variants_448x252_full_qwen38/nobudget"
TOK_DIR = ""
FILES = {  # dataset name -> (source file, output file)
    "nobudget_f150_40k": ("gapframes_bins_train_f150_40k.json", "gapframes_nobudget_train_f150_40k.json"),
    "nobudget_16": ("sixteen.json", "sixteen_nobudget.json"),
    "nobudget_eval": ("gapframes_bins_eval.json", "gapframes_nobudget_eval.json"),
}
BUDGET_RE = re.compile(r"<\|budget_\d+\|>")
PROMPT_START = "You are an audio description writer."
AD_RE = re.compile(r"^(\d+:\d\d\.\d) \[AD\] (.*)$")

os.makedirs(DST, exist_ok=True)
src_info = json.load(open(f"{SRC}/dataset_info.json"))
template = src_info["bins_frames_eval"]
legend = None
report = {"files": {}, "examples": []}

tok = None
if TOK_DIR:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    tok.add_special_tokens({"additional_special_tokens": [f"<|budget_{i}|>" for i in range(9)]})


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"]) if tok else 0


info = {}
for name, (src_fn, dst_fn) in FILES.items():
    data = json.load(open(f"{SRC}/{src_fn}"))
    out, stats = [], {"samples": len(data), "budget_tokens_removed": 0, "ad_lines": 0,
                      "ad_lines_with_budget": 0, "dialogue_lines": 0, "images": 0,
                      "user_chars_before": 0, "user_chars_after": 0,
                      "asst_chars_before": 0, "asst_chars_after": 0,
                      "text_tokens_before": 0, "text_tokens_after": 0, "errors": []}
    for k, r in enumerate(data):
        msgs = []
        for m in r["messages"]:
            c = m["content"]
            if m["role"] == "user":
                i = c.find(PROMPT_START)
                head = c[:i]
                if legend is None:
                    legend = head
                if head != legend or "<|budget_" in c[i:]:
                    stats["errors"].append(f"{k}: unexpected prompt header")
                new = c[i:]
                stats["user_chars_before"] += len(c); stats["user_chars_after"] += len(new)
            else:
                n = len(BUDGET_RE.findall(c))
                stats["budget_tokens_removed"] += n
                new = BUDGET_RE.sub("", c)
                stats["asst_chars_before"] += len(c); stats["asst_chars_after"] += len(new)
                for line in c.split("\n"):
                    if AD_RE.match(line):
                        stats["ad_lines"] += 1
                        stats["ad_lines_with_budget"] += bool(BUDGET_RE.search(line))
                    elif "[DIALOGUE]" in line:
                        stats["dialogue_lines"] += 1
                if "<|" in new:
                    stats["errors"].append(f"{k}: leftover special token")
            if tok and k < 400:  # token deltas on a sample of 400 per file
                stats["text_tokens_before"] += ntok(c); stats["text_tokens_after"] += ntok(new)
            msgs.append({**m, "content": new})
        stats["images"] += len(r.get("images", []))
        nr = {**r, "messages": msgs}
        out.append(nr)
        if name == "nobudget_f150_40k" and len(report["examples"]) < 3 and k in (0, 1000, 5000):
            a0 = [m["content"] for m in r["messages"] if m["role"] == "assistant"][0]
            a1 = [m["content"] for m in msgs if m["role"] == "assistant"][0]
            ads0 = [l for l in a0.split("\n") if "[AD]" in l][:4]
            ads1 = [l for l in a1.split("\n") if "[AD]" in l][:4]
            report["examples"].append({"index": k, "before": ads0, "after": ads1})
    # independent re-check of the written data
    for k, (a, b) in enumerate(zip(data, out)):
        if a.get("images") != b.get("images"):
            stats["errors"].append(f"{k}: images differ")
        if [m["role"] for m in a["messages"]] != [m["role"] for m in b["messages"]]:
            stats["errors"].append(f"{k}: roles differ")
        for ma, mb in zip(a["messages"], b["messages"]):
            if "budget_" in mb["content"] and "<|budget_" in mb["content"]:
                stats["errors"].append(f"{k}: budget token left")
            if ma["role"] == "assistant":
                la, lb = ma["content"].split("\n"), mb["content"].split("\n")
                if len(la) != len(lb):
                    stats["errors"].append(f"{k}: line count differs")
                for x, y in zip(la, lb):
                    if BUDGET_RE.sub("", x) != y:
                        stats["errors"].append(f"{k}: line text differs")
                    if " ".join(BUDGET_RE.sub(" ", x).split()) != " ".join(y.split()):
                        stats["errors"].append(f"{k}: word sequence differs")
            else:
                if not ma["content"].endswith(mb["content"]):
                    stats["errors"].append(f"{k}: prompt body changed")
    json.dump(out, open(f"{DST}/{dst_fn}", "w"), ensure_ascii=False)
    info[name] = {**template, "file_name": dst_fn}
    stats["errors"] = stats["errors"][:20]
    stats["error_count"] = len(stats["errors"])
    report["files"][name] = {"source": f"{SRC}/{src_fn}", "output": f"{DST}/{dst_fn}", **stats}
    print(name, json.dumps({k: v for k, v in stats.items() if k != "errors"}), "errors:", stats["errors"][:3], flush=True)

json.dump(info, open(f"{DST}/dataset_info.json", "w"), indent=2)
report["legend_removed"] = legend
report["prompt_start_kept"] = PROMPT_START
json.dump(report, open(f"{DST}/nobudget_report.json", "w"), indent=1, ensure_ascii=False)
print("wrote", DST)
