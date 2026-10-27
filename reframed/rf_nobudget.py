"""No-budget copies of the REFRAMED budget datasets, same rule as CMD-AD's make_nobudget_qwen38.py:
remove the "## Time budget tokens" legend from each prompt and every <|budget_N|> token from each target.
Same clips, same frames, nothing else changes (checked)."""
import json, re
D = "/scratch/rf_datasets/frame_variants_448x252"
BUDGET_RE = re.compile(r"<\|budget_\d+\|>")
PROMPT_START = "You are an audio description writer."
info = json.load(open(f"{D}/dataset_info.json"))
for src, dst in (("reframed_bins_f150_40k_train", "reframed_nobudget_f150_40k_train"),
                 ("reframed_bins_f150_999k_validation", "reframed_nobudget_f150_999k_validation"),
                 ("reframed_bins_f150_999k_test", "reframed_nobudget_f150_999k_test")):
    data, out, removed = json.load(open(f"{D}/{src}.json")), [], 0
    for r in data:
        u, a = r["messages"][0]["content"], r["messages"][1]["content"]
        i = u.find(PROMPT_START)
        assert u[:i].startswith("## Time budget tokens") and "<|budget_" not in u[i:]
        removed += len(BUDGET_RE.findall(a))
        a2 = BUDGET_RE.sub("", a)
        assert "<|" not in a2
        assert re.sub(r"\s+", " ", a2) == re.sub(r"\s+", " ", BUDGET_RE.sub("", a))
        out.append({**r, "messages": [{**r["messages"][0], "content": u[i:]}, {**r["messages"][1], "content": a2}]})
        assert out[-1]["messages"][0]["content"].count("<image>") == len(r["images"])
    json.dump(out, open(f"{D}/{dst}.json", "w"), ensure_ascii=False)
    info[dst] = dict(info[src], file_name=f"{dst}.json")
    print(dst, len(out), "clips, budget tokens removed:", removed)
json.dump(info, open(f"{D}/dataset_info.json", "w"), indent=2)
