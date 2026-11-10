#!/usr/bin/env python3
"""LLM-AD-Eval through a vLLM OpenAI-compatible server: same prompts, sampling and parsing as llm_ad_eval_hf.py.

Only the backend changes (HF transformers fp16 -> vLLM fp16). The prompt strings come from llm_ad_eval_hf.build_prompts
(imported, not copied), so they are identical byte for byte:
  l2*   the AutoAD-III "[INST] <<SYS>> ... [/INST] " string, sent to /v1/completions with add_special_tokens=False (no BOS,
        as the HF scorer and the official script).
  l3*   the Llama-3 chat template rendered locally by the judge's tokenizer (it already contains <|begin_of_text|>), sent to
        /v1/completions with add_special_tokens=False (no double BOS).
Sampling: temperature 1.0, top_p 0.9, max_tokens 8, plus top_k 50, which HF generate applies implicitly (GenerationConfig
default) and vLLM does not. Stop ids: eos (+ <|eot_id|> for Llama-3). Output decoded with skip_special_tokens. Parsing:
ast.literal_eval of the stripped output, ["score"], failure -> 0.  Per-request seed = --seed * 1_000_003 + row index.

Diagnostics (not the E1X protocol): --bos (add BOS for l2), --top_k -1 (vLLM default), --endpoint chat (let the server
apply the judge's own chat template to messages [system, user]; for l2 system=SYS, the way a plain vLLM chat client would).

  # server:  vllm serve <judge dir> --dtype float16 --generation-config vllm --port 8000 --max-model-len 4096
  python llm_ad_eval_vllm.py --csv a.csv --variant l2 --model_dir <judge dir> --out_dir DIR [--seed 0] [--port 8000]
           [--first_clips 50] [--tag s0]
  -> DIR/<csv stem>[__first50]__<variant>[__tag].json   (same keys as the HF scorer + backend info)
"""
import argparse, ast, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_ad_eval_hf import ADAPTER, MODELS, SYS, VARIANTS, build_prompts, user  # noqa: E402


def parse(o):
    try:
        return float(ast.literal_eval(o.replace("<s>", "").replace("</s>", "").strip())["score"]), 0
    except Exception:
        return 0.0, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_dir", required=True, help="judge snapshot dir (also the served model name)")
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--first_clips", type=int, default=0, help="only the first N clips (cmd_filename) in CSV order")
    ap.add_argument("--top_k", type=int, default=50); ap.add_argument("--bos", action="store_true")
    ap.add_argument("--endpoint", choices=["completions", "chat"], default="completions"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    stop_ids = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if a.variant.startswith("l3") else [])
    base = f"http://127.0.0.1:{a.port}/v1"
    served = requests.get(f"{base}/models", timeout=30).json()["data"][0]["id"]
    vver = requests.get(f"http://127.0.0.1:{a.port}/version", timeout=30).json().get("version")
    sess = requests.Session(); sess.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=a.workers))
    samp = dict(max_tokens=8, temperature=1.0, top_p=0.9, top_k=a.top_k, stop_token_ids=stop_ids, skip_special_tokens=True,
                n=1)
    for csv in a.csv:
        stem = os.path.basename(csv)[:-4] + (f"__first{a.first_clips}" if a.first_clips else "")
        out = os.path.join(a.out_dir, f"{stem}__{a.variant}" + (f"__{a.tag}" if a.tag else "") + ".json")
        if os.path.exists(out):
            print("exists", out); continue
        df = pd.read_csv(csv, keep_default_na=False)
        rows = np.arange(len(df))
        if a.first_clips:
            keep = list(dict.fromkeys(df.cmd_filename))[:a.first_clips]
            rows = np.flatnonzero(df.cmd_filename.isin(keep).values); df = df.iloc[rows].reset_index(drop=True)
        gts, preds = df.text_gt.tolist(), df.text_gen.tolist()
        if a.endpoint == "completions":
            prompts = build_prompts(a.variant, tok, gts, preds)
            if a.bos:
                prompts = [tok.bos_token + p for p in prompts]
        else:
            ad = ADAPTER if a.variant.endswith("lenient") else ""
            from llm_ad_eval_hf import L2
            sysmsg = L2 if a.variant.startswith("l3_zero") else SYS
            prompts = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": user(g, p, ad)}] for g, p in zip(gts, preds)]

        def one(i):
            body = dict(model=served, seed=a.seed * 1_000_003 + int(i), **samp)
            if a.endpoint == "completions":
                body.update(prompt=prompts[i], add_special_tokens=False)
                r = sess.post(f"{base}/completions", json=body, timeout=600); r.raise_for_status()
                return r.json()["choices"][0]["text"]
            body.update(messages=prompts[i])
            r = sess.post(f"{base}/chat/completions", json=body, timeout=600); r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""

        with ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(one, range(len(prompts))))
        sc = [parse(o) for o in outs]
        scores = [s for s, _ in sc]; fails = sum(f for _, f in sc)
        ex_prompt = prompts[0] if a.endpoint == "completions" else json.dumps(prompts[0])
        res = {"model": MODELS[a.variant[:2]], "variant": a.variant, "csv": csv, "n": len(scores),
               "mean": round(float(np.mean(scores)), 4), "parse_fail": fails, "seed": a.seed, "backend": "vllm",
               "vllm_version": vver, "endpoint": a.endpoint, "bos": a.bos, "top_k": a.top_k, "first_clips": a.first_clips,
               "example_prompt": ex_prompt}
        json.dump({**res, "rows": rows.tolist(), "scores": scores, "raw": outs}, open(out, "w"))
        print(json.dumps({k: v for k, v in res.items() if k != "example_prompt"}), flush=True)


if __name__ == "__main__":
    main()
