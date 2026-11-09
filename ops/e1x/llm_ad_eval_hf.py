#!/usr/bin/env python3
"""LLM-AD-Eval (AutoAD-III) with HF transformers, with the prompt variants needed to calibrate against published CMD-AD numbers.

Base: tasks_nov8/T3/code/llm_ad_eval_hf.py (T3). Prompt text, sampling (do_sample, top_p 0.9, temperature 1.0,
max_new_tokens 8) and parsing (ast.literal_eval of "{'score': k}", failure -> 0) are copied from
AutoAD/autoad_iii/metrics/llm_ad_eval_llama.py.

Variants (--variant):
  l2            Llama-2-7b-chat, official AutoAD-III "[INST] <<SYS>> ... [/INST]" string, no BOS (as official).
  l2_lenient    same + AutoAD-Zero's score-adapter sentence.
  l3_chat       Llama-3-8B-Instruct, system=SYS / user=prompt through the Llama-3 chat template (= T3).
  l3_chat_lenient  l3_chat + score adapter  (AutoAD-Zero App. Alg. 3: "For LLaMA3-8B: score_adapter =
                'Be generous at scoring, and focus on matching of sentence meaning.'"; for LLaMA2-7B it is empty).
  l3_zero_lenient  literal AutoAD-Zero Alg. 3 messages: system = the AutoAD-III "[INST] <<SYS>> ... {} [/INST]" string
                verbatim, user = prompt with score adapter, through the Llama-3 chat template.
The adapter is inserted after "... with 5 indicating the highest level of match. " (its position in Alg. 3).

  python llm_ad_eval_hf.py --csv a.csv [b.csv ...] --variant l3_chat_lenient --out_dir DIR [--seed 0]
  -> DIR/<csv stem>__<variant>.json ; each csv needs columns text_gt, text_gen. The model is loaded once.
"""
import argparse, ast, json, os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = ("You are an intelligent chatbot designed for evaluating the quality of generative outputs for movie audio descriptions. "
       "Your task is to compare the predicted audio descriptions with the correct audio descriptions and determine its level of match, "
       "considering mainly the visual elements like actions, objects and interactions. Here's how you can accomplish the task:------"
       "##INSTRUCTIONS: - Check if the predicted audio description covers the main visual events from the movie, especially focusing on the verbs and nouns.\n"
       "- Evaluate whether the predicted audio description includes specific details rather than just generic points. It should provide comprehensive "
       "information that is tied to specific elements of the video.\n"
       "- Consider synonyms or paraphrases as valid matches. Consider pronouns like 'he' or 'she' as valid matches with character names. "
       "Consider different character names as valid matches. \n"
       "- Provide a single evaluation score that reflects the level of match of the prediction, considering the visual elements like actions, objects and interactions. \n")
L2 = "[INST] <<SYS>>\n" + SYS + "<</SYS>>\n\n{} [/INST] "
ADAPTER = "Be generous at scoring, and focus on matching of sentence meaning. "


def user(gt, pred, adapter=""):
    return ("Please evaluate the following movie audio description pair:\n\n"
            f"Correct Audio Description: {gt}\n"
            f"Predicted Audio Description: {pred}\n\n"
            "Provide your evaluation only as a matching score where the matching score is an integer value between 0 and 5, with 5 indicating the highest level of match. "
            + adapter +
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the matching score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'score': }.")


MODELS = {"l2": "NousResearch/Llama-2-7b-chat-hf", "l3": "NousResearch/Meta-Llama-3-8B-Instruct"}
VARIANTS = ["l2", "l2_lenient", "l3_chat", "l3_chat_lenient", "l3_zero_lenient"]


def build_prompts(variant, tok, gts, preds):
    ad = ADAPTER if variant.endswith("lenient") else ""
    if variant.startswith("l2"):
        return [L2.format(user(g, p, ad)) for g, p in zip(gts, preds)]
    sysmsg = L2 if variant.startswith("l3_zero") else SYS
    return [tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": user(g, p, ad)}],
                                    tokenize=False, add_generation_prompt=True) for g, p in zip(gts, preds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True); ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--model_dir", default=None, help="local snapshot dir (default: HF id)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    mid = a.model_dir or MODELS[a.variant[:2]]
    tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if a.variant.startswith("l3") else [])
    for csv in a.csv:
        out = os.path.join(a.out_dir, f"{os.path.basename(csv)[:-4]}__{a.variant}.json")
        if os.path.exists(out):
            print("exists", out); continue
        torch.manual_seed(a.seed)
        df = pd.read_csv(csv, keep_default_na=False)
        prompts = build_prompts(a.variant, tok, df.text_gt.tolist(), df.text_gen.tolist())
        order = np.argsort([len(p) for p in prompts], kind="stable")   # length-sorted batches; results put back in row order
        outs = [None] * len(prompts)
        for i in range(0, len(prompts), a.bs):
            ids = order[i:i + a.bs]
            enc = tok([prompts[j] for j in ids], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=8, do_sample=True, top_p=0.9, temperature=1.0, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for j, o in zip(ids, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                outs[j] = o
        scores, fails = [], 0
        for o in outs:
            try:
                scores.append(float(ast.literal_eval(o.replace("<s>", "").replace("</s>", "").strip())["score"]))
            except Exception:
                scores.append(0.0); fails += 1
        res = {"model": MODELS[a.variant[:2]], "variant": a.variant, "csv": csv, "n": len(scores), "mean": round(float(np.mean(scores)), 4),
               "parse_fail": fails, "seed": a.seed, "example_prompt": prompts[0]}
        json.dump({**res, "scores": scores, "raw": outs}, open(out, "w"))
        print(json.dumps({k: v for k, v in res.items() if k != "example_prompt"}), flush=True)


if __name__ == "__main__":
    main()
