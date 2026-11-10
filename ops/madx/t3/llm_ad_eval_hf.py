#!/usr/bin/env python3
"""LLM-AD-Eval (AutoAD-III) re-implemented with HF transformers instead of the Video-LLaMA wrapper.

Prompt text, sampling (do_sample, top_p 0.9, temperature 1.0, max_new_tokens 8) and score parsing
(ast.literal_eval of "{'score': k}", parse failure -> 0) are copied from AutoAD/autoad_iii/metrics/llm_ad_eval_llama.py.
Llama-2: the official "[INST] <<SYS>> ... <</SYS>> ... [/INST]" string, no BOS added (add_special_tokens=False, as official).
Llama-3: same system / user text through the Llama-3 chat template (the official repo has no Llama-3 script).
Weights: ungated mirrors of meta-llama/Llama-2-7b-chat-hf and Meta-Llama-3-8B-Instruct (identical files).

  python llm_ad_eval_hf.py --csv preds_tf.csv --model llama2|llama3 --out out.json [--seed 0]
"""
import argparse, ast, json

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


def user(gt, pred):
    return ("Please evaluate the following movie audio description pair:\n\n"
            f"Correct Audio Description: {gt}\n"
            f"Predicted Audio Description: {pred}\n\n"
            "Provide your evaluation only as a matching score where the matching score is an integer value between 0 and 5, with 5 indicating the highest level of match. "
            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the matching score in INTEGER, not STRING."
            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
            "For example, your response should look like this: {'score': }.")


MODELS = {"llama2": "NousResearch/Llama-2-7b-chat-hf", "llama3": "NousResearch/Meta-Llama-3-8B-Instruct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    df = pd.read_csv(a.csv, keep_default_na=False)
    tok = AutoTokenizer.from_pretrained(MODELS[a.model]); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.float16, device_map="cuda").eval()
    except TypeError:   # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], torch_dtype=torch.float16, device_map="cuda").eval()
    if a.model == "llama2":
        prompts = [L2.format(user(g, p)) for g, p in zip(df.text_gt, df.text_gen)]
        enc_kw = dict(add_special_tokens=False)
    else:
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user(g, p)}],
                                           tokenize=False, add_generation_prompt=True) for g, p in zip(df.text_gt, df.text_gen)]
        enc_kw = dict(add_special_tokens=False)
    eos = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|eot_id|>")] if a.model == "llama3" else [])
    outs = []
    for i in range(0, len(prompts), a.bs):
        enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True, **enc_kw).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=8, do_sample=True, top_p=0.9, temperature=1.0, eos_token_id=eos, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    scores, fails = [], 0
    for o in outs:
        try:
            scores.append(float(ast.literal_eval(o.replace("<s>", "").replace("</s>", "").strip())["score"]))
        except Exception:
            scores.append(0.0); fails += 1
    res = {"model": MODELS[a.model], "n": len(scores), "mean": round(float(np.mean(scores)), 3), "parse_fail": fails, "seed": a.seed}
    json.dump({**res, "raw": outs}, open(a.out, "w"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
