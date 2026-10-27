"""Pre-training checks for REFRAMED budget training on box C: make top-5 (heaviest) and random-5 subsets,
write the probe / overfit / inference yamls from the real training yaml (same recipe)."""
import json, random, re
D = "/scratch/rf_datasets/frame_variants_448x252"
LF = "/home/ubuntu/adlm/LLaMA-Factory/examples/budget_ad"
tr = json.load(open(f"{D}/reframed_bins_f150_40k_train.json"))
est = lambda r: 112 * len(r["images"]) + (len(r["messages"][0]["content"]) + len(r["messages"][1]["content"])) / 3.6
top5 = sorted(tr, key=est, reverse=True)[:5]
ovf5 = random.Random(0).sample(tr, 5)
info = json.load(open(f"{D}/dataset_info.json"))
for name, recs in (("reframed_top5", top5), ("reframed_ovf5", ovf5)):
    json.dump(recs, open(f"{D}/{name}.json", "w"), ensure_ascii=False)
    info[name] = dict(info["reframed_bins_f150_40k_train"], file_name=f"{name}.json")
    print(name, [round(est(r)) for r in recs], [len(r["images"]) for r in recs])
json.dump(info, open(f"{D}/dataset_info.json", "w"), indent=2)

base = open(f"{LF}/reframed_qwen38_bins_frames.yaml").read()
def train_yaml(name, dataset, out, epochs, ga):
    s = base.replace("dataset: reframed_bins_f150_40k_train", f"dataset: {dataset}")
    s = s.replace("tokenized_path: /scratch/tokcache/reframed_bins_f150_40k\n", "")
    s = s.replace("output_dir: /scratch/runs/reframed_qwen38_bins_frames", f"output_dir: {out}")
    s = s.replace("num_train_epochs: 3.0", f"num_train_epochs: {epochs}")
    s = s.replace("gradient_accumulation_steps: 8", f"gradient_accumulation_steps: {ga}")
    s = s.replace("logging_steps: 5", "logging_steps: 1").replace("report_to: wandb", "report_to: none")
    s = s.replace("save_strategy: epoch", "save_strategy: \"no\"").replace("overwrite_output_dir: false", "overwrite_output_dir: true")
    s = s.replace("train_sampling_strategy: group_by_length\n", "")
    open(f"{LF}/{name}.yaml", "w").write(s + "save_only_model: true\n" if False else s)
train_yaml("reframed_probe_top5", "reframed_top5", "/scratch/runs/probe_top5", "1.0", 1)
train_yaml("reframed_overfit5", "reframed_ovf5", "/scratch/runs/overfit5", "30.0", 1)
for mode in ("tf", "free"):
    lines = [
        "model_name_or_path: Qwen/Qwen3.5-4B", "adapter_name_or_path: /scratch/runs/overfit5", "trust_remote_code: true",
        "image_max_pixels: 589824", "image_min_pixels: 1024", "enable_liger_kernel: true",
        "new_special_tokens_config: examples/budget_ad/bin_tokens.yaml", "init_special_tokens: desc_init", "resize_vocab: false",
        "stage: sft", "do_predict: true", "finetuning_type: lora", "budget_variant: none",
        f"dataset_dir: {D}", "eval_dataset: reframed_ovf5", "template: qwen3_5_nothink", "cutoff_len: 49152",
        "overwrite_cache: true", f"output_dir: /scratch/runs/overfit5_pred_{mode}", "overwrite_output_dir: true",
        "report_to: none", "per_device_eval_batch_size: 1", "bf16: true"]
    lines += (["budget_teacher_forced: true", "budget_prefill_chunk: 2048"] if mode == "tf" else
              ["predict_with_generate: true", "max_new_tokens: 6144", "do_sample: false", "skip_special_tokens: false"])
    open(f"{LF}/reframed_overfit5_{mode}.yaml", "w").write("\n".join(lines) + "\n")
print("yamls written")
