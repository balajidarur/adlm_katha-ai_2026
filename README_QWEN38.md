# Qwen3.8 dense descriptions → frame-interleaved ADLM datasets → LLaMA-Factory training

This branch keeps everything used for the Qwen3.8-27B dense-description run and the datasets/training built
on it, side by side with the Gemini 3.5 Flash Lite equivalents, so every training run can be traced to the
exact dataset and code that produced it.

```
dense_qwen38/            Qwen3.8-27B-FP8 dense descriptions (character bank + face circles)
  prep_charbank.py         pre-VLM prep: shots (Gemini cache), 3 FPS 360p frames, InsightFace, circles
  charbank_pipeline.py     per-clip sequential shot calls to vLLM (--prompt-variant, --no-prop-loc-tags)
  prompts_adlm.py          base prompt (shared with the Gemini pipeline)
  prompts_adlm_charbank.py cast/face-circle section, writing-style variants, circle-leak filter
  run_dense_all.py         many clips x many vLLM servers driver
  eval_entailment.py       GT-AD visual-claim entailment (Gemini claims + DeBERTa-xlarge-MNLI)
  prompt_eval.sh, periodic_eval.sh, build_*_report.py   prompt study + reports
  vllm/                    serve / env / setup scripts (vLLM 0.31.0, TRITON_ATTN, MTP, 1 server per GPU)
  ops/                     multi-box sharding, slim data copy, progress/status, finish+shutdown
  PREP_LAYOUT.md           layout of the prep output (/prep/prep_360p on box A)
dataset/                 frame-interleaved dataset build (was ~/adlm/dataset_prep + ~/adlm/*.py on box A)
  build_frame_datasets.py  variants A/B/C; DENSE_PATH_TMPL env switches the dense source (default Gemini)
  frames_lib.py            sampling rules (1 FPS 448x252, gaps >=1 s padded ±25%, cap, token budget)
  run_full_build.sh        Gemini build;  run_qwen_build.sh  Qwen3.8 build (identical args)
  add_bins_to_gapframes.py budget-bin tokens on top of variant B
  mk150.py / mk40k.py      150-frame cap + 40k token filter  (mk40k_qwen38.py = same, Qwen paths)
  make_nobudget.py         strip legend + <|budget_N|>        (make_nobudget_qwen38.py = same, Qwen paths)
training/
  LLaMA-Factory/           the patched fork used on the AWS boxes (streamed predictions, budget tokens,
                           restrict_new_token_grad, ...). Configs in examples/budget_ad/
  launch/                  run_* / start_* (tmux) / watchdog_* (auto-resume) scripts
```

## Dense-description runs

| run | model | prompt | clips | where |
|---|---|---|---|---|
| Gemini | gemini-3.5-flash-lite | prompts_adlm.py | 8,613 | `/data/cmd_ad/dense/{vid}/gemini-3.5-flash-lite/` |
| Qwen3.8 v2 | Qwen3.8-27B-FP8 (vLLM) | `--prompt-variant desc_action_v2` (charbank + writing style) | 8,553 | box A `/prep/qwen_dense_final/{vid}/Qwen3.8-27B-FP8/` |

Both are archived in `s3://balu-transfer-933516006154-apne1/cmd_ad_dense/` (tar.gz) with the code.

Prompt study (20 clips, visual-claim entailment of GT ADs): Gemini 38.4%, Qwen base prompt 23.3%,
desc_action 39.4%, **desc_action_v2 38.7% (used)**, v5 (char-only tags, <25 words) 37.7%.

## Datasets and the configs that train on them

Build rules are identical for both dense sources: 1 FPS 448x252 frame pool (`/data/cmd_ad/frames/1fps_448x252`),
variant B = speech-free gaps >=1 s padded ±25%, cap 200 frames, 48k token budget; then budget bins;
then 150-frame cap + 40k token filter for training.

| dataset (LLaMA-Factory name) | dense | budget tokens | file | clips / frames | config | box |
|---|---|---|---|---|---|---|
| `bins_f150_40k` | Gemini | yes | `bins_frames/gapframes_bins_train_f150_40k.json` | 7,395 / 617,234 | `bins_frames_full.yaml` | A (done, ckpt 696) |
| `nobudget_f150_40k` | Gemini | no | `nobudget_frames/gapframes_nobudget_train_f150_40k.json` | 7,395 | `nobudget_frames_full.yaml` | C (done, ckpt 696) |
| `bins_f150_40k` | **Qwen3.8** | yes | `frame_variants_448x252_full_qwen38/gapframes_bins_train_f150_40k.json` | 7,763 / 668,930 | `qwen38_bins_frames_full.yaml` | A |
| `nobudget_f150_40k` | **Qwen3.8** | no | `frame_variants_448x252_full_qwen38/nobudget/gapframes_nobudget_train_f150_40k.json` | 7,763 / 668,930 | `qwen38_nobudget_frames_full.yaml` | C |
| `bins_frames_16` | Qwen3.8 | yes | `.../sixteen.json` (same 16 eval clips as Gemini, 1,388 frames) | 16 | `qwen38_bins_frames_16.yaml` (overfit) | A |
| `nobudget_16` | Qwen3.8 | no | `.../nobudget/sixteen_nobudget.json` | 16 | `qwen38_nobudget_frames_16.yaml` (overfit) | C |

Paths are under `/data/cmd_ad/datasets/` on boxes A and C. Qwen keeps more clips under the 40k filter
(7,763 vs 7,395) because its descriptions are shorter; all frame-selection rules are unchanged.

Order of work for every new dataset: **overfit the 16-clip set first**, then the full run.

## Reproduce

```bash
# Qwen dataset (box A)
bash dataset/run_qwen_build.sh                      # A/B/C variants with Qwen dense
python dataset/add_bins_to_gapframes.py --src $Q/cmdad_dds_gapframes_train.json \
    --meta $Q/cmdad_dds_gapframes_train_meta.json --out $Q/gapframes_bins_train.json   # (and eval)
python dataset/mk40k_qwen38.py                     # 150-frame cap + 40k filter
python dataset/make_nobudget_qwen38.py             # no-budget copies (train, 16, eval)
# training
bash training/launch/run_qwen38_bins_frames_16.sh  # overfit check, then start_qwen38.sh for the full run
```
