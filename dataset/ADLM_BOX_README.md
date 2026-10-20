# ADLM — audio description generation from CMD-AD clips

## Experiment 3 (current): do video frames help, on top of / instead of dense descriptions?

Three datasets, same 8,552 clips, same target (dialogue verbatim + ADs, in time order):

| dataset | input | frames |
|---|---|---|
| `cmdad_frames_only`   | frames + dialogue                      | 1 FPS over the whole clip |
| `cmdad_dds_gapframes` | dense descriptions + dialogue + frames | 1 FPS inside the dialogue gaps only |
| `cmdad_dds_frames`    | dense descriptions + dialogue + frames | 1 FPS over the whole clip |

Data, build method and all measurements: **`dataset_prep/DATASET_PLAN.md`** (§9 = paths and layout).

## Layout

    configs/        exp3_frames_only_4b.yaml (train) · _probe.yaml (5 steps) · _test.yaml (predict)
                    archive/ — configs of experiments 1-2
    scripts/        run_exp3_frames_only.sh [cfg] [log]      · run_exp3_frames_only_test.sh <ckpt>
    logs/           this experiment's logs
    dataset_prep/   dataset build scripts, DATASET_PLAN.md, build log
    notebooks/      frame_sampling_analysis.ipynb · token_distribution_analysis.ipynb
    runs -> /data/cmd_ad/runs        experiment outputs (checkpoints, predictions)
    saves/ tokcache/ wandb/          experiments 1-2 outputs and caches
    tmp/            archived logs, scripts, old run outputs, tarballs
    LLaMA-Factory/ venv/ hf_cache/ ad_interleave_dataset/ lf_qwen35/

## Run

    tmux new -s exp3                                   # detach: Ctrl-b d
    bash ~/adlm/scripts/run_exp3_frames_only.sh        # 4 GPUs, logs to logs/train_exp3_frames_only.log
    bash ~/adlm/scripts/run_exp3_frames_only_test.sh /data/cmd_ad/runs/exp3_frames_only_4b/lora/8ep/checkpoint-XXXX

Data lives on the `/data` volume (`LABEL=cmdad`, gp3 1 TB); the JSONs reference frames by absolute path.

## Measured on this box (4x L40S 46 GB, Qwen3.5-4B LoRA r32, bs 1 x 8 accum x 4 GPUs = 32)

| | |
|---|---|
| tokenization (one-time, cached) | 11 min 29 s -> 2.0 GB at `/data/cmd_ad/tokcache/exp3_frames_only` |
| step time | 103 s steady state (first step 128 s) |
| per epoch | 249 steps, ~7.1 h |
| 8 epochs | ~57 h (2.4 days) |
| peak VRAM | 25.9-30.0 GiB of 45.0 |

`preprocessing_num_workers: 48` with `OMP_NUM_THREADS=1` — more threads thrash the 48 vCPUs (load 333, 4x slower).
