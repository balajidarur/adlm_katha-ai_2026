# Dataset build plan — frame-interleaved CMD-AD, variants A / B / C

**Status (2026-09-11):** resolution fixed at **448×252**. Pilot built — 5 random clips per split
(seed 0), all three variants, in `/data/cmd_ad/datasets/frame_variants_448x252_pilot5/`.

**Full build done (2026-09-11, 35.6 min on 64 cores):** `/data/cmd_ad/datasets/frame_variants_448x252_full/`
— 8,552 clips (7,982 train / 570 eval), the 46 unreadable mp4s skipped. Verified: every `<image>`
count matches its images, all 1,259,992 image files exist, targets identical across A/B/C, exact
tokens ≤ 59,938. Frames in `/data/cmd_ad/frames/1fps_448x252/` (1,174,005 `f_` + 85,987 `u_`).
Log: `dataset_prep/logs/build_full_20260911_135030.log`.
Full build not yet run. Every number below is measured over the **8,552 usable clips** of the
8,598-clip text-only baseline (8,023 train / 575 eval, 96,624 GT ADs), not estimated.
Analysis notebooks (both in `/home/ubuntu/adlm/`, kernel "adlm venv (py3.11)"):
- `frame_sampling_analysis.ipynb` — resolution, outro, frames per clip, gap windows, AD coverage, pilot check
- `token_distribution_analysis.ipynb` — token distributions per dataset, frame rate (1 → ¼ FPS),
  frame cap, the 60k budget, exact-token check

Machine: `balu-tokyo-cpu64` (64 cores — `nproc` prints 1 only because the shell exports
`OMP_NUM_THREADS=1`). Source data and all outputs live on `/data` (984 GB, 258 GB free).

**Moved 2026-09-11:** the `/data` volume (`vol-0a4baa96e4eb1b41c`) is now attached to the 4×L40S box
`balu-tokyo-g6e-12x` — training paths and data layout are in **§9**.

---

## The three variants

The target is identical in all three — dialogue verbatim, interleaved with the GT ADs, in time
order — and identical to the text-only baseline. Only the input changes.

| | name (files + LLaMA-Factory datasets) | input | frames |
|---|---|---|---|
| **A** | `A_uniform1fps_dialogue` | frames + dialogue — **no dense descriptions** | uniform 1 FPS over the whole clip |
| **B** | `B_dense_dialogue_gapframes` | dense descriptions + dialogue + frames | 1 FPS **only inside dialogue gaps** ≥ 1.0 s, each padded ±25% of its own length |
| **C** | `C_dense_dialogue_uniform1fps` | dense descriptions + dialogue + frames | uniform 1 FPS over the whole clip |

### Measured cost and coverage (448×252, no cap)

| | frames/clip p50 · p99 · max | total frames | AD with ≥1 frame | AD window-covered | sequence p50 · p99 |
|---|---|---|---|---|---|
| **A** | 133 · 233 · 300 | 1,174,005 | 99.9% | 100% | 18,778 · 32,670 |
| **B** | 84 · 197 · 300 | 756,438 | 90.3% | 88.7% | 23,871 · 51,267 |
| **C** | 133 · 233 · 300 | 1,174,005 | 99.9% | 100% | 30,336 · 58,488 |
| text-only baseline (as built, dialogue start-only) | — | — | — | — | 12,860 · 29,743 |

### Storage — one shared frame pool

B's frames are the same 1 FPS grid points (each second's midpoint) that fall inside the gap
windows, so **every B frame is also an A/C frame** and B adds no storage — except in the 83 clips
whose B span is resampled (§2). All three reference one
pool, `/data/cmd_ad/frames/1fps_448x252/`.

With the 200-frame cap, uniform resampling and the 60k token budget (decided 2026-09-11):

| | train (7,982 clips) | eval/test (570) | total |
|---|---|---|---|
| 1 FPS grid frames `f_*.jpg` — 1,174,005 | 12.95 GB | 0.93 GB | 13.9 GB |
| resampled frames `u_*.jpg` — 85,987 | 0.97 GB | 0.05 GB | 1.0 GB |
| `A_uniform1fps_dialogue` JSON — 1,165,867 frames | 131 MB | 9.4 MB | 140 MB |
| `B_dense_dialogue_gapframes` JSON — 754,704 frames | 433 MB | 32 MB | 464 MB |
| `C_dense_dialogue_uniform1fps` JSON — 1,165,494 frames | 464 MB | 34 MB | 498 MB |
| **everything** | 14.9 GB | 1.05 GB | **~16.0 GB** |

The build extracts every clip's full 1 FPS grid, even where a variant resamples: 41,950 grid frames
(0.5 GB) end up unreferenced, and keeping them means the cap, the budget or a ½ FPS subset can
change later without re-extracting.

Frame size **12.4 KB** at 448×252, `-q:v 3`: measured on 100 random clips (95% CI 11.6–13.1 KB;
per-clip averages 4–24 KB — dark scenes compress smaller). The 10 pilot clips run small (10.8 KB).
JSON sizes are exact: every record rendered and serialized. Stored separately, the three variants
would need 13.8 + 8.9 + 13.8 = 36.5 GB of JPEGs. `/data` has 258 GB free.

*AD with ≥1 frame* = at least one sampled frame lies inside the AD's span. *Window-covered* =
≥ 50% of the AD's span lies inside a padded gap window. A/C miss 0.1% only because **97 GT ADs
start after the last dense shot** — past the content end, where no frame is sampled.

### What each comparison isolates

- **C vs baseline** — do frames add anything on top of dense descriptions + dialogue?
- **A vs C** — same frames; does the dense text still matter once the model can see?
- **B vs C** — same text; does restricting frames to the gaps lose anything, at **64% of the frames**?

---

## 1. Resolution — 448×252 (decided 2026-09-11; 360×252 is not 16:9)

360 / 252 = 1.43 (10:7); the sources are 16:9 = 1.78 (93.0% of clips; 91.3% are 1280×720).
Qwen3.5 then resizes every image to multiples of 32 px (`patch 16 × merge 2`, `min_pixels 65,536`),
so the model never sees 360×252 anyway:

| extract | model sees | tokens / frame | picture | C seq p99 | C clips > 32,768 |
|---|---|---|---|---|---|
| 360×252 stretched | 352×256 | 88 | **squashed 23%** | — | — |
| 360×252 letterboxed | 352×256 | 88 | 20% of the frame is black bars | — | — |
| **448×252** (16:9, pilot) | 448×256 | **112** | −1.6% | 58,488 | 39.1% |
| 384×216 (16:9) | 384×224 | 84 | −3.6% | 52,111 | 24.7% |
| 360×202 (16:9) | 352×192 | 66 | +3.1% | 48,032 | 16.9% |

112 tokens/frame was verified with the real Qwen3.5 image processor on the pilot JPEGs
(`image_grid_thw = 1×16×28`). LLaMA-Factory's `qwen3_5` template wraps each image as
`<|vision_start|>` + N `<|image_pad|>` + `<|vision_end|>`, and each `[frame M:SS.s]` label line
adds 11 tokens (the space before `<|vision_start|>` is a token of its own), so one frame costs
**N + 13 = 125** tokens — verified against exact tokenization of built records.

**Decision: 448×252** — true 16:9, 112 tokens/frame. If sequence length later forces a smaller
size, it is one flag: `--width 384 --height 216`.

Non-16:9 sources (7%) are **letterboxed**, never stretched, so every frame has the same size and
token count.

---

## 2. Frame cap, token budget and frame rate

### Frame cap — 250 costs almost nothing, but it is not the memory fix

| cap | A/C clips over | A/C frames dropped | B clips over | ADs losing their last frame |
|---|---|---|---|---|
| 150 | 2,816 (32.9%) | 6.8% | 709 (8.3%) | 25 (B) |
| 200 | 344 (4.0%) | 0.7% | 78 (0.9%) | 2 (B) |
| **250** | **38 (0.44%)** | **0.07%** | **8 (0.09%)** | **0** |
| 300 | 0 | 0 | 0 | 0 |

The longest clip is 300 s (`2019/JgTgQvvqRqE`), so the uncapped max is 300 frames. At cap 250 the
worst frame spacing is 1.2 s. A cap subsamples **evenly across the clip** (`even_cap`) — never by
cutting the tail.

**But a frame cap does not bound the sequence.** Long sequences come from long dense text *plus*
many frames, not from frame count alone:

| 448×252 | > 32,768 (current `cutoff_len`) | > 50,000 | > 60,645 (4B, one A6000: 45.0 GB peak) | max |
|---|---|---|---|---|
| A | 1.0% | 0.0% | 0.0% | 41,397 |
| B | 19.4% | 1.3% | 0.2% | 68,695 |
| C | 39.1% | 3.6% | **0.7%** | 78,354 |
| C, cap 250 | 39.1% | 3.6% | 0.6% | 74,854 |

LLaMA-Factory splits `cutoff_len` across prompt and target, so an over-length clip would lose the
**end of its AD target**, not just input.

**Cap: 200 frames per clip, resampled uniformly** (decided 2026-09-11; `--cap 200`, the build
default). No clip is dropped or cut short. A span with more than 200 frames at 1 FPS gets 200
frames spread **uniformly over the same span** at a lower rate — the whole clip for A/C, the
dialogue-gap windows laid end to end for B. Spans of 200 s or less stay at exactly 1 FPS.

| | clips resampled (eval) | lowest rate | |
|---|---|---|---|
| A | 344 (17) | 0.67 FPS | a 300 s clip → 200 frames, one every 1.5 s |
| B | 83 (3) | 0.67 FPS | uniform over the gap windows |
| C | 344 (17) | 0.47 FPS | the budget (below) takes a few C clips under 200 |

Resampled frames are the last source frame at or before each target time — the 1 FPS grid's
convention — saved as `u_{ms}.jpg`. `frames_lib.extract_at` picks them by frame index in one decode
pass, with the same scaling and JPEG quality. Checked on `UephHvStdWw` (207 s → 200 frames every
1.036 s, 0:00.5 … 3:26.6): each `u_` file matches the source frame at its time (mean abs diff
1.5–1.9, vs 2.0–9.4 for the neighbouring source frames). No GT AD loses its frames. After cap and
budget, sequence p99 / max: A 29,111 / 33,609 · B 50,521 / 59,919 · C 55,311 / 59,928.

### Token budget — 60,000, over-budget clips shrink by frames (decided 2026-09-11)

The 4B model on one 48 GB A6000 (bs 1, LoRA r32, gradient checkpointing, Liger) trained at
60,645 tokens (45.0 GB peak) and OOMed at 64,728 on step 2. The datasets were first built at
**budget 60,000**.

> **Superseded on the 4x L40S (2026-09-14): the budget is now 48,000** — see §10. The L40S has
> 44.4 GiB usable against the A6000's 48, so 60k OOMs. Rebuild with
> `build_frame_datasets.py --token-budget 48000`.

**Rule:** a clip over budget keeps all of its text and gets fewer frames, resampled uniformly over
its span as for the cap above, until it fits (`frames_lib.fit_frames`; 64 tokens held back for the chat template). Text
alone never exceeds 43,608 tokens, so no clip is truncated or dropped. Train and eval/test get the
same rule.

| budget | B clips over (train / eval) | C clips over (train / eval) | frames dropped B · C | affected clips keep | ADs losing their frames B · C |
|---|---|---|---|---|---|
| 50,000 | 109 (103 / 6) | 318 (301 / 17) | 0.62% · 1.28% | ~84% | 27 · 54 |
| 55,000 | 49 (44 / 5) | 143 (137 / 6) | 0.24% · 0.54% | ~85% | 5 · 13 |
| **60,000** | **20 (19 / 1)** | **63 (60 / 3)** | **0.07% · 0.19%** | **~90%, ≥ 130 frames** | **0 · 2** |

A never exceeds 50k (max 41,397). Sequence percentiles before shrinking:

| | p50 | p90 | p95 | p98 | p99 | p99.5 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|
| A | 18,778 | 25,642 | 27,508 | 30,445 | 32,670 | 34,569 | 39,260 | 41,397 |
| B | 23,871 | 37,643 | 41,980 | 47,472 | 51,267 | 55,627 | 63,329 | 68,695 |
| C | 30,336 | 44,100 | 48,177 | 54,041 | 58,488 | 61,643 | 69,907 | 78,354 |

**Verified exactly:** the four longest C clips, built with the budget and tokenized for real (images
expanded to 112 `<|image_pad|>`, chat template included), came out at 50,470–59,879 tokens with the cap and uniform resampling (all 12 A/B/C records
≤ 60k). On the
pilot's 30 records, exact − estimate is a constant 10–13 tokens whatever the frame count.

### Frame rate — ½ FPS loses almost nothing

Clips are short (median 133 s; 41% are 2:00–2:30), and GT AD spans are ~2.2 s, so a frame every
2 s still lands inside nearly every AD. Lower rates take every 2nd / 3rd / 4th frame of the 1 FPS
pool — no re-extraction.

| FPS | A/C frames p50 | C tokens p50 · p99 | C clips > 60k | AD ≥1 frame A/C · B | frame pool |
|---|---|---|---|---|---|
| 1 | 133 | 30,336 · 58,488 | 63 | 99.9% · 90.3% | 13.9 GB |
| **½** | **67** | **21,843 · 44,133** | **1** | **99.9% · 89.2%** | **7.0 GB** |
| ⅓ | 45 | 19,029 · 39,589 | 0 | 80.8% · 72.2% | 4.7 GB |
| ¼ | 34 | 17,619 · 37,438 | 0 | 65.9% · 59.0% | 3.5 GB |

½ FPS halves A's tokens (p50 18.8k → 10.5k) and removes the budget problem, at no cost in "≥1
frame per AD". What it does cost: ADs with ≥2 frames, i.e. motion within an AD's span. ⅓ FPS is
where coverage breaks. The build is 1 FPS; ½ FPS is a stride-2 selection from the same frames if
the pilot training says frame density does not matter.

---

## 3. AD coverage of variant B

- **9,373 of 96,624 ADs (9.7%)** get no B frame. **97.9%** of those overlap speech for ≥ 50% of
  their span — they are ADs spoken over dialogue, which no gap-based rule can reach.
- 38.6% of the missed ADs are more than 3 s from the nearest window; 21.6% more than 5 s.
- Eval and train behave the same (B window-covered 88.3% eval / 88.8% train).

### Why ≥ 1.0 s gaps and ±25% padding (sweep over all clips)

| min gap | pad | timeline sampled | B frames mean | AD ≥1 frame |
|---|---|---|---|---|
| 0 s | ±25% | 70.3% | 97.2 | 91.8% |
| **1.0 s** | **±25%** | **63.6%** | **88.5** | **90.3%** |
| 1.0 s | ±50% | 72.0% | 99.8 | 93.4% |
| 1.0 s | ±2 s fixed | 72.3% | 99.8 | 90.5% |
| 2.0 s | ±25% | 56.5% | 79.0 | 87.6% |

±2 s fixed reaches the same coverage as ±25% with **13% more frames**. Going from ±25% to ±50%
buys +3.1 points of coverage for +13% frames — a reasonable alternative if B under-performs.

---

## 4. Data facts that shape the build

| fact | consequence |
|---|---|
| The mp4s run past the last dense shot — by ~30 s (most clips), ~10 s or ~0 s: a MovieClips outro. Dense events, dialogue and ADs stop before it. | Frames are sampled on `[0, content_end)`, content end = last dense shot (or the mp4 end in the 17 clips where the mp4 is shorter). Sampling to the end of the file would add **17% frames of outro**. |
| ffmpeg `fps=1` returns, as frame *k*, the last source frame before *k* + 0.5 s (measured 0.46–0.49 s). | The grid is each second's **midpoint**: `f_0000k.jpg` is labelled `[frame M:SS.5]`. Labelling it *k*.0 would misplace every frame by half a second. |
| Cutting the input at the content end (`-t end`) drops the last frame — the filter needs input to *k* + 0.5 s before it emits frame *k*. | Input runs 1 s past the last frame; output is capped with `-frames:v N`. Verified: 60/60, 131/131. |
| **46 mp4s are truncated downloads** (no moov atom; 5 of them eval). | Excluded from every frame variant: 8,552 usable of 8,598. Re-download to recover them. |
| 5 clips have no ASR. | `(no dialogue)`; for B the whole clip is one gap. |
| The shell exports `HF_HOME=/mnt/disk1/...` from another machine. | Tokenizer/processor are loaded from the local snapshot path, not the hub id. |

---

## 5. Output layout

```
/data/cmd_ad/frames/1fps_448x252/{video_id}/f_{k:05d}.jpg     1 FPS grid, shared by A/B/C
/data/cmd_ad/frames/1fps_448x252/{video_id}/u_{ms:07d}.jpg    resampled frames (spans over 200 frames)
/data/cmd_ad/datasets/frame_variants_448x252_pilot5/          (full run: ..._full/)
    cmdad_A_uniform1fps_dialogue_{train,eval}.json         ShareGPT: messages + images (absolute paths)
    cmdad_B_dense_dialogue_gapframes_{train,eval}.json
    cmdad_C_dense_dialogue_uniform1fps_{train,eval}.json
    cmdad_*_{train,eval}_meta.json     per clip: variant, content_end, n_grid, n_frames, n_ad,
                                       ad_window_covered, ad_with_frame, ...
    dataset_info.json                  LLaMA-Factory registration, keys = file names without .json
    build_stats.json                   args, per-variant totals, failures, extraction time
```

Image paths are absolute; LLaMA-Factory's `os.path.join(media_dir, path)` passes them through.

**User turn** — the baseline prompt with the opening sentence changed per variant, then:
- **A:** `## Frames` (one `[frame 0:17.5] <image>` line per frame) → `## Dialogue`
- **B / C:** `## Characters` / `## Locations` / `## Props` → `## Visual events` with the frame lines
  interleaved among the dense events by time (a frame precedes an event at the same timestamp) →
  `## Dialogue`

Timestamps in the input (decided 2026-09-11):

| | format | why |
|---|---|---|
| frame | `[frame 0:17.5] <image>` | the moment the picture was taken |
| dense event | `0:00.0  Eye-level medium shot ...` — **start only** | the dense JSON has no per-event end time |
| dialogue | `0:13.1-0:18.5  You nearly made me wet myself.` — **start and end** | the gaps between lines are where ADs go; WhisperX gives both ends. Costs ~280 tokens/clip at the median |

**Assistant turn** — dialogue verbatim with its start and end, ADs with their start, in time
order (decided 2026-09-11):

```
0:01.7-0:02.0 [DIALOGUE] Joe?
0:04.5-0:04.8 [DIALOGUE] Joe?
0:05.9 [AD] He sneaks up on her, blowing on the back of her head.
```

The prompt's rules match: `M:SS.s [AD] text` / `M:SS.s-M:SS.s [DIALOGUE] text`, and "Reproduce
every dialogue line exactly, with its start and end time".

Consequences of dialogue end times in the output:
- target tokens **+26%** (median 878 → 1,092); ADs become 20.7% of target tokens instead of 26.2%
- **5.9% of targets (504 clips, 33 eval) exceed 2,048 tokens** — the `max_new_tokens` in
  `fast_infer.py`. Generate with **`max_new_tokens: 3072`** (0.7% over)
- the output format now differs from the text-only baseline's: the grounding judge's parser must
  accept `M:SS.s-M:SS.s [DIALOGUE]` lines. The judge is not on this machine — check before scoring

**AD end times: not added.** GT AD duration follows from the text's length (duration ≈ 0.22 s ×
words + 0.82 s, r = 0.86, median error 0.33 s), so predicting it teaches little. 12.4% of GT ADs
run past the next dialogue line's start and 25.2% overlap dialogue, so the ends don't encode a
clean fit-in-the-gap rule either.

Every record is asserted to have `<image>` count = `len(images)`, every image on disk, and the
same target across A/B/C.

### Pilot clips (seed 0)

| split | clips | A / C frames | B frames | ADs | B: ADs with a frame |
|---|---|---|---|---|---|
| train | `9nrVYO6LU6I` `OxBd2RQI4eQ` `QQAd5xx0XHc` `n2YCseaZK0Q` `tYs7uguB_JQ` | 727 | 610 | 103 | 94 |
| eval | `164eTBYcySQ` `7m3Mv9f_P1U` `H4zYhsldk8M` `IgrJkJ8NYrw` `UephHvStdWw` | 705 | 385 | 28 | 27 |

`UephHvStdWw` (207 s) is resampled from 207 to 200 frames, spread uniformly (one every 1.036 s).

`QQAd5xx0XHc` is a 640×264 (2.42:1) source — letterboxed, still 112 tokens/frame.

---

## 6. Building

```bash
cd /home/ubuntu/adlm/dataset_prep
export OMP_NUM_THREADS=1
# pilot (done)
../venv/bin/python build_frame_datasets.py --n-per-split 5
# full run -> /data/cmd_ad/datasets/frame_variants_448x252_full/
../venv/bin/python build_frame_datasets.py --n-per-split 0 --workers 60
```

| file | role |
|---|---|
| `frames_lib.py` | loading, gap windows, frame grid, coverage, prompt rendering, ffmpeg extraction |
| `build_frame_datasets.py` | extracts the pool once per clip and writes all three variants |
| `analysis_lib.py` | per-clip facts + token counts for the notebook |

Measured: 2.4 s/clip extraction with 10 parallel workers (1280×720 H.264 decode). The full pool
is ~8,552 clips × 2.4 s / 60 workers ≈ **6–10 min** and ~13.9 GB of frames.

---

## 7. Open decisions

Decided 2026-09-11:
1. **Resolution** — 448×252.
2. **Splits** — `eval` (570 usable clips) is the test set; no separate validation split.
3. **The 46 truncated mp4s** — skipped; the build excludes them (8,552 clips).
4. **Token budget** — 60,000, over-budget clips shrink by frames (§2).
5. **Timestamps** — dialogue start–end in input and output; dense events and ADs start only (§5).
6. **Frame cap** — 200 frames per clip; longer spans are resampled uniformly at a lower rate, never
   cut or dropped (§2).

Still open: 1 FPS vs ½ FPS (§2) — worth settling in the pilot training.

---

## 8. Training notes carried over (measured earlier on 4B, A6000, bs 1, LoRA r32 — not re-measured here)

- `enable_liger_kernel: true` is mandatory: without fused cross-entropy the `seq × 248,320`
  logits cost ~22 GB before any image (34.3 GB → 12.5 GB at zero frames).
- 250 frames at 112 tokens: 60,645-token sequence, 45.0 GB peak, **183 s/step** — 28× the
  text-only step, because the frozen vision tower re-encodes every frame every step. The vision
  tower carries no LoRA adapters, so **cache its outputs once** and reuse them across epochs.
- Tokenize once with `tokenized_path` set; `preprocessing_num_workers: 32` with
  `OMP_NUM_THREADS=1` (unpinned workers fight over cores).
- Pilot-train each variant on ~500 clips × 2 epochs before committing to a full run; compare the
  grounding rate against the text-only baseline (**23.1% GROUNDED / 58.3% UNGROUNDED**) and the
  human ceiling (**26.1% / 45.0%**).

---

## 9. Training handoff — where everything is on the 4×L40S box

**Machine:** `balu-tokyo-g6e-12x` · `i-0f7aa5fa06eb456f0` · g6e.12xlarge, 4× L40S 48 GB ·
ap-northeast-1a · private IP `172.31.45.166` (the public IP changes on every start).
**Disk:** `vol-0a4baa96e4eb1b41c` (gp3, 1000 GB) mounted at **`/data`** — `fstab` entry
`LABEL=cmdad /data ext4 defaults,nofail 0 2`, so it mounts on every boot.

### How the data is stored

```
/data/cmd_ad/
├── datasets/
│   ├── frame_variants_448x252_full/          ← TRAINING DATA (LLaMA-Factory dataset_dir)
│   │   ├── dataset_info.json                 the 6 datasets below, registered (messages + images)
│   │   ├── cmdad_A_uniform1fps_dialogue_{train,eval}.json
│   │   ├── cmdad_B_dense_dialogue_gapframes_{train,eval}.json
│   │   ├── cmdad_C_dense_dialogue_uniform1fps_{train,eval}.json
│   │   ├── cmdad_*_{train,eval}_meta.json    one row per record, same order: video_id, year, split,
│   │   │                                     sampling (1fps|uniform), fps, n_frames, n_frames_1fps,
│   │   │                                     seq_tokens, n_ad, ad_with_frame, ...
│   │   └── build_stats.json                  build args, per-dataset totals, the 46 skipped clips
│   ├── frame_variants_448x252_pilot5/        5 train + 5 eval clips, same format (smoke tests)
│   └── data/                                 the text-only baseline (ad_interleave_{train,eval}.json)
├── frames/1fps_448x252/{video_id}/           448×252 JPEGs, 15.6 GB, referenced by absolute path
│   ├── f_{k:05d}.jpg                         1 FPS grid: the picture at t = k + 0.5 s
│   └── u_{ms:07d}.jpg                        resampled frames (spans over 200 frames): t = ms / 1000 s
├── videos/ dense/ asr/ annotations/ ads/     the sources the datasets were built from
/data/claude_transfer/                        the carried-over chats + this folder (see §9 end)
```

**Record format** (ShareGPT): `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
"images": ["/data/cmd_ad/frames/1fps_448x252/<vid>/f_00000.jpg", ...]}`. The user turn holds one
`<image>` per entry of `images`, in the same order (`[frame M:SS.s] <image>` lines); the assistant
turn is the target (dialogue `M:SS.s-M:SS.s [DIALOGUE]`, ADs `M:SS.s [AD]`), identical across A/B/C.

| dataset name (in `dataset_info.json`) | records | frames | JSON |
|---|---|---|---|
| `cmdad_A_uniform1fps_dialogue_train` / `_eval` | 7,982 / 570 | 1,087,368 / 78,499 | 131 / 9 MB |
| `cmdad_B_dense_dialogue_gapframes_train` / `_eval` | 7,982 / 570 | 703,350 / 51,354 | 433 / 32 MB |
| `cmdad_C_dense_dialogue_uniform1fps_train` / `_eval` | 7,982 / 570 | 1,087,030 / 78,464 | 464 / 34 MB |

`_eval` is the **test set** (the official CMD-AD eval split) — keep it out of training; there is no
separate validation split.

### LLaMA-Factory settings these datasets need

```yaml
dataset_dir: /data/cmd_ad/datasets/frame_variants_448x252_full
dataset: cmdad_C_dense_dialogue_uniform1fps_train   # or cmdad_A_uniform1fps_dialogue_train / cmdad_B_dense_dialogue_gapframes_train
template: qwen3_5_nothink
cutoff_len: 60000              # every record is ≤ 59,938 tokens exact, images included — nothing truncates
image_max_pixels: 131072       # ≥ 448×256, so frames reach the Qwen processor unresized (112 tokens each)
enable_liger_kernel: true      # mandatory (248k vocab)
gradient_checkpointing: true   # 60k context on 48 GB
per_device_train_batch_size: 1
tokenized_path: /data/cmd_ad/tokcache/<dataset>   # one cache per dataset; reruns skip preprocessing
output_dir: /data/cmd_ad/runs/<dataset>           # /data has ~239 GB free
```

For generation on `_eval`: **`max_new_tokens` ≥ 3072** — 5.9% of targets are longer than 2,048
tokens now that dialogue carries end times. The grounding judge's parser must accept
`M:SS.s-M:SS.s [DIALOGUE]` lines.

Memory reference (4B, one 48 GB GPU, bs 1, LoRA r32, gradient checkpointing, Liger): 60,645 tokens
peaked at 45.0 GB; 64,728 OOMed on step 2 — the 60k budget sits just under that.

### Carried-over context

The box's home directory is a clone of the dataset machine's, so the training environment is the
same: `~/adlm/venv` (Python env), `~/adlm/LLaMA-Factory`, earlier configs and launchers in
`~/adlm/configs/` and `~/adlm/run_4b.sh` (the 4B text-only baseline is `configs/ad4b_full.yaml`).
What the clone lacked was copied over directly: `~/adlm/dataset_prep/` (this folder), the two
analysis notebooks in `~/adlm/`, and the Claude Code chats + memory in `~/.claude/projects/` —
resume the dataset session with `cd ~/adlm && claude --resume`. `/data/claude_transfer/` holds an
earlier copy of the same files.


---

## 10. Experiment 3 runs — 4x L40S (`balu-tokyo-g6e-12x`)

### frames_only (A) — done 2026-09-14

| | |
|---|---|
| config / runner | `configs/exp3_frames_only_4b.yaml` · `scripts/run_exp3_frames_only.sh` |
| wall time | **53.9 h** · 97 s/step · 6.7 h/epoch · 2,000 steps, 8 epochs |
| loss | 0.425 -> 0.117 (train_loss 0.242) |
| output | `/data/cmd_ad/runs/exp3_frames_only_4b/lora/8ep` — 16 checkpoints + final adapter, 13 GB |
| wandb | `adlm-ad-interleave/exp3-frames-only-4b-8ep` |
| peak VRAM | 30.0 GiB of 44.4 at its 33,609-token maximum |

### What this box changes

- **`ffmpeg` must be installed** (`sudo apt install ffmpeg`). It was missing here, and a rebuild
  silently failed 467 clips (every clip needing a *new* resampled frame) before it was noticed.
- **Token ceiling ~57k, not 64k.** A's 30.0 GiB at 33,609 tokens gives ~0.60 GiB per 1k tokens on top
  of ~10 GiB of weights and optimizer, so 44.4 GiB runs out near 57k. B OOMed at its 59,919-token max
  (it wanted 2.06 GiB more). **Budget lowered to 48,000** (peak ~38.8 GiB): it re-plans 149 B clips
  (1.9%) and 392 C clips (4.9%), which keep all their text and lose a few frames each. A is untouched
  (its longest record is 33,609) and its run and tokenized cache remain valid.
- **`gradient_checkpointing: false` OOMs** — and LLaMA-Factory ignores that key anyway; its own flag is
  `disable_gradient_checkpointing`.
- **flash-attn 2 is already in use** through torch SDPA ("Using torch SDPA" in the log, torch 2.11's
  flash backend enabled). The standalone package cannot build here: no CUDA toolkit (`nvcc`).
  Only 8 of 32 layers use full attention anyway.
- **Length grouping pays ~6%**: `train_sampling_strategy: group_by_length` (the transformers v5 name;
  `group_by_length` is rejected by the parser). Predicted 8.4% from the token distribution, measured
  97 s/step against the ungrouped probe's 103 s.
- **The vision tower is only ~6% of a step** (133 frames encode in 0.71 s; a sample costs 12.9 s per
  GPU). Caching its outputs — the big win the old plan assumed — is not worth building.
- **`preprocessing_num_workers: 48` needs `OMP_NUM_THREADS=1`**: with 8 threads the 48 workers drove
  load to 333 on 48 vCPUs and tokenization ran 4x slower (37 min vs 11.5 min).
