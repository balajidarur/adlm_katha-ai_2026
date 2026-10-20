# Frame-Interleaved Dataset — preparation and transfer

Building the next ADLM training set: the existing text input (dense visual descriptions +
dialogue) **with video frames interleaved at their timestamps**, so the model sees pixels
where the audio descriptions actually land instead of only a text summary of them.

Everything below is measured on this data, not estimated. Source of the numbers:
`experiment2_4b_ep3/RESULTS.md`.

---

## 1. Why

Judged against the dense descriptions it was given, the 4B model is **23.1% GROUNDED /
58.3% UNGROUNDED**. The human-written ADs score **26.1% / 45.0%** against the same evidence
— so the model is already at **88% of the human ceiling**, and that ceiling is set by the
dense descriptions, not by the model. A causal test (strip character names from ungrounded
ADs, re-judge) showed **93.8% of ungrounded ADs fail because the event was never described
at all**, not because of naming.

Scaling the model cannot fix that. Giving it the frames can.

---

## 2. Sampling design (all measured over 8,598 clips / 97,173 GT ADs)

| parameter | value | why |
|---|---|---|
| resolution | **448 x 252** | 112 tokens/frame (measured, `patch 16 x merge 2` = 32x32 px/token) |
| rate | **1 FPS** | |
| windows | speech-free gaps **>= 1.0 s**, padded **±25% of each gap's own length** | |
| frames/clip | median **77**, p90 109, max 300 | |
| total frames | **702,512** | |
| dataset size | **~9.0 GB** (13.4 KB/frame measured) | |

**Coverage achieved** — an AD counts as covered when >=50% of its span falls inside a sampled window:

| window | timeline sampled | AD coverage |
|---|---|---|
| gap only | 49.5% | 83.2% |
| **±25% padded** | **63.6%** | **88.8%** |

Sampling ~64% of the timeline captures ~89% of the ADs, because ADs live in the gaps.

**Why ±25% proportional, not fixed padding:** ±2s fixed reaches the same 87.5% coverage but
needs 100 frames/clip instead of 83 — 20% more tokens for nothing. Long gaps hold most of the
ADs and deserve the padding; the many short gaps do not.

**Why >= 1.0 s minimum gap:** dropping to 0 s adds only 1.3 points of AD coverage while more
than doubling the number of segments (13 -> 32 per clip), each needing its own boundary markers.

The remaining **11.2% of ADs sit more than 3 s inside speech** and no amount of gap padding
reaches them.

---

## 3. Memory and compute (measured on an A6000, bs=1, LoRA r32, gradient checkpointing, Liger)

| frames | sequence | peak GPU | step time |
|---|---|---|---|
| 0 (text only) | 12,011 | 15.7 GB | 6.6 s |
| 109 | 24,437 | 22.3 GB | 15.4 s |
| **250** | **60,645** | **45.0 GB** | **183 s** |
| 300 (worst clip) | 64,728 | 48.4 GB | **OOM** |

- **`enable_liger_kernel: true` is mandatory.** Without fused cross-entropy the logits tensor
  (`sequence x 248,320 vocab`) costs ~22 GB before a single image. Measured: 34.3 GB -> 12.5 GB
  at zero frames.
- **Cap `cutoff_len` at ~50,000–60,000.** The 300-frame worst case OOMs; it survived step 1 and
  died on step 2 when AdamW allocated its optimizer state.
- **Compute, not memory, is the binding constraint.** 250 frames costs 183 s/step — 28x the
  text-only step for 5x the sequence, because the frozen vision tower re-encodes every frame
  on every step.
- **Therefore: cache the vision-encoder outputs.** Frames are static and the encoder is frozen
  (verified: 496 LoRA adapters, **zero** on the vision tower). Encode once, store the 112
  embeddings per frame, reuse across all epochs. This is the single biggest win available.
- DeepSpeed does **not** help here: 37.2 GB of the peak is activations, and only 0.72 GB is
  offloadable optimizer state because LoRA makes it tiny. On one GPU there is nothing to shard.

---

## 4. Where data may be written  **(read this first)**

`/home` is quota-limited to ~50 GB/user on a shared box. A previous run wrote ~45,000 JPGs
under home and added ~150 GB, disrupting other users.

| size | location |
|---|---|
| a handful of files | anywhere |
| up to ~30-40 GB | `/workspace/dbalu` |
| larger | **ask first**, then `/storage/users/dbalu` |

**Current state — check before writing:**
- `/workspace` is at **100% (13 GB free of 3.5 TB)** — it cannot absorb 9 GB with margin
- `/storage/user/dbalu` does **not** exist; the real path is `/storage/users/`
- Resolve where the 9 GB goes **before** extracting a single frame

---

## 5. Extraction

```bash
ffmpeg -nostdin -loglevel error -i "$VIDEO" \
       -vf "fps=1,scale=448:252" -q:v 3 "$OUT/f_%04d.jpg" -y
```

Measured: a full 137 s clip yields 131 frames in **0.385 s**. Across 8,598 clips, parallelised
over the 48 local cores, extraction is **~10 minutes** — it is not the bottleneck.

Frames are then subsampled to the padded-gap windows (keep the frame whose second falls inside
a window). Subsample **evenly across the clip**, never by truncating the tail, or the model
goes blind to the end of long clips.

---

## 6. Transfer to `balu-tokyo-cpu64`

```
Host balu-tokyo-cpu64
  HostName 54.178.46.80
  User ubuntu
  IdentityFile ~/.ssh/balu-aws-test.pem
  StrictHostKeyChecking accept-new
```

Instance: `i-0d5f654f929a2bccd` · **c6i.16xlarge** · ap-northeast-1a · target dir
`/home/ubuntu/adlm/dataset_prep/`

**Direct SSH from the katha-ai box does not work** — port 22 times out; this network blocks
outbound SSH to arbitrary hosts. Two options:

1. **S3 relay (preferred).** Create a bucket in **ap-northeast-1** (existing buckets are Seoul
   `apne2` and Mumbai `apsouth1` — cross-region would add egress cost and latency).
   - measured uplink: **6.7 MB/s** -> ~23 min for 9 GB
   - S3 -> EC2 same-region: ~5 min
   - **Tar into ~100 bundles first.** 702k individual PUTs would cost ~$3.60 in request charges
     alone and per-object overhead would dominate.
2. **EC2 Instance Connect Endpoint**, as used for the GPU box
   (`/home/dbalu/.claude/jobs/e71e2924/tmp/ssh_tokyo.sh` is the working pattern).

End-to-end estimate: **~40 min to 1 hour**, dominated by extraction and upload.

---

## 7. Order of work

1. Decide the write location (see §4) — blocking
2. Extract frames at 448x252, 1 FPS, restricted to `>=1s` gaps padded ±25%
3. Bundle into ~100 tarballs
4. Upload to an ap-northeast-1 bucket, pull down onto `balu-tokyo-cpu64`
5. Build the interleaved ShareGPT JSON (`messages` + `images`), `<image>` markers placed at
   their timestamps inside the `## Visual events` section
6. Tokenize **once** with `tokenized_path` set, so reruns skip preprocessing
   (measured: 32 workers with `OMP_NUM_THREADS=1` finished in 62 s where 8 threaded workers
   had not finished in minutes — pin threads or the workers fight over cores)
7. Pilot before committing: ~500 clips x 2 epochs on a single GPU, ~9 h, ~$45

---

## 8. Reference: the stress test that produced §3

`/home/dbalu/workspace/ad_gen/stress_test/` — the longest clip in the dataset
(`2019/JgTgQvvqRqE`, 300.14 s), every frame at 1 FPS, repeated 100x, trained for 1 epoch.
Token split at 250 frames: **32,645 text + 28,000 image = 60,645**.
