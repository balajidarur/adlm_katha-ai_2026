# MTG (Mind the Gap) Qwen3.8-27B-FP8 dense descriptions (T4)

Self-contained copy of the Qwen3.8 charbank pipeline for non-film video (no cast, no circles):
`prep_nocast.py` (shots 3-10 s, 3 fps 360p frames, MTG transcript_segments -> SRT, optional chop offset),
`charbank_pipeline.py --no-cast --no-circles`, `run_dense_all.py --no-cast`.
Prompt variants: `nocast_desc_action_v2` (pilot), `nocast_mtg_b` (no-repeat, skip speech captions, no micro-motions),
`nocast_mtg_c` (b + setting-first, specific nouns, demo steps). Box scripts in `ops/`.
