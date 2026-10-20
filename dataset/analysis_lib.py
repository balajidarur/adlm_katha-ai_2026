"""Per-clip facts for frame_sampling_analysis.ipynb — parsed once, in parallel.

No frames are extracted here; frame counts follow from the 1 FPS grid on [0, content_end).
Token counts cover the text only; each frame's cost is added per variant in the notebook.
"""

import numpy as np

import frames_lib as fl

# a local snapshot path, not a hub id: the shell exports an HF_HOME from another machine
TOKENIZER = ("/home/ubuntu/adlm/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/"
             "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
_tok = None


def ntok(text):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return len(_tok(text, add_special_tokens=False)["input_ids"])


def analyze_clip(job):
    row, ads = job
    vid, year = row["video_id"], row["year"]
    info = fl.video_info(fl.video_path(year, vid))
    dense, events, dense_end = fl.load_dense(vid)
    dialogue = fl.parse_srt(fl.asr_path(year, vid))
    no_frames = np.array([])
    return {
        **row,
        "readable": info is not None,
        "w": info["w"] if info else None,
        "h": info["h"] if info else None,
        "video_dur": info["dur"] if info else np.nan,
        "dense_end": dense_end,
        "content_end": min(dense_end, info["dur"]) if info else dense_end,
        "n_events": len(events),
        "n_dialogue": len(dialogue),
        "n_ad": len(ads),
        "dialogue": [(s, e) for s, e, _ in dialogue],
        "ads": [(s, e) for s, e, _ in ads],
        "tok_A_text": ntok(fl.render_user("A", dense, events, dialogue, no_frames)),
        "tok_B_text": ntok(fl.render_user("B", dense, events, dialogue, no_frames)),
        "tok_C_text": ntok(fl.render_user("C", dense, events, dialogue, no_frames)),
        "tok_target": ntok(fl.render_target(dialogue, ads)),
    }
