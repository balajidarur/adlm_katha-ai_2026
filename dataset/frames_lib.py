"""Shared logic for the frame-interleaved CMD-AD datasets (variants A, B, C).

A  uniform 1 FPS frames + dialogue                              (no dense descriptions)
B  dense descriptions + dialogue + 1 FPS frames only inside dialogue gaps
   (gaps >= GAP_MIN s, each padded by ±PAD_FRAC of its own length)
C  dense descriptions + dialogue + uniform 1 FPS frames

Clip content ends at the dense JSON's last shot, not at the end of the mp4: the
source videos carry a ~30 s MovieClips outro that no dense event, dialogue line
or AD reaches. Frames are sampled on [0, content_end) only.

Dense event timestamps are SHOT-relative; absolute clip time is
shot["time_offset"] (falling back to shot["start_abs"]) + event timestamp.
"""

import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict

import numpy as np

ROOT = "/data/cmd_ad"
DENSE_MODEL = "gemini-3.5-flash-lite"
BUILT_DIR = f"{ROOT}/datasets/data"          # the text-only baseline this extends
ANNO_CSV = f"{ROOT}/annotations/cmd_ad_anno_v1.csv"

FPS = 1
GAP_MIN = 1.0
PAD_FRAC = 0.25
VARIANTS = ("A", "B", "C")
# file and dataset names: the plan's letter + what the model is given
VARIANT_NAMES = {
    "A": "frames_only",
    "B": "dds_gapframes",
    "C": "dds_frames",
}

_RULES = (
    "Produce the clip's soundtrack script: the dialogue reproduced verbatim, "
    "interleaved in time order with the audio descriptions you write.\n"
    "Rules:\n"
    "- One line per entry: `M:SS.s [AD] text` for an audio description, "
    "`M:SS.s-M:SS.s [DIALOGUE] text` for a dialogue line.\n"
    "- Reproduce every dialogue line exactly, with its start and end time.\n"
    "- Write audio descriptions only where they fit between dialogue lines, "
    "describing what a sighted viewer sees that the dialogue does not convey.\n"
    "- Refer to characters by name once the dialogue reveals it.\n"
    "- Keep descriptions concise and in the present tense.\n"
    "- Output lines in strictly increasing timestamp order."
)
_TIME_NOTE = ("Each frame is labelled `[frame M:SS.s]` with the time it was taken, and each "
              "dialogue line with its start and end time, `M:SS.s-M:SS.s`.")
PROMPTS = {
    "A": ("You are an audio description writer. You are given frames sampled from a film "
          "clip at one frame per second and every line of dialogue in it. " + _TIME_NOTE
          + "\n" + _RULES),
    "B": ("You are an audio description writer. You are given the visual events of a film "
          "clip, each with its start time, frames sampled at one frame per second from the "
          "stretches between dialogue lines, and every line of dialogue in it. " + _TIME_NOTE
          + "\n" + _RULES),
    "C": ("You are an audio description writer. You are given the visual events of a film "
          "clip, each with its start time, frames sampled from it at one frame per second, "
          "and every line of dialogue in it. " + _TIME_NOTE + "\n" + _RULES),
}

SRT_TIME_RE = re.compile(
    r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


# ---------------------------------------------------------------- paths / loading

def video_path(year, vid):
    return f"{ROOT}/videos/{year}/{vid}.mp4"


def asr_path(year, vid):
    return f"{ROOT}/asr/SRT/{year}/{vid}.srt"


# DENSE_PATH_TMPL switches the dense-description source (default: Gemini), e.g. the Qwen3.8 run:
#   DENSE_PATH_TMPL=/prep/qwen_dense_final/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json
DENSE_PATH_TMPL = os.environ.get("DENSE_PATH_TMPL")


def dense_path(vid):
    if DENSE_PATH_TMPL:
        return DENSE_PATH_TMPL.format(vid=vid)
    return f"{ROOT}/dense/{vid}/{DENSE_MODEL}/adlm_descriptions.json"


def fmt(t):
    """Seconds -> 'M:SS.s' at 0.1 s precision."""
    t = max(0.0, round(t, 1))
    m = int(t // 60)
    return f"{m}:{t - m * 60:04.1f}"


def parse_tc(tc):
    """'MM:SS.s' (shot-relative timecode) -> seconds."""
    try:
        mm, rest = tc.split(":")
        return int(mm) * 60 + float(rest)
    except Exception:
        return None


def _secs(h, m, s, frac):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)


def parse_srt(path):
    """[(start, end, text)] from a WhisperX SRT; [] if the file is missing."""
    out = []
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    for block in re.split(r"\n\s*\n", raw):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        for i, l in enumerate(lines):
            m = SRT_TIME_RE.search(l)
            if m:
                break
        else:
            continue
        text = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
        if text:
            g = m.groups()
            out.append((_secs(*g[:4]), _secs(*g[4:]), text))
    return out


def load_index():
    """The clips of the text-only baseline: [{video_id, year, split}] (8,023 train + 575 eval)."""
    rows = []
    for split in ("train", "eval"):
        for m in json.load(open(f"{BUILT_DIR}/ad_interleave_{split}_meta.json")):
            rows.append({"video_id": m["video_id"], "year": m["year"], "split": split})
    return rows


def load_ads():
    """video_id -> [(start, end, text)] sorted by start, from the CMD-AD annotation CSV."""
    ads = defaultdict(list)
    with open(ANNO_CSV) as fh:
        for r in csv.DictReader(fh):
            vid = r["cmd_filename"].split("/")[1]
            try:
                ads[vid].append((float(r["scaled_start"]), float(r["scaled_end"]),
                                 r["text"].strip()))
            except (ValueError, KeyError):
                continue
    return {v: sorted(a) for v, a in ads.items()}


def load_dense(vid):
    """(dense json, [(abs_time, description)] sorted, content_end = last shot end)."""
    dense = json.load(open(dense_path(vid)))
    events, end = [], 0.0
    for shot in dense.get("shots", []):
        off = shot.get("time_offset")
        if off is None:
            off = shot.get("start_abs", 0.0)
        end = max(end, float(shot.get("end_abs") or 0.0))
        for ev in shot.get("events", []) or []:
            rel = parse_tc(ev.get("timestamp", ""))
            desc = (ev.get("description") or "").strip()
            if rel is not None and desc:
                events.append((off + rel, desc))
    events.sort()
    return dense, events, end


def video_info(path):
    """{'dur', 'w', 'h'} of the video stream, or None if the file cannot be opened
    (46 of the 8,598 baseline clips are truncated downloads with no moov atom)."""
    import av
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
            return {"dur": dur, "w": s.codec_context.width, "h": s.codec_context.height}
    except Exception:
        return None


# ---------------------------------------------------------------- sampling

def frame_times(content_end, fps=FPS):
    """The 1 FPS grid on [0, content_end): the midpoint of each second, t = 0.5, 1.5, ...

    Midpoints because that is what ffmpeg's fps filter delivers: output frame k is the
    last source frame before k + 0.5 s (measured offsets 0.46-0.49 s at 23.98 fps).
    """
    return np.arange(0.5 / fps, content_end, 1.0 / fps)


def speech_spans(dialogue, end):
    """Merged [start, end] speech spans clipped to [0, end]."""
    merged = []
    for s, e in sorted((max(0.0, s), min(end, e)) for s, e, *_ in dialogue):
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def dialogue_gaps(dialogue, end, min_gap=GAP_MIN):
    """Speech-free stretches of [0, end] at least `min_gap` long, lead-in and tail included."""
    gaps, cur = [], 0.0
    for s, e in speech_spans(dialogue, end):
        if s - cur > 0 and s - cur >= min_gap:
            gaps.append((cur, s))
        cur = max(cur, e)
    if end - cur > 0 and end - cur >= min_gap:
        gaps.append((cur, end))
    return gaps


def pad_windows(gaps, end, frac=PAD_FRAC, fixed=0.0):
    """Grow each gap by ±(frac × its own length + fixed), clip to [0, end], merge overlaps."""
    grown = sorted((max(0.0, s - frac * (e - s) - fixed), min(end, e + frac * (e - s) + fixed))
                   for s, e in gaps)
    out = []
    for s, e in grown:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def in_windows(t, windows):
    """Boolean mask: which times fall inside any [start, end] window."""
    t = np.asarray(t, dtype=float)
    if not len(windows):
        return np.zeros(t.shape, dtype=bool)
    w = np.asarray(windows, dtype=float)
    i = np.searchsorted(w[:, 0], t, side="right") - 1
    return (i >= 0) & (t <= w[np.clip(i, 0, None), 1])


def windows_for(dialogue, end, min_gap=GAP_MIN, frac=PAD_FRAC, fixed=0.0):
    return pad_windows(dialogue_gaps(dialogue, end, min_gap), end, frac, fixed)


def select_frames(variant, grid, windows):
    """A and C take the whole 1 FPS grid; B keeps only grid points inside the padded gaps."""
    return grid if variant in ("A", "C") else grid[in_windows(grid, windows)]


BUDGET_MARGIN = 64   # chat-template tokens around the two turns, with room to spare


def image_tokens(width, height):
    """Qwen3.5 visual tokens for one width x height frame (patch 16 x merge 2 = 32 px/token)."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    h, w = smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216)
    return (h // 32) * (w // 32)


def frame_tokens(width, height):
    """Sequence cost of one frame: image pads + <|vision_start|>/<|vision_end|> + its
    `[frame M:SS.s] ` label and newline — 11 tokens, because the space before
    <|vision_start|> is a token of its own (checked against exact tokenization of built
    records; measuring the label with `<image>` deleted merges that space and gives 10).
    125 at 448x252."""
    return image_tokens(width, height) + 2 + 11


def fit_frames(n_frames, text_tokens, budget, per_frame, margin=BUDGET_MARGIN):
    """How many of n_frames fit a prompt+target token budget next to the clip's text.
    budget 0 / None = no limit."""
    if not budget:
        return n_frames
    return int(max(0, min(n_frames, (budget - margin - text_tokens) // per_frame)))


def uniform_times(spans, n):
    """n frame times spread uniformly over the union of spans: the midpoints of n equal slices of
    the spans laid end to end. With the single span [0, end] that is n frames evenly over the whole
    clip — the 1 FPS grid's midpoint convention at a rate of n / end."""
    spans = [(float(s), float(e)) for s, e in spans if e > s]
    if n <= 0 or not spans:
        return np.array([])
    starts = np.array([s for s, _ in spans])
    cum = np.concatenate([[0.0], np.cumsum([e - s for s, e in spans])])
    u = (np.arange(n) + 0.5) * cum[-1] / n
    i = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(spans) - 1)
    return starts[i] + (u - cum[i])


def plan_frames(variant, grid, windows, end, n):
    """Frame times for one variant allowed n frames. Returns (times, on_grid).

    The variant's span is the whole clip (A, C) or the dialogue-gap windows (B). If the span's 1 FPS
    grid frames fit in n they are used as they are; otherwise n frames are sampled uniformly over
    the same span, so the model still sees all of it, at a lower frame rate."""
    base = select_frames(variant, grid, windows)
    if len(base) <= n:
        return base, True
    return uniform_times([(0.0, end)] if variant in ("A", "C") else windows, n), False


def even_cap(times, cap):
    """Keep at most `cap` frames, spread evenly over the clip — never by cutting the tail."""
    n = len(times)
    if cap is None or n <= cap:
        return times
    return times[np.round(np.linspace(0, n - 1, cap)).astype(int)]


# ---------------------------------------------------------------- coverage

def window_overlap(ads, windows):
    """Fraction of each AD's span [start, end] that lies inside the windows."""
    out = np.zeros(len(ads))
    for k, (s, e, *_) in enumerate(ads):
        if e <= s:
            out[k] = float(in_windows([s], windows)[0])
            continue
        tot = sum(max(0.0, min(e, we) - max(s, ws)) for ws, we in windows)
        out[k] = tot / (e - s)
    return out


def frames_in_ads(ads, times):
    """Number of sampled frames that land inside each AD's span."""
    times = np.asarray(times, dtype=float)
    s = np.array([a[0] for a in ads], dtype=float)
    e = np.array([a[1] for a in ads], dtype=float)
    return np.searchsorted(times, e, side="right") - np.searchsorted(times, s, side="left")


# ---------------------------------------------------------------- rendering

def registry_block(dense):
    """The character / location / prop registries, as in the text-only baseline."""
    parts = []
    for key, head in (("character_registry", "Characters"),
                      ("location_registry", "Locations"),
                      ("props_registry", "Props")):
        lines = []
        for it in dense.get(key) or []:
            if not isinstance(it, dict):
                continue
            bits = f"[{it.get('id', '')}] {it.get('name', '')}"
            desc = it.get("appearance") or it.get("description") or ""
            if desc:
                bits += f" — {desc}"
            if it.get("role"):
                bits += f" ({it['role']})"
            lines.append(bits)
        if lines:
            parts.append(f"## {head}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def render_user(variant, dense, events, dialogue, times):
    """User turn. A: frames + dialogue. B/C: registries + frames interleaved into the
    visual events by time (a frame precedes an event at the same timestamp) + dialogue."""
    frames = [(t, f"[frame {fmt(t)}] <image>") for t in times]
    blocks = []
    if variant == "A":
        if frames:
            blocks.append("## Frames\n" + "\n".join(line for _, line in frames))
    else:
        reg = registry_block(dense)
        if reg:
            blocks.append(reg)
        merged = sorted([(t, 0, line) for t, line in frames] +
                        [(t, 1, f"{fmt(t)}  {d}") for t, d in events],
                        key=lambda x: (x[0], x[1]))
        blocks.append("## Visual events\n" + "\n".join(line for *_, line in merged))
    # dialogue carries start-end (the gaps between lines are where ADs go); dense events
    # have no end time in the source, so they stay start-only
    blocks.append("## Dialogue\n" + ("\n".join(f"{fmt(s)}-{fmt(e)}  {d}" for s, e, d in dialogue)
                                     if dialogue else "(no dialogue)"))
    return PROMPTS[variant] + "\n\n" + "\n\n".join(blocks)


def render_target(dialogue, ads):
    """Assistant turn: dialogue verbatim with start-end, ADs with their start, in time order.
    ADs carry no end: GT AD duration follows from text length (r = 0.86 with word count)."""
    merged = ([(s, 0, f"{fmt(s)}-{fmt(e)} [DIALOGUE] {d}") for s, e, d in dialogue] +
              [(s, 1, f"{fmt(s)} [AD] {a}") for s, _, a in ads])
    # dialogue first on exact ties so speech anchors the timeline
    merged.sort(key=lambda x: (round(x[0], 1), x[1]))
    return "\n".join(line for *_, line in merged)


# ---------------------------------------------------------------- extraction

def frame_file(frame_root, vid, t, fps=FPS):
    """f_0000k.jpg holds the picture at t = (k + 0.5) / fps."""
    return f"{frame_root}/{vid}/f_{int(t * fps):05d}.jpg"


def resampled_file(frame_root, vid, t):
    """u_{milliseconds}.jpg holds the picture at t seconds, for frames off the 1 FPS grid."""
    return f"{frame_root}/{vid}/u_{int(round(t * 1000)):07d}.jpg"


def source_frame_times(video):
    """Presentation time of every video frame, from packet timestamps (no decoding)."""
    import av
    with av.open(video) as c:
        s = c.streams.video[0]
        return np.array(sorted(float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None))


def extract_at(video, frame_root, vid, times, width, height, quality=3, threads=0):
    """Write resampled_file() for each time: the last source frame at or before it — the same
    convention ffmpeg's fps filter follows for the 1 FPS grid. One decode pass, frames picked by
    index with ffmpeg's select filter, same scaling and JPEG quality as the grid frames."""
    out_dir = f"{frame_root}/{vid}"
    os.makedirs(out_dir, exist_ok=True)
    src = source_frame_times(video)
    idx = np.clip(np.searchsorted(src, times, side="right") - 1, 0, len(src) - 1)
    uniq = sorted({int(i) for i in idx})
    tmp = f"{out_dir}/_sel_%05d.jpg"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", str(threads),
         "-t", f"{src[uniq[-1]] + 1:.3f}", "-i", video,
         "-vf", "select='" + "+".join(f"eq(n,{i})" for i in uniq) + "'," + scale_filter(width, height),
         "-vsync", "vfr", "-q:v", str(quality), "-start_number", "0", tmp],
        check=True)
    got = sorted(f for f in os.listdir(out_dir) if f.startswith("_sel_"))
    assert len(got) == len(uniq), (video, len(got), len(uniq))
    pos = {i: k for k, i in enumerate(uniq)}
    for t, i in zip(times, idx):
        shutil.copyfile(tmp % pos[int(i)], resampled_file(frame_root, vid, t))
    for k in range(len(uniq)):
        os.remove(tmp % k)


def scale_filter(width, height):
    """Scale into width x height; non-16:9 sources are letterboxed, never stretched."""
    return (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black")


def extract_frames(video, out_dir, n_frames, width, height, fps=FPS, quality=3, threads=0):
    """Write f_00000.jpg (t=0.5 s), f_00001.jpg (t=1.5 s), ... — exactly n_frames of them.

    Input runs one second past the last frame: the fps filter only emits frame k once
    it has seen input up to k + 0.5 s, so cutting at the content end drops the last one.
    """
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", str(threads),
         "-t", f"{n_frames / fps + 1:.3f}", "-i", video,
         "-vf", f"fps={fps}," + scale_filter(width, height),
         "-frames:v", str(n_frames), "-q:v", str(quality),
         "-start_number", "0", f"{out_dir}/f_%05d.jpg"],
        check=True)
    return sorted(f for f in os.listdir(out_dir) if f.startswith("f_"))
