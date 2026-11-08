#!/usr/bin/env python
"""build_frame_datasets.py, unchanged, on a box that has the frame pool but not the videos.

The builder opens each video only for its duration (video_info) and, when frames are missing, to extract
them. With the pool complete, the duration is the only thing it needs. Here video_info() returns the
duration the reference build recorded in its *_meta.json (video_dur; 1280x720 as on box A) and None for
clips the reference build could not open, so the clip set and every record match the reference build.
A clip whose pool has fewer 1 FPS frames than its grid (video shorter than the dense timeline: 20 of M3's clips)
makes the builder re-run ffmpeg; without the video, extract_frames() returns the frames already in the pool,
which is exactly what that ffmpeg run left there on box A. For those clips the video's duration ends the clip, and
the rounded video_dur would shift the uniformly resampled frame times; exact_dur() recovers it from the pool.

Validate by rebuilding with the reference dense descriptions and comparing records (E3 did this with the
Qwen3.8 build of frame_variants_448x252_full_qwen38). A clip that would need frames not in the pool fails
("No such file" from ffmpeg) and is listed in build_stats.json.

Clips whose video is present (e.g. fetched because the new descriptions need new resampled frames) use the
builder's own video_info / ffmpeg extraction unchanged.

  DUR_META=a_meta.json,b_meta.json DENSE_PATH_TMPL=... python build_frame_datasets_durs.py <builder args>
"""
import json
import os

import frames_lib as fl

import numpy as np

DUR, BOUND, NUNI = {}, set(), {}
for f in os.environ["DUR_META"].split(","):
    for m in json.load(open(f)):
        DUR[m["video_id"]] = m["video_dur"]
        if m["content_end"] == m["video_dur"]:
            BOUND.add(m["video_id"])        # the video, not the dense timeline, ends the clip
    # the uniform-resampled variants (A, C) of the same build: their frame counts pin the exact duration
    for v in ("frames_only", "dds_frames"):
        g = f.replace("dds_gapframes", v)
        if g != f and os.path.exists(g):
            for m in json.load(open(g)):
                if m["sampling"] == "uniform":
                    NUNI.setdefault(m["video_id"], set()).add(m["n_frames"])


def exact_dur(vid, d0, frame_root="/data/cmd_ad/frames/1fps_448x252"):
    """meta's video_dur is rounded to 0.01 s. When the video ends the clip, the resampled frames
    u_{ms}.jpg sit at (i + 0.5) * dur / n, so the pool pins dur: take the middle of the interval of
    durations whose resampled times all name frames in the pool (None found -> d0)."""
    try:
        have = {int(x[2:9]) for x in os.listdir(f"{frame_root}/{vid}") if x.startswith("u_")}
    except OSError:
        return d0
    cand = d0 + np.arange(-6000, 6001) * 1e-6
    ok = np.ones(len(cand), bool)
    for n in NUNI.get(vid, ()):
        ms = np.rint(((np.arange(n) + 0.5)[None, :] * cand[:, None] / n) * 1000).astype(int)
        ok &= np.isin(ms, list(have)).all(1)
    return float(cand[ok][len(cand[ok]) // 2]) if ok.any() and NUNI.get(vid) else d0


_EXACT = {}


_video_info = fl.video_info


def video_info(path):
    if os.path.exists(path):                 # the video is here after all: the builder's own reading
        return _video_info(path)
    vid = os.path.basename(path).rsplit(".", 1)[0]
    d = DUR.get(vid)
    if d is None:
        return None
    if vid in BOUND:
        if vid not in _EXACT:
            _EXACT[vid] = exact_dur(vid, float(d))
        d = _EXACT[vid]
    return {"dur": float(d), "w": 1280, "h": 720}


_extract_frames = fl.extract_frames


def extract_frames(video, out_dir, *args, **kwargs):
    if os.path.exists(video):
        return _extract_frames(video, out_dir, *args, **kwargs)
    return sorted(f for f in os.listdir(out_dir) if f.startswith("f_"))


fl.video_info = video_info          # build_frame_datasets uses this same module object
fl.extract_frames = extract_frames
# BUILD_VARIANTS=B builds only the gap-frames variant (the one M3 trains on). Each variant's record depends only on its
# own frame plan, so B's records are unchanged; it stops a clip from failing because variant C (all frames + the
# longer dense text) would need resampled frames that only the video can provide.
if os.environ.get("BUILD_VARIANTS"):
    fl.VARIANTS = tuple(os.environ["BUILD_VARIANTS"].split(","))

import build_frame_datasets  # noqa: E402

if __name__ == "__main__":
    print(f"video_info from DUR_META: {len(DUR)} clips")
    build_frame_datasets.main()
