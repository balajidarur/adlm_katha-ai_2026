#!/usr/bin/env python
"""build_frame_datasets.py, unchanged, on a box that has the frame pool but not the videos.

The builder opens each video only for its duration (video_info) and, when frames are missing, to extract
them. With the pool complete, the duration is the only thing it needs. Here video_info() returns the
duration the reference build recorded in its *_meta.json (video_dur; 1280x720 as on box A) and None for
clips the reference build could not open, so the clip set and every record match the reference build.

Validate by rebuilding with the reference dense descriptions and comparing records (E3 did this with the
Qwen3.8 build of frame_variants_448x252_full_qwen38). A clip that would need frames not in the pool fails
("No such file" from ffmpeg) and is listed in build_stats.json.

  DUR_META=a_meta.json,b_meta.json DENSE_PATH_TMPL=... python build_frame_datasets_durs.py <builder args>
"""
import json
import os

import frames_lib as fl

DUR = {}
for f in os.environ["DUR_META"].split(","):
    for m in json.load(open(f)):
        DUR[m["video_id"]] = m["video_dur"]


def video_info(path):
    d = DUR.get(os.path.basename(path).rsplit(".", 1)[0])
    return None if d is None else {"dur": float(d), "w": 1280, "h": 720}


fl.video_info = video_info          # build_frame_datasets uses this same module object

import build_frame_datasets  # noqa: E402

if __name__ == "__main__":
    print(f"video_info from DUR_META: {len(DUR)} clips")
    build_frame_datasets.main()
