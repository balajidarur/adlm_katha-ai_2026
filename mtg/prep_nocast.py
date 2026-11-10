"""
Prep for videos without a cast (e.g. Mind the Gap vlogs/tutorials), in the prep_charbank.py layout so that
charbank_pipeline.py --prep-root ... --no-circles --no-cast can run on them.

Per video writes {out_root}/{key}/meta.json, faces.json (empty cast), frames/shot_NNN_MMMM.jpg (3 fps, 360p),
and, when a transcript is given, {srt_root}/{key}.srt (for WHISPERX_SRT).

    python prep_nocast.py --video /scratch/mtg/videos/CC-6RPCGguw.mp4 --key mtg/CC-6RPCGguw \
        --out-root /scratch/mtg/prep --srt-root /scratch/mtg/srt --mtg-jsonl /scratch/mtg/hf/data/annotations/test.jsonl
"""
import argparse
import ast
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_adlm_descriptions import detect_shots, extract_frames  # noqa: E402

TARGET_FPS = 3      # same as the CMD-AD prep
FRAME_HEIGHT = 360  # same as prep_360p


def write_srt(segments, path, offset=0.0, dur=None):
    """MTG transcript_segments -> SRT; with offset/dur (a chopped clip) times are shifted and clipped to [0, dur]."""
    import srt as srt_lib
    if isinstance(segments, str):  # the HF jsonl stores the list as a Python literal string
        segments = ast.literal_eval(segments)
    segs = []
    for s in segments:
        a, b = s["start_seconds"] - offset, s["end_seconds"] - offset
        if dur is not None:
            a, b = max(a, 0.0), min(b, dur)
        if b > a and s["text"].strip():
            segs.append((a, b, s["text"].strip()))
    subs = [srt_lib.Subtitle(i + 1, dt.timedelta(seconds=a), dt.timedelta(seconds=b), t) for i, (a, b, t) in enumerate(segs)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(srt_lib.compose(subs))
    return len(subs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--key", required=True, help="clip key used by charbank_pipeline --video, e.g. mtg/<youtube_id>")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--srt-root", default=None)
    ap.add_argument("--mtg-jsonl", default=None, help="MTG annotations jsonl; its transcript_segments become the SRT")
    ap.add_argument("--video-id", default=None, help="MTG video_id when the key is not mtg/<video_id> (chopped clips)")
    ap.add_argument("--offset", type=float, default=0.0, help="chopped clip: its start in the full video (s)")
    ap.add_argument("--clip-dur", type=float, default=None, help="chopped clip: its duration (s)")
    a = ap.parse_args()

    d = os.path.join(a.out_root, a.key)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    shots = detect_shots(a.video)  # ffmpeg scene 0.12, shots merged/split to 3-10 s, as for CMD-AD
    shots_meta, n = [], 0
    for s in shots:
        prefix = f"shot_{s['id']:03d}"
        fr = extract_frames(a.video, s["start"], s["end"], TARGET_FPS, os.path.join(d, "frames"), prefix, FRAME_HEIGHT)
        frames = [{"idx": n + i, "file": os.path.basename(fp), "t": round(t, 3)} for i, (fp, t) in enumerate(fr)]
        n += len(frames)
        shots_meta.append({**s, "frames": frames})
    meta = {"cmd_filename": a.key, "imdbid": None, "video": a.video, "fps": TARGET_FPS, "frame_height": FRAME_HEIGHT,
            "n_frames": n, "n_faces": 0, "shots_source": "detect_shots (ffmpeg scene 0.12, 3-10 s)", "shots": shots_meta}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    faces = {"cast": [], "shots": {str(s["id"]): [{"frame": f["file"], "t": f["t"], "dets": [], "raw_dets": []}
                                                  for f in s["frames"]] for s in shots_meta}}
    json.dump(faces, open(os.path.join(d, "faces.json"), "w"))
    n_subs = None
    if a.mtg_jsonl and a.srt_root:
        vid = a.video_id or a.key.split("/")[-1]
        row = next((json.loads(l) for l in open(a.mtg_jsonl) if json.loads(l)["video_id"] == vid), None)
        if row is not None:
            n_subs = write_srt(row["transcript_segments"] or [], os.path.join(a.srt_root, a.key + ".srt"),
                               a.offset, a.clip_dur)
    print(json.dumps({"key": a.key, "shots": len(shots), "frames": n, "srt_lines": n_subs,
                      "dur": shots[-1]["end"] if shots else 0}))


if __name__ == "__main__":
    main()
