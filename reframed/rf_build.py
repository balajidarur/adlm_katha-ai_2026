"""Build the frame-interleaved REFRAMED datasets with the CMD-AD builder, unchanged.

Runs dataset_prep/build_frame_datasets.py (variants A / B / C, 1 FPS 448x252, cap 200, token budget)
after pointing frames_lib at REFRAMED:
  clips   -> {train, validation, test} excerpts (video_id = excerpt id, "year" = IMDb movie id)
  videos  -> /scratch/reframed/{split}/excerpt/video/{movie}/{excerpt}.mp4
  dialogue-> /scratch2/rf_srt/srt/dialogue/{movie}/{excerpt}.srt (train: cleaned SRTs; val/test: raw subtitles)
  dense   -> /scratch2/dd_full/{excerpt}/Qwen3.8-27B-FP8/adlm_descriptions.json (scene-shot Qwen3.8 run)
  GT ADs  -> US: train = sentence ADs with measured word times (reframed_train/ad_word_times/ad_us),
             val/test = excerpt/audio_description_us JSON rows

  python rf_build.py --out /scratch/rf_datasets/frame_variants_448x252 --workers 44
"""

import glob
import json
import os
import sys

CODE = "/scratch2/codeA/dataset_prep"
RF = "/scratch/reframed"
SRT = "/scratch2/rf_srt/srt/dialogue"
WORDS = "/scratch2/rf_train/ad_word_times/ad_us"
os.environ.setdefault("DENSE_PATH_TMPL", "/scratch2/dd_full/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json")
sys.path.insert(0, CODE)
os.chdir(CODE)
import frames_lib as fl  # noqa: E402

SPLITS = ("train", "validation", "test")
VIDEO = {}
for s in SPLITS:
    for p in glob.glob(f"{RF}/{s}/excerpt/video/*/*.mp4"):
        VIDEO[os.path.basename(p)[:-4]] = (s, p.split("/")[-2], p)


def load_index():
    return [{"video_id": v, "year": movie, "split": s} for v, (s, movie, _) in sorted(VIDEO.items())]


def join_pieces(rows):
    """Val/test AD rows are pieces of sentences (no is_eos flag); join a piece with the next one when it
    does not end a sentence and the next starts lower-case, like the train SRTs (one AD = one sentence)."""
    out = []
    for s, e, t in sorted(rows):
        if out and not out[-1][2].rstrip().endswith((".", "!", "?", '"', "'")) and t[:1].islower():
            out[-1] = (out[-1][0], e, out[-1][2] + " " + t)
        else:
            out.append((s, e, t))
    return out


def load_ads():
    ads = {}
    for v, (s, movie, _) in VIDEO.items():
        if s == "train":
            p = f"{WORDS}/{movie}/{v}.json"
            rows = [(a["start"], a["end"], a["text"].strip()) for a in json.load(open(p))["ads"]] if os.path.exists(p) else []
        else:
            p = f"{RF}/{s}/excerpt/audio_description_us/{movie}/{v}.json"
            rows = join_pieces([(a["start"], a["end"], a["text"].strip()) for a in json.load(open(p)).values()]) if os.path.exists(p) else []
        ads[v] = sorted(r for r in rows if r[2])
    return ads


fl.ROOT = os.environ.get("RF_ROOT", "/scratch2/rf_root")       # holds dense/shots_cache_*.json for job ordering
fl.load_index = load_index
fl.load_ads = load_ads
fl.video_path = lambda year, vid: VIDEO[vid][2]
fl.asr_path = lambda year, vid: f"{SRT}/{year}/{vid}.srt"

_extract_frames_cpu = fl.extract_frames


def extract_frames_nvdec(video, out_dir, n_frames, width, height, fps=fl.FPS, quality=3, threads=0):
    """fl.extract_frames() with the decode on a GPU (NVDEC); same filters, byte-identical JPEGs
    (checked on a REFRAMED train clip). Workers spread over the GPUs by pid; CPU fallback on error."""
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", str(threads),
             "-hwaccel", "cuda", "-hwaccel_device", str(os.getpid() % int(os.environ.get("N_GPUS", "4"))),
             "-t", f"{n_frames / fps + 1:.3f}", "-i", video,
             "-vf", f"fps={fps}," + fl.scale_filter(width, height),
             "-frames:v", str(n_frames), "-q:v", str(quality),
             "-start_number", "0", f"{out_dir}/f_%05d.jpg"],
            check=True)
        return sorted(f for f in os.listdir(out_dir) if f.startswith("f_"))
    except subprocess.CalledProcessError:
        return _extract_frames_cpu(video, out_dir, n_frames, width, height, fps, quality, threads)


fl.extract_frames = extract_frames_nvdec

import build_frame_datasets as bfd  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--splits", *SPLITS, "--n-per-split", "0"] + sys.argv[1:]
    bfd.main()
