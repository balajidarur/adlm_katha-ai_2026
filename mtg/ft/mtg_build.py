"""Build MTG (Mind the Gap) writer datasets with the CMD-AD builder, unchanged (same pattern as reframed/rf_build.py).

  clips    -> split.json {train (279, MTG validation 90%), val10 (34), test (97)}; one sample = one whole video
  videos   -> $MTG/videos_480p/{validation,test}/{vid}.mp4
  dialogue -> $MTG/srt/{vid}.srt written from the official transcript_segments
  dense    -> $MTG/dd/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json (T4 run nocast_mtg_b)
  GT ADs   -> gt_ad_segments: start = start_seconds, end = start + Piper (en_US-lessac-medium, 1.2x) duration
              (MTG gives only starts; the official eval also derives AD length from TTS at 1.2x)

  python mtg_build.py --out $MTG/ds --cap 250 --token-budget 56000 --workers 40
"""
import glob, json, os, sys

MTG = os.environ.get("MTG", "/scratch/mtg")
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset")
os.environ.setdefault("DENSE_PATH_TMPL", MTG + "/dd/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json")
sys.path.insert(0, os.path.abspath(CODE)); os.chdir(os.path.abspath(CODE))
import frames_lib as fl  # noqa: E402

SPLIT = json.load(open(f"{MTG}/split.json"))
ANNO = {}
for f in ("validation", "test"):
    for l in open(f"{MTG}/hf/data/{f}.jsonl" if os.path.exists(f"{MTG}/hf/data/{f}.jsonl") else f"{MTG}/anno/{f}.jsonl"):
        r = json.loads(l); ANNO[r["video_id"]] = r
TTS = json.load(open(f"{MTG}/gt_tts.json"))
VIDEO = {}
for s in ("train", "val10", "test"):
    for v in SPLIT[s]:
        src = "test" if s == "test" else "validation"
        VIDEO[v] = (s, f"{MTG}/videos_480p/{src}/{v}.mp4")


def srt_ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s - int(s)) * 1000)) % 1000:03d}"


def write_srts():
    os.makedirs(f"{MTG}/srt", exist_ok=True)
    for v in VIDEO:
        segs = ANNO[v]["transcript_segments"]
        with open(f"{MTG}/srt/{v}.srt", "w") as fh:
            for i, s in enumerate(segs, 1):
                fh.write(f"{i}\n{srt_ts(s['start_seconds'])} --> {srt_ts(s['end_seconds'])}\n{s['text'].strip()}\n\n")


def load_index():
    return [{"video_id": v, "year": "mtg", "split": s} for v, (s, _) in sorted(VIDEO.items())]


def load_ads():
    out = {}
    for v in VIDEO:
        rows = [(a["start_seconds"], a["start_seconds"] + TTS[a["text"].strip()], a["text"].strip())
                for a in ANNO[v]["gt_ad_segments"] if a["text"].strip()]
        out[v] = sorted(rows)
    return out


fl.ROOT = f"{MTG}/root"
fl.DENSE_MODEL = "Qwen3.8-27B-FP8"
fl.load_index = load_index
fl.load_ads = load_ads
fl.video_path = lambda year, vid: VIDEO[vid][1]
fl.asr_path = lambda year, vid: f"{MTG}/srt/{vid}.srt"

import build_frame_datasets as bfd  # noqa: E402

if __name__ == "__main__":
    write_srts()
    os.makedirs(f"{fl.ROOT}/dense", exist_ok=True)   # builder orders jobs longest-first from this cache
    json.dump({v: [{"end": float(ANNO[v]["duration_seconds"])}] for v in VIDEO},
              open(f"{fl.ROOT}/dense/shots_cache_{fl.DENSE_MODEL}.json", "w"))
    sys.argv = [sys.argv[0], "--splits", "train", "val10", "test", "--n-per-split", "0"] + sys.argv[1:]
    bfd.main()
