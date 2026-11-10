"""Prepare MTG official-eval inputs (run_ad_eval.py) for one system, with Piper TTS durations.

  --preds P.jsonl : our writer output (one row per video: video_id + predict, "M:SS.s [AD] <|budget_k|>text" lines)
  --auto-ad DIR   : or an existing auto_ad dir (output_{vid}.txt), e.g. a released baseline or gt_ad
Writes OUT/auto_ad/output_{vid}.txt ("HH:MM:SS.mmm-STANDARD text"), OUT/tts/{vid}_tts_metadata.json
(parse_ad_file segments, hashcode = md5(text), duration_tts = Piper en_US-lessac-medium at 1.2x),
OUT/visual_info/output_{vid}.txt (annotation field), OUT/ids.txt.
  python mtg_eval_prep.py --preds p.jsonl --ids ids.json --anno-dir ANNO --out OUT
"""
import argparse, json, os, re, subprocess, sys, tempfile

MTG_REPO = os.environ.get("MTG_REPO", "/workspace/dbalu/tasks_nov8/T4/mtg-ad")
PIPER_PY = os.environ.get("PIPER_PY", "/workspace/dbalu/piper/venv/bin/python")
FAST_TTS = os.environ.get("FAST_TTS", "/storage/users/dbalu/mtg_temporal/reframed/fast_tts.py")
VOICE = os.environ.get("PIPER_VOICE", "/workspace/dbalu/piper/en_US-lessac-medium.onnx")
sys.path.insert(0, MTG_REPO)
from common import tts  # noqa: E402

BUD = re.compile(r"<\|budget_\d\|>")
AD = re.compile(r"^(\d+):(\d+(?:\.\d+)?)(?:-\d+:\d+(?:\.\d+)?)? \[AD\] ?(.*)$")


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def ours_to_mtg(predict):
    out = []
    for l in predict.split("\n"):
        m = AD.match(l.strip())
        if m:
            txt = BUD.sub("", m.group(3)).strip()
            if txt:
                out.append(f"{hms(int(m.group(1)) * 60 + float(m.group(2)))}-STANDARD {txt}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds"); ap.add_argument("--auto-ad"); ap.add_argument("--ids", required=True)
    ap.add_argument("--anno-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    ids = json.load(open(a.ids))
    for d in ("auto_ad", "tts", "visual_info"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    vi = {}
    for f in ("validation", "test"):
        for l in open(f"{a.anno_dir}/{f}.jsonl"):
            r = json.loads(l); vi[r["video_id"]] = r.get("visual_info") or ""
    if a.preds:
        P = {}
        for l in open(a.preds):
            r = json.loads(l); P[r["video_id"]] = ours_to_mtg(r["predict"])
    texts, segs = set(), {}
    for v in ids:
        if a.preds:
            ad = P.get(v, "")
        else:
            p = f"{a.auto_ad}/output_{v}.txt"; ad = open(p).read() if os.path.exists(p) else ""
        open(f"{a.out}/auto_ad/output_{v}.txt", "w").write(ad)
        open(f"{a.out}/visual_info/output_{v}.txt", "w").write(vi.get(v, ""))
        try:
            segs[v] = tts.parse_ad_file(ad) if ad.strip() else []
        except Exception:
            segs[v] = []
        texts |= {t for _, t in segs[v]}
    with tempfile.TemporaryDirectory() as td:
        json.dump(sorted(texts), open(f"{td}/t.json", "w"))
        subprocess.run([PIPER_PY, FAST_TTS, f"{td}/t.json", f"{td}/d.json", VOICE, str(a.workers)], check=True)
        D = json.load(open(f"{td}/d.json"))
    n_seg = 0
    for v in ids:
        info = []
        for ts, t in segs[v]:
            s = tts.timestamp_to_seconds(ts)
            info.append({"timestamp": ts, "start": s, "text": t, "hashcode": tts.text_hash(t),
                         "duration_tts": D[t], "end_tts": s + D[t], "tts_backend": "piper:en_US-lessac-medium@1.2x"})
        n_seg += len(info)
        if info:
            json.dump(info, open(f"{a.out}/tts/{v}_tts_metadata.json", "w"), indent=1)
    open(f"{a.out}/ids.txt", "w").write("\n".join(ids) + "\n")
    print(f"{len(ids)} videos, {n_seg} AD segments, {sum(not segs[v] for v in ids)} without parsable ADs")


if __name__ == "__main__":
    main()
