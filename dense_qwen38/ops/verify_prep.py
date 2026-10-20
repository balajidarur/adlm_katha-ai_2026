"""Check finished prep clips have everything charbank_pipeline.py --prep-root needs."""
import glob, json, os, random, collections
import numpy as np
ROOT = "/prep/prep_360p"
clips_all = [l.strip() for l in open("/home/ubuntu/prep_code/clips_all.txt") if l.strip()]
done = [c for c in clips_all if os.path.exists(f"{ROOT}/{c}/meta.json")]
random.seed(0); sample = random.sample(done, min(400, len(done)))
P = collections.Counter(); bad = collections.defaultdict(list)
for c in sample:
    d = f"{ROOT}/{c}"
    m = json.load(open(f"{d}/meta.json")); f = json.load(open(f"{d}/faces.json")); z = np.load(f"{d}/faces.npz")
    files = [fr["file"] for s in m["shots"] for fr in s["frames"]]
    ok = True
    def chk(cond, name):
        global ok
        P[name] += bool(cond)
        if not cond: bad[name].append(c)
    chk(len(files) == m["n_frames"] and len(m["shots"]) > 0, "meta shots/frames consistent")
    chk(all(os.path.exists(f"{d}/frames/{x}") for x in files), "all plain frames on disk")
    chk(all(os.path.exists(f"{d}/circled/{x}") for x in files), "all circled frames on disk")
    chk(all(len(s["frames"]) >= 1 for s in m["shots"]), "every shot has >=1 frame")
    chk(all(abs(fr["t"] - (s["start"] + i / 3)) < 1e-3 for s in m["shots"] for i, fr in enumerate(s["frames"])), "frame times = start + i/3 (3 FPS)")
    chk(set(f["shots"]) == {str(s["id"]) for s in m["shots"]} and all(len(f["shots"][str(s["id"])]) == len(s["frames"]) for s in m["shots"]), "faces.json covers every frame")
    chk(len(f["cast"]) > 0 and all("colour" in x for x in f["cast"]), "cast list with colours")
    chk(all(os.path.exists(f"{ROOT}/{x['portrait_crop']}") for x in f["cast"] if x.get("portrait_crop")), "portrait crops on disk")
    chk(len(z["emb"]) == m["n_faces"], "faces.npz matches n_faces")
    w, h = m["frame_wh"]; chk(h <= 360 and w <= 640, "frame size <= 640x360")
    chk(os.path.exists(f"/data/cmd_ad/asr/SRT/{c}.srt"), "WhisperX SRT exists")
print(f"finished clips: {len(done)}/{len(clips_all)}; checked a random {len(sample)}")
for k, v in P.items():
    print(f"  {v:>4}/{len(sample)}  {k}" + (f"   e.g. {bad[k][:3]}" if bad[k] else ""))
srt_all = sum(os.path.exists(f"/data/cmd_ad/asr/SRT/{c}.srt") for c in clips_all)
print(f"WhisperX SRT available for {srt_all}/{len(clips_all)} clips (whole dataset)")
cast_ok = collections.Counter()
for c in sample:
    cast = json.load(open(f"{ROOT}/{c}/faces.json"))["cast"]
    cast_ok[sum(1 for x in cast if x.get("portrait_face_found"))] += 1
print("usable portraits per clip (of top-10):", dict(sorted(cast_ok.items())))
cropped = sum(json.load(open(f"{ROOT}/{c}/meta.json"))["frame_wh"][1] < 360 for c in sample)
print(f"letterbox cropped: {cropped}/{len(sample)}")
