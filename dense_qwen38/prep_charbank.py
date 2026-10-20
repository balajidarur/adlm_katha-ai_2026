"""
Pre-VLM preprocessing for character-bank dense descriptions.

Does every step that does not need the VLM, once per clip, and stores the results so
any downstream model (Qwen, Gemini, ...) can build its requests without redoing them:

  shots     Gemini-run shot cache (same boundaries as the Gemini descriptions); optionally
            re-detected on this copy of the clip to check they still line up
  frames    3 FPS for every shot (context shots take every 3rd = 1 FPS), scaled to 360p,
            letterbox bars cropped                         frames/shot_XXX_NNNN.jpg
  faces     InsightFace buffalo_l boxes, landmarks, det scores, 512-d ArcFace embeddings
            for every face in every frame                  faces.npz
  matches   cosine vs the movie's top-10 cast portraits (>= 0.2, one face per identity),
            then shot-level smoothing                      faces.json
  circled   frames with a closed coloured ellipse per recognised face. No timestamp:
            it is relative to each request's window, so it is burnt at request time.
            Frames without a circle are hard links to frames/.  circled/shot_XXX_NNNN.jpg

Because embeddings are stored, changing the threshold, top-k or smoothing only needs
`--rematch` (CPU, seconds per clip), never the GPU pass again.

Layout: {out_root}/{year}/{video_id}/{meta.json, faces.npz, faces.json, frames/, circled/}
        {out_root}/_portraits/{actor_imdb_id}.{jpg,npz} and _crops/{actor_imdb_id}.jpg

Usage:
  python prep_charbank.py --clips clips.txt --gpus 0,1,2,3 --workers-per-gpu 4
"""

import os

# one thread per library per worker: parallelism comes from the worker pool
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DATA = os.environ.get("CMD_AD_ROOT", "/data/cmd_ad")
CSV_PATH = f"{DATA}/annotations/cmd_ad_anno_v1.csv"
META_PATH = f"{DATA}/annotations/character_bank/metadata.json"
SHOTS_CACHE = f"{DATA}/dense/shots_cache_gemini-3.5-flash-lite.json"
VIDEO_BASE = f"{DATA}/videos"

TARGET_FPS = 3
FRAME_HEIGHT = 360
JPEG_QUALITY = 92
TOP_K = 10
NVDEC = os.environ.get("PREP_NVDEC", "1") == "1"


# ---------------------------------------------------------------- portraits

def portrait_url(url, height=600):
    # IMDb image CDN resizes on request: ..._V1_.jpg -> ..._V1_UY600_.jpg
    return url.replace("._V1_.jpg", f"._V1_UY{height}_.jpg") if "._V1_.jpg" in url else url


def fetch(url, path):
    if os.path.exists(path):
        return True
    for u in (portrait_url(url), url):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=30).read()
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return True
        except Exception:
            continue
    return False


def build_portraits(actors, out_root, gpu):
    """Download + embed every needed actor portrait once (cached across movies)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cv2
    from char_bank import save_face_crop
    pdir, cdir = f"{out_root}/_portraits", f"{out_root}/_crops"
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    todo = {a: u for a, u in actors.items() if not os.path.exists(f"{pdir}/{a}.npz")}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(32) as ex:
        ok = dict(zip(todo, ex.map(lambda a: fetch(todo[a], f"{pdir}/{a}.jpg"), todo)))
    t_fetch = time.perf_counter() - t0
    app = load_face_app()
    t0 = time.perf_counter()
    n_face = 0
    for a in todo:
        img = cv2.imread(f"{pdir}/{a}.jpg") if ok.get(a) else None
        faces = app.get(img) if img is not None else []
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            save_face_crop(img, f.bbox, f"{cdir}/{a}.jpg")
            np.savez(f"{pdir}/{a}.npz", emb=f.normed_embedding.astype(np.float32), found=True)
            n_face += 1
        else:
            np.savez(f"{pdir}/{a}.npz", emb=np.zeros(512, np.float32), found=False)
    return {"actors": len(actors), "new": len(todo), "downloaded": sum(ok.values()), "face_found": n_face,
            "fetch_s": round(t_fetch, 2), "embed_s": round(time.perf_counter() - t0, 2)}


def load_face_app(ort_threads=1):
    import onnxruntime
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls()
    # insightface builds sessions with default options (one CPU thread per core each);
    # with many workers per box that oversubscribes the CPU, so cap the threads.
    from insightface.model_zoo import model_zoo
    _init = model_zoo.PickableInferenceSession.__init__

    def _capped_init(self, model_path, **kw):
        so = kw.get("sess_options") or onnxruntime.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = ort_threads
        kw["sess_options"] = so
        _init(self, model_path, **kw)
    model_zoo.PickableInferenceSession.__init__ = _capped_init
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
    return app


# ---------------------------------------------------------------- per clip

def probe(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", video],
                         capture_output=True, text=True).stdout
    j = json.loads(out)
    return j["streams"][0]["width"], j["streams"][0]["height"], float(j["format"]["duration"])


def frame_rate(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate,r_frame_rate,codec_name", "-of", "json", video],
                         capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = map(int, st["avg_frame_rate"].split("/"))
    return num / den, st["codec_name"]


def decode_clip_nvdec(video, shots, w, h, fps):
    """One NVDEC pass over the clip. Slots are every 1/3 s from the shot start while
    start + k/3 < end; each slot holds the last frame of the shot with pts < slot + 1/6 s,
    which is the frame the per-shot `ffmpeg -ss -to -vf fps=3` of extract_frames() picks
    (pixel-identical). ffmpeg 4.4 sometimes drops a shot's final slot; this rule keeps it.
    Assumes constant frame rate. Returns per-shot frame lists, or None on a count mismatch."""
    eps = 1e-6
    targets = []  # (shot_idx, slot time, decoded frame number)
    for si, s in enumerate(shots):
        n_first = int(np.ceil(s["start"] * fps - eps))
        n_last = int(np.ceil(s["end"] * fps - eps)) - 1
        if n_last < n_first:
            continue
        n_slots = max(1, int(np.ceil((s["end"] - s["start"]) * TARGET_FPS - eps)))
        for k in range(n_slots):
            n = int(np.ceil((s["start"] + (k + 0.5) / TARGET_FPS) * fps - eps)) - 1
            targets.append((si, s["start"] + k / TARGET_FPS, min(max(n, n_first), n_last)))
    want = sorted({n for _, _, n in targets})
    expr = "+".join(f"eq(n\\,{n})" for n in want)
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-to", f"{shots[-1]['end'] + 1:.3f}", "-i", video,
           "-vf", f"select={expr},scale_cuda={w}:{h},hwdownload,format=nv12",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    if n != len(want):
        return None
    arr = np.frombuffer(raw, np.uint8).reshape(n, h, w, 3)
    pos = {fn: k for k, fn in enumerate(want)}
    per_shot = [[] for _ in shots]
    for si, t, fn in targets:
        per_shot[si].append(arr[pos[fn]].copy())
    return per_shot


def decode_shot(video, start, end, w, h):
    """3 FPS frames of one shot as BGR arrays at 360p (same sampling as extract_frames)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                          "-i", video, "-vf", f"fps={TARGET_FPS},scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return list(np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3))


def detect_letterbox(frames, thresh=24, min_frac=0.3, min_share=0.1):
    """Crop box [x, y, w, h] of the picture inside black bars.

    A row (column) is picture if it is bright (> min_frac of its pixels above thresh) in at
    least min_share of the sampled frames. Watermarks in the bars cover well under min_frac of
    a row, and a few bright frames (an end card, a flash) can't cancel the crop."""
    stack = np.stack([f[::2, ::2].max(axis=2) for f in frames[::max(1, len(frames) // 120)]])
    bright = stack > thresh
    rows = (bright.mean(axis=2) > min_frac).mean(axis=0) >= min_share
    cols = (bright.mean(axis=1) > min_frac).mean(axis=0) >= min_share
    H, W = frames[0].shape[:2]
    if rows.sum() < 0.5 * len(rows) or cols.sum() < 0.5 * len(cols):
        return [0, 0, W, H]
    y0, y1 = np.where(rows)[0][[0, -1]] * 2
    x0, x1 = np.where(cols)[0][[0, -1]] * 2
    y0, x0 = int(y0) & ~1, int(x0) & ~1
    y1, x1 = min(H, int(y1) + 2), min(W, int(x1) + 2)
    return [x0, y0, x1 - x0, y1 - y0]


def recrop_clip(job):
    """Fix a finished clip whose crop was missed: crop the saved frames in place, shift the
    face boxes/landmarks, then redo matching + circles (CPU only, no GPU pass)."""
    import cv2
    clip_dir, imdbid, out_root, meta_path = job
    m = json.load(open(f"{clip_dir}/meta.json"))
    files = [f["file"] for s in m["shots"] for f in s["frames"]]
    imgs = [cv2.imread(f"{clip_dir}/frames/{f}") for f in files]
    x, y, w, h = detect_letterbox(imgs)
    H, W = imgs[0].shape[:2]
    if (w, h) == (W, H):
        return {"clip": clip_dir, "changed": False}
    for f, im in zip(files, imgs):
        cv2.imwrite(f"{clip_dir}/frames/{f}", np.ascontiguousarray(im[y:y + h, x:x + w]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    z = dict(np.load(f"{clip_dir}/faces.npz"))
    z["bbox"] = z["bbox"] - np.array([x, y, x, y], np.float32)
    z["kps"] = z["kps"] - np.array([x, y], np.float32)
    np.savez(f"{clip_dir}/faces.npz", **z)
    cx, cy = m["crop_xywh"][:2]
    m["crop_xywh"], m["frame_wh"] = [cx + x, cy + y, w, h], [w, h]
    m["recropped"] = True
    json.dump(m, open(f"{clip_dir}/meta.json", "w"), indent=1)
    match_and_draw(clip_dir, m, cast_for_movie(imdbid, json.load(open(meta_path)), out_root))
    return {"clip": clip_dir, "changed": True, "crop": m["crop_xywh"]}


def cast_for_movie(imdbid, meta, out_root):
    from char_bank import CIRCLE_COLOURS, slugify
    cast = []
    for i, e in enumerate(meta.get(imdbid, {}).values()):
        if i >= TOP_K:
            break
        if not e.get("characters"):
            continue
        p = f"{out_root}/_portraits/{e['imdb_id']}.npz"
        z = np.load(p) if os.path.exists(p) else None
        cast.append({"id": f"char_{slugify(e['characters'][0])}", "name": e["characters"][0], "actor": e["name"],
                     "actor_imdb_id": e["imdb_id"], "portrait_face_found": bool(z is not None and z["found"]),
                     "portrait_crop": f"_crops/{e['imdb_id']}.jpg" if z is not None and z["found"] else None,
                     "_emb": z["emb"] if z is not None else np.zeros(512, np.float32)})
    for i, c in enumerate(cast):
        c["colour"], c["bgr"] = CIRCLE_COLOURS[i % len(CIRCLE_COLOURS)]
    return cast


def match_and_draw(clip_dir, meta_clip, cast, write_images=True):
    """CPU stage: embeddings -> matches -> smoothing -> circled frames."""
    import cv2
    from char_bank import draw_circles, match_embeddings, smooth_shot_detections
    z = np.load(f"{clip_dir}/faces.npz")
    bank = np.stack([c["_emb"] for c in cast]) if cast else np.zeros((0, 512), np.float32)
    valid = [c["portrait_face_found"] for c in cast]
    fidx, starts = z["frame_idx"], np.searchsorted(z["frame_idx"], np.arange(meta_clip["n_frames"] + 1))
    out_shots, gi, n_circled = {}, 0, 0
    os.makedirs(f"{clip_dir}/circled", exist_ok=True)
    for s in meta_clip["shots"]:
        raw = []
        for _ in s["frames"]:
            a, b = starts[gi], starts[gi + 1]
            raw.append(match_embeddings(z["emb"][a:b], z["bbox"][a:b], z["det_score"][a:b], bank, valid)
                       if len(cast) and b > a else [])
            gi += 1
        sm = smooth_shot_detections(raw)
        out_shots[str(s["id"])] = [{"frame": f["file"], "t": f["t"], "n_faces": int(starts[f["idx"] + 1] - starts[f["idx"]]),
                                    "dets": d, "raw_dets": r} for f, d, r in zip(s["frames"], sm, raw)]
        if write_images:
            for f, d in zip(s["frames"], sm):
                dst = f"{clip_dir}/circled/{f['file']}"
                if os.path.lexists(dst):
                    os.remove(dst)
                if d:
                    img = cv2.imread(f"{clip_dir}/frames/{f['file']}")
                    cv2.imwrite(dst, draw_circles(img, d, cast), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    n_circled += 1
                else:
                    os.link(f"{clip_dir}/frames/{f['file']}", dst)
    cast_out = [{k: v for k, v in c.items() if k not in ("_emb", "bgr")} for c in cast]
    json.dump({"cast": cast_out, "score_thresh": 0.2, "smoothing": {"min_frames": 2, "strong_score": 0.35},
               "shots": out_shots}, open(f"{clip_dir}/faces.json", "w"))
    return n_circled


_APP = None


def init_worker(gpu_queue):
    gpu = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cv2
    cv2.setNumThreads(1)
    global _APP
    _APP = load_face_app()
    _APP.gpu = gpu


def process_clip(job):
    import cv2
    cmd_fn, imdbid, shots, out_root, meta_path, verify_shots = job
    year, vid = cmd_fn.split("/")
    clip_dir = f"{out_root}/{year}/{vid}"
    T = {}
    if os.path.exists(f"{clip_dir}/meta.json"):  # resume: already finished
        return {"clip": cmd_fn, "ok": True, "skipped": True, "frames": 0}
    try:
        video = f"{VIDEO_BASE}/{cmd_fn}.mp4"
        t = time.perf_counter()
        sw, sh_, dur = probe(video)
        w = int(round(sw * FRAME_HEIGHT / sh_ / 2)) * 2
        h = FRAME_HEIGHT
        fps, codec = frame_rate(video)
        shot_frames = decode_clip_nvdec(video, shots, w, h, fps) if NVDEC else None
        decoder = "nvdec_single_pass"
        if shot_frames is None:  # fallback: per-shot CPU decode, as in extract_frames()
            shot_frames = [decode_shot(video, s["start"], s["end"], w, h) for s in shots]
            decoder = "cpu_per_shot"
        per_shot = [len(fr) for fr in shot_frames]
        all_frames = [f for fr in shot_frames for f in fr]
        del shot_frames
        T["decode"] = time.perf_counter() - t

        t = time.perf_counter()
        x, y, cw, ch = detect_letterbox(all_frames)
        os.makedirs(f"{clip_dir}/frames", exist_ok=True)
        shots_meta, k = [], 0
        for s, n in zip(shots, per_shot):
            frames = []
            for i in range(n):
                name = f"shot_{s['id']:03d}_{i + 1:04d}.jpg"
                all_frames[k] = np.ascontiguousarray(all_frames[k][y:y + ch, x:x + cw])
                cv2.imwrite(f"{clip_dir}/frames/{name}", all_frames[k], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frames.append({"idx": k, "file": name, "t": round(s["start"] + i / TARGET_FPS, 3)})
                k += 1
            shots_meta.append({**s, "frames": frames})
        T["crop_write"] = time.perf_counter() - t

        t = time.perf_counter()
        fi, bb, kp, ds, em = [], [], [], [], []
        for k, img in enumerate(all_frames):
            for f in _APP.get(img):
                fi.append(k)
                bb.append(f.bbox)
                kp.append(f.kps)
                ds.append(f.det_score)
                em.append(f.normed_embedding)
        np.savez(f"{clip_dir}/faces.npz", frame_idx=np.array(fi, np.int32),
                 bbox=np.array(bb, np.float32).reshape(-1, 4), kps=np.array(kp, np.float32).reshape(-1, 5, 2),
                 det_score=np.array(ds, np.float32), emb=np.array(em, np.float16).reshape(-1, 512))
        T["faces_gpu"] = time.perf_counter() - t

        meta_clip = {"cmd_filename": cmd_fn, "imdbid": imdbid, "video": video, "source_wh": [sw, sh_],
                     "duration": dur, "shots_end": shots[-1]["end"], "fps": TARGET_FPS, "src_fps": fps, "decoder": decoder, "scaled_wh": [w, h],
                     "crop_xywh": [x, y, cw, ch], "frame_wh": [cw, ch], "n_frames": len(all_frames),
                     "n_faces": len(fi), "shots_source": "gemini shots cache", "shots": shots_meta}
        if verify_shots:
            from test_adlm_descriptions import detect_shots
            t = time.perf_counter()
            det = detect_shots(video)
            T["shot_redetect"] = time.perf_counter() - t
            cut_c = [s["start"] for s in shots[1:]]
            cut_d = [s["start"] for s in det[1:] if s["start"] < shots[-1]["end"]]
            meta_clip["shot_check"] = {"cache_cuts": len(cut_c), "redetected_cuts": len(cut_d),
                                       "matched_within_0.2s": sum(any(abs(a - b) <= 0.2 for b in cut_d) for a in cut_c)}

        t = time.perf_counter()
        meta = json.load(open(meta_path))
        cast = cast_for_movie(imdbid, meta, out_root)
        n_circled = match_and_draw(clip_dir, meta_clip, cast)
        T["match_smooth_circles"] = time.perf_counter() - t
        meta_clip["n_circled_frames"] = n_circled
        meta_clip["timings_s"] = {k: round(v, 3) for k, v in T.items()}
        json.dump(meta_clip, open(f"{clip_dir}/meta.json", "w"), indent=1)
        return {"clip": cmd_fn, "ok": True, "gpu": _APP.gpu, "frames": len(all_frames), "faces": len(fi),
                "circled": n_circled, "crop": [x, y, cw, ch], "src": [sw, sh_], "dur": dur,
                "shots_end": shots[-1]["end"], "check": meta_clip.get("shot_check"), "decoder": decoder, **meta_clip["timings_s"]}
    except Exception:
        return {"clip": cmd_fn, "ok": False, "error": traceback.format_exc()}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="file with one cmd_filename (year/video_id) per line")
    ap.add_argument("--out-root", default=f"{DATA}/prep_360p")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--verify-shots", action="store_true", help="also re-detect shots on this copy and compare")
    ap.add_argument("--rematch", action="store_true", help="only redo matching/smoothing/circles from faces.npz")
    ap.add_argument("--recrop", action="store_true", help="re-detect letterbox on finished clips and fix them (CPU)")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    clips = [l.strip() for l in open(args.clips) if l.strip()]
    fn2imdb = {}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            fn2imdb[r["cmd_filename"]] = r["imdbid"]
    meta = json.load(open(META_PATH))
    cache = json.load(open(SHOTS_CACHE))
    gpus = [int(g) for g in args.gpus.split(",")]
    os.makedirs(args.out_root, exist_ok=True)
    log = open(args.log or f"{args.out_root}/prep_log.jsonl", "a")

    t_all = time.perf_counter()
    actors = {}
    for c in clips:
        for i, e in enumerate(meta.get(fn2imdb.get(c), {}).values()):
            if i < TOP_K and e.get("picture_url"):
                actors[e["imdb_id"]] = e["picture_url"]
    ctx = mp.get_context("spawn")
    if not (args.rematch or args.recrop):
        with ctx.Pool(1) as p:
            pstats = p.apply(build_portraits, (actors, args.out_root, gpus[0]))
        print("portraits:", json.dumps(pstats), flush=True)
        log.write(json.dumps({"stage": "portraits", **pstats}) + "\n")

    jobs = []
    for c in clips:
        vid = c.split("/")[1]
        if vid not in cache or c not in fn2imdb:
            print("skip (no shots/imdb):", c, flush=True)
            continue
        jobs.append((c, fn2imdb[c], cache[vid], args.out_root, META_PATH, args.verify_shots))
    jobs.sort(key=lambda j: -j[2][-1]["end"])  # longest first

    if args.recrop:
        todo = [(f"{args.out_root}/{c}", imdbid, args.out_root, META_PATH) for c, imdbid, *_ in jobs
                if os.path.exists(f"{args.out_root}/{c}/meta.json")]
        with ctx.Pool(40) as pool:
            res = pool.map(recrop_clip, todo, chunksize=4)
        print(f"recrop: {len(res)} clips checked, {sum(r['changed'] for r in res)} cropped, "
              f"{time.perf_counter() - t_all:.1f}s", flush=True)
        log.write(json.dumps({"stage": "recrop", "checked": len(res), "changed": sum(r["changed"] for r in res)}) + "\n")
        return

    if args.rematch:
        for c, imdbid, _, out_root, _, _ in jobs:
            d = f"{out_root}/{c}"
            mc = json.load(open(f"{d}/meta.json"))
            match_and_draw(d, mc, cast_for_movie(imdbid, meta, out_root))
        print(f"rematched {len(jobs)} clips in {time.perf_counter() - t_all:.1f}s")
        return

    q = ctx.Manager().Queue()
    for g in gpus:
        for _ in range(args.workers_per_gpu):
            q.put(g)
    n_workers = len(gpus) * args.workers_per_gpu
    t_pool = time.perf_counter()
    done = frames = 0
    with ctx.Pool(n_workers, initializer=init_worker, initargs=(q,)) as pool:
        t_ready = time.perf_counter()
        for r in pool.imap_unordered(process_clip, jobs):
            done += 1
            frames += r.get("frames", 0)
            log.write(json.dumps(r) + "\n")
            log.flush()
            el = time.perf_counter() - t_ready
            print(f"[{done}/{len(jobs)}] {r['clip']} ok={r['ok']} frames={r.get('frames')} faces={r.get('faces')} "
                  f"{ {k: r[k] for k in ('decode', 'crop_write', 'faces_gpu', 'match_smooth_circles') if k in r} } "
                  f"| {frames / el:.0f} frames/s overall" + ("" if r["ok"] else "\n" + r["error"]), flush=True)
    summary = {"stage": "clips", "clips": len(jobs), "frames": frames, "workers": n_workers,
               "pool_start_s": round(t_ready - t_pool, 1), "wall_s": round(time.perf_counter() - t_ready, 1),
               "total_s": round(time.perf_counter() - t_all, 1)}
    print("summary:", json.dumps(summary), flush=True)
    log.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
