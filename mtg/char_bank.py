"""
Character bank + face recognition for ADLM dense descriptions (AutoAD-Zero style).

AutoAD-Zero (Xie et al., 2024) recipe, reproduced here:
  - Top-10 billed IMDb cast per movie, one profile portrait per actor
  - InsightFace buffalo_l (RetinaFace det + ArcFace rec), det_thresh=0.4
  - Cosine similarity between in-frame faces and portrait faces, keep >= 0.2
  - One face per identity per frame (highest score wins)
  - Recognised faces are circled in a per-character colour; the prompt lists
    "Name (colour)" and tells the model never to mention the circles

Additions over the paper:
  - Shot-level smoothing: an identity seen in only one frame of a shot with a
    weak score is treated as a false positive and dropped
  - Portrait face crops are also passed to the VLM (labelled with name + colour)
"""

import csv
import json
import os
import re

import cv2
import numpy as np

CHAR_BANK_META = os.environ.get("CHAR_BANK_META", "/storage/datasets/CMD-AD/data/character_bank/metadata.json")
CHAR_BANK_IMAGES = os.environ.get("CHAR_BANK_IMAGES", "/storage/datasets/CMD-AD/data/character_bank/images")
CMD_AD_CSV = os.environ.get("CMD_AD_CSV", "/storage/datasets/CMD-AD/data/cmd_ad_anno_v1.csv")

# (name, BGR) — chosen to be mutually distinguishable and distinct from the white timestamp
CIRCLE_COLOURS = [
    ("red", (0, 0, 255)),
    ("green", (0, 200, 0)),
    ("blue", (255, 80, 0)),
    ("yellow", (0, 230, 255)),
    ("cyan", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("orange", (0, 140, 255)),
    ("purple", (160, 32, 120)),
    ("pink", (180, 150, 255)),
    ("brown", (30, 75, 140)),
]


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def imdbid_for_clip(cmd_filename, csv_path=CMD_AD_CSV):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["cmd_filename"] == cmd_filename:
                return row["imdbid"], row["movie_title"]
    return None, None


def load_cast(imdbid, top_k=10, meta_path=CHAR_BANK_META, images_dir=CHAR_BANK_IMAGES):
    """Top-k billed cast (billing order) that have a portrait on disk."""
    with open(meta_path) as f:
        meta = json.load(f)
    cast = []
    for i, entry in enumerate(meta.get(imdbid, {}).values()):
        if i >= top_k:
            break
        portrait = os.path.join(images_dir, f"{entry['imdb_id']}.jpg")
        if not os.path.exists(portrait) or not entry.get("characters"):
            continue
        char_name = entry["characters"][0]
        cast.append({
            "id": f"char_{slugify(char_name)}",
            "name": char_name,
            "actor": entry["name"],
            "actor_imdb_id": entry["imdb_id"],
            "portrait": portrait,
        })
    for i, c in enumerate(cast):
        c["colour"], c["bgr"] = CIRCLE_COLOURS[i % len(CIRCLE_COLOURS)]
    return cast


class FaceMatcher:
    def __init__(self, det_size=640, det_thresh=0.4, score_thresh=0.2, ctx_id=0):
        import onnxruntime
        if hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()  # pick up pip-installed CUDA/cuDNN libs
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=det_thresh)
        self.score_thresh = score_thresh
        self.bank = None  # (N, 512) normalised portrait embeddings
        self.bank_valid = []

    def build_bank(self, cast, crop_dir=None):
        """Embed the largest face in each portrait; optionally save a face crop per actor."""
        feats, valid = [], []
        for c in cast:
            img = cv2.imread(c["portrait"])
            faces = self.app.get(img) if img is not None else []
            if not faces:
                feats.append(np.zeros(512, np.float32))
                valid.append(False)
                c["portrait_face_found"] = False
                continue
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            feats.append(f.normed_embedding.astype(np.float32))
            valid.append(True)
            c["portrait_face_found"] = True
            if crop_dir:
                os.makedirs(crop_dir, exist_ok=True)
                c["portrait_crop"] = save_face_crop(img, f.bbox, os.path.join(crop_dir, f"{c['id']}.jpg"))
        self.bank = np.stack(feats)
        self.bank_valid = valid
        return cast

    def match_frame(self, img):
        """Return [{bbox, cast_idx, score, det_score}] — at most one face per identity."""
        faces = self.app.get(img)
        if not faces or self.bank is None:
            return [], len(faces)
        emb = np.stack([f.normed_embedding for f in faces])
        out = match_embeddings(emb, np.stack([f.bbox for f in faces]), np.array([f.det_score for f in faces]),
                               self.bank, self.bank_valid, self.score_thresh)
        return out, len(faces)


def match_embeddings(emb, bboxes, det_scores, bank, bank_valid, score_thresh=0.2):
    """Match one frame's face embeddings (F, 512) against the cast bank (N, 512).

    Greedy assignment by descending similarity == "duplicate identities keep the top match".
    """
    if len(emb) == 0:
        return []
    sim = emb.astype(np.float32) @ bank.T  # (faces, cast)
    sim[:, ~np.asarray(bank_valid, bool)] = -1
    out, used_faces, used_ids = [], set(), set()
    for fi, ci in sorted(np.ndindex(sim.shape), key=lambda ij: -sim[ij]):
        if sim[fi, ci] < score_thresh:
            break
        if fi in used_faces or ci in used_ids:
            continue
        used_faces.add(fi)
        used_ids.add(ci)
        out.append({
            "bbox": [round(float(v), 1) for v in bboxes[fi]],
            "cast_idx": int(ci),
            "score": round(float(sim[fi, ci]), 3),
            "det_score": round(float(det_scores[fi]), 3),
        })
    return out


def save_face_crop(img, bbox, path, margin=0.45, size=224):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(bw, bh) * (1 + margin) / 2
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    crop = cv2.resize(img[y1:y2, x1:x2], (size, size))
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def smooth_shot_detections(frame_dets, min_frames=2, strong_score=0.35):
    """Drop identities that appear in < min_frames frames of a shot unless strongly matched.

    frame_dets: list (per frame) of match_frame() outputs. Returns filtered copy.
    """
    counts, best = {}, {}
    for dets in frame_dets:
        for d in dets:
            counts[d["cast_idx"]] = counts.get(d["cast_idx"], 0) + 1
            best[d["cast_idx"]] = max(best.get(d["cast_idx"], 0), d["score"])
    keep = {ci for ci in counts if counts[ci] >= min_frames or best[ci] >= strong_score}
    return [[d for d in dets if d["cast_idx"] in keep] for dets in frame_dets]


def draw_circles(img, dets, cast, thickness=None):
    """Draw a coloured ellipse around each recognised face (in place).

    Where the ellipse leaves the frame, its points are clamped to the frame edge, so the
    out-of-frame arc becomes a straight segment along the border and the shape stays closed.
    """
    h, w = img.shape[:2]
    t = thickness or max(2, h // 160)
    lo = t // 2
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        centre = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int((x2 - x1) * 0.65), int((y2 - y1) * 0.75))
        pts = cv2.ellipse2Poly(centre, axes, 0, 0, 360, 2)
        pts[:, 0] = np.clip(pts[:, 0], lo, w - 1 - lo)
        pts[:, 1] = np.clip(pts[:, 1], lo, h - 1 - lo)
        cv2.polylines(img, [pts], True, cast[d["cast_idx"]]["bgr"], t, cv2.LINE_AA)
    return img


def cast_prompt_block(cast):
    lines = []
    for c in cast:
        note = "" if c.get("portrait_face_found", True) else "  (no usable portrait — never circled)"
        lines.append(f"- {c['name']} [{c['id']}] — played by {c['actor']} — circle colour: {c['colour'].upper()}{note}")
    return "\n".join(lines)


def shot_face_summary(frame_dets_with_time, cast, time_fmt):
    """Text summary: which cast members were recognised in a shot and when."""
    seen = {}
    for t, dets in frame_dets_with_time:
        for d in dets:
            seen.setdefault(d["cast_idx"], []).append((t, d["score"]))
    if not seen:
        return "No cast faces recognised."
    lines = []
    for ci, hits in sorted(seen.items(), key=lambda kv: kv[1][0][0]):
        c = cast[ci]
        lines.append(f"- {c['name']} [{c['id']}] ({c['colour']} circle): {len(hits)} frames, "
                     f"{time_fmt(hits[0][0])}–{time_fmt(hits[-1][0])}, max sim {max(s for _, s in hits):.2f}")
    return "\n".join(lines)
