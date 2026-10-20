"""
Character-bank-aware ADLM dense descriptions with an open VLM served by vLLM.

Per video:
  1. Shots           — PySceneDetect (or the existing shots cache)
  2. Frames          — 3 FPS, 480p, extracted once per shot
  3. Face pre-pass   — InsightFace buffalo_l vs top-10 cast portraits (AutoAD-Zero recipe)
  4. Per-shot calls  — sequential (shot N sees shot N-1 registries), frames carry coloured
                       face circles + burnt timestamps, portraits + cast list in the prompt
  5. Outputs         — adlm_descriptions.json, adlm_timeline.json, faces.json, timings.json

Usage:
    python charbank_pipeline.py --video 2011/Od1ZBTstSHg \
        --server http://localhost:8000/v1 --model Qwen/Qwen3.8-27B-FP8
    # replay all built requests concurrently to measure server throughput
    python charbank_pipeline.py --video 2011/Od1ZBTstSHg --bench-only --bench-concurrency 1 4 8 17
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from char_bank import (FaceMatcher, cast_prompt_block, draw_circles, imdbid_for_clip,
                       load_cast, shot_face_summary, smooth_shot_detections)
from prompts_adlm_charbank import (ADLM_CHARBANK_CHUNK_PROMPT, ADLM_CHARBANK_SCHEMA,
                                   ADLM_CHARBANK_SYSTEM_PROMPT, PROMPT_VARIANTS,
                                   ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS, ADLM_CHARBANK_SCHEMA_NOTAGS,
                                   ADLM_CHARBANK_SYSTEM_PROMPT_NOTAGS, drop_circle_leaks)
from test_adlm_descriptions import (build_merged_timeline, detect_shots, events_to_summary,
                                    extract_frames, format_mmss, get_shifted_subtitles, subsample)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPERX_SRT = os.environ.get("WHISPERX_SRT", "/workspace/.common_cache/common_share_for_users/srt_for_ads/datasets/cmd_ad/whisperx_ASR_subtitles/SRT")
CMD_VIDEO_BASE = os.environ.get("CMD_VIDEO_BASE", "/storage/datasets/CMD/data/videos")


def gemini_schema_to_json_schema(s):
    """prompts_adlm schemas use Gemini's upper-case types; vLLM wants standard JSON Schema."""
    if isinstance(s, dict):
        out = {}
        for k, v in s.items():
            out[k] = v.lower() if k == "type" and isinstance(v, str) else gemini_schema_to_json_schema(v)
        return out
    if isinstance(s, list):
        return [gemini_schema_to_json_schema(x) for x in s]
    return s


def jpeg_data_url(img, quality=85):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def burn_ts(img, ts):
    h = img.shape[0]
    fs, th = max(0.5, h / 600), max(1, int(h / 300))
    cv2.putText(img, ts, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2)
    cv2.putText(img, ts, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    return img


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.t[name] = timer.t.get(name, 0) + time.perf_counter() - self.s
        return _Ctx()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="cmd_filename, e.g. 2011/Od1ZBTstSHg")
    ap.add_argument("--server", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--shots-cache", default=os.path.join(SCRIPT_DIR, "output/shots_cache_gemini-3.5-flash-lite.json"))
    ap.add_argument("--out-root", default=os.path.join(SCRIPT_DIR, "output_charbank"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-portraits", action="store_true")
    ap.add_argument("--no-circles", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--faces-only", action="store_true", help="stop after the face pre-pass")
    ap.add_argument("--prep-root", default=None,
                    help="use prep_charbank.py outputs ({root}/{year}/{id}/): shots, 360p frames, faces, circles")
    ap.add_argument("--prompt-variant", default="base", choices=sorted(PROMPT_VARIANTS))
    ap.add_argument("--no-prop-loc-tags", action="store_true", help="props/locations in plain words, no [prop_]/[loc_] tags or registries")
    ap.add_argument("--no-save-requests", action="store_true", help="don't write requests/*.json (full-dataset runs)")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--bench-concurrency", type=int, nargs="*", default=[])
    args = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(SCRIPT_DIR, "config.yaml"))
    p = cfg.pipeline
    vid = args.video.split("/")[-1]
    video_path = next((v for v in (os.path.join(CMD_VIDEO_BASE, f"{args.video}{ext}") for ext in (".mkv", ".mp4"))
                       if os.path.exists(v)), None)
    model_tag = args.model.split("/")[-1]
    out_dir = os.path.join(args.out_root, vid, model_tag)
    frames_dir = os.path.join(out_dir, "frames")
    circled_dir = os.path.join(out_dir, "frames_circled")
    req_dir = os.path.join(out_dir, "requests")
    for d in (out_dir,) + (() if args.no_save_requests else (req_dir,)) + (() if args.prep_root else (frames_dir, circled_dir)):
        os.makedirs(d, exist_ok=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.server, api_key="EMPTY", timeout=1800)
    json_schema = gemini_schema_to_json_schema(ADLM_CHARBANK_SCHEMA_NOTAGS if args.no_prop_loc_tags else ADLM_CHARBANK_SCHEMA)
    system_base = ADLM_CHARBANK_SYSTEM_PROMPT_NOTAGS if args.no_prop_loc_tags else ADLM_CHARBANK_SYSTEM_PROMPT
    chunk_tmpl = ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS if args.no_prop_loc_tags else ADLM_CHARBANK_CHUNK_PROMPT

    def call(messages):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=args.model, messages=messages, temperature=cfg.model.temperature,
            max_tokens=args.max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": "adlm_shot", "schema": json_schema}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        dt = time.perf_counter() - t0
        return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens, dt

    if args.bench_only:
        run_bench(call, req_dir, out_dir, args.bench_concurrency)
        return

    T = Timer()
    srt_path = os.path.join(WHISPERX_SRT, f"{args.video}.srt")
    srt_path = srt_path if os.path.exists(srt_path) else ""

    if args.prep_root:
        shots, shot_frames, faces, cast, imdbid, title, n_frames, n_raw_faces = load_prep(args, T)
    else:
        # 1. shots
        with T("1_shot_detection"):
            cache = json.load(open(args.shots_cache)) if os.path.exists(args.shots_cache) else {}
            if vid in cache:
                shots = cache[vid]
            else:
                shots = detect_shots(video_path, p.scene_threshold, p.max_shot_duration, p.min_shot_duration,
                                     p.shot_duration_tolerance, p.shot_absorb_threshold)

        # 2. frames (3 FPS for every shot; context shots are sub-sampled to 1 FPS later)
        shot_frames = {}
        with T("2_frame_extraction"):
            for s in shots:
                prefix = f"shot_{s['id']:03d}"
                existing = sorted(glob.glob(os.path.join(frames_dir, f"{prefix}_[0-9]*.jpg")))
                if not existing:
                    existing = [fp for fp, _ in extract_frames(video_path, s["start"], s["end"], p.target_fps,
                                                               frames_dir, prefix, p.frame_height)]
                shot_frames[s["id"]] = [(fp, s["start"] + i / p.target_fps) for i, fp in enumerate(existing)]
        n_frames = sum(len(v) for v in shot_frames.values())

        # 3. face pre-pass
        imdbid, title = imdbid_for_clip(args.video)
        with T("3a_face_model_load"):
            matcher = FaceMatcher()
        with T("3b_portrait_bank"):
            cast = matcher.build_bank(load_cast(imdbid, args.top_k), crop_dir=os.path.join(out_dir, "portraits"))
        faces = {}
        n_raw_faces = 0
        with T("3c_face_matching"):
            for s in shots:
                dets = []
                for fp, _ in shot_frames[s["id"]]:
                    d, n = matcher.match_frame(cv2.imread(fp))
                    dets.append(d)
                    n_raw_faces += n
                raw = dets
                dets = smooth_shot_detections(dets)
                faces[s["id"]] = [{"frame": os.path.basename(fp), "t": round(t, 3), "dets": d, "raw_dets": r}
                                  for (fp, t), d, r in zip(shot_frames[s["id"]], dets, raw)]
        json.dump({"imdbid": imdbid, "title": title, "cast": [{k: v for k, v in c.items() if k != "bgr"} for c in cast],
                   "shots": faces}, open(os.path.join(out_dir, "faces.json"), "w"), indent=1)

        print(f"faces: {n_frames} frames, {n_raw_faces} raw faces, {T.t['3c_face_matching']:.1f}s, providers={matcher.app.models['detection'].session.get_providers()}")
        if args.faces_only:
            return

    def frame_img(shot_id, idx, offset):
        fp, t = shot_frames[shot_id][idx]
        img = cv2.imread(fp)
        if not args.no_circles and not args.prep_root:  # prep frames come from circled/ already
            draw_circles(img, faces[shot_id][idx]["dets"], cast)
        return burn_ts(img, format_mmss(t - offset))

    # portrait block, identical for every request of this video
    portrait_parts = [{"type": "text", "text": "[CAST PORTRAITS — actor face crops with their character name and circle colour]"}]
    if not args.no_portraits:
        for c in cast:
            if c.get("portrait_crop"):
                portrait_parts.append({"type": "text", "text": f"{c['name']} [{c['id']}] — played by {c['actor']} — circle colour: {c['colour'].upper()}"})
                portrait_parts.append({"type": "image_url", "image_url": {"url": jpeg_data_url(cv2.imread(c["portrait_crop"]), 90)}})
    cast_text = cast_prompt_block(cast)

    # 4. sequential per-shot generation
    character_registry = [{"id": c["id"], "name": c["name"], "appearance": f"cast member, played by {c['actor']}",
                           "role": "main cast", "circle_colour": c["colour"]} for c in cast]
    props_registry, location_registry, history, results = [], [], [], []
    per_call = []
    for si, shot in enumerate(shots):
        tb = time.perf_counter()
        ctx_start, ctx_end = max(0, si - 2), min(len(shots) - 1, si + 2)
        offset = shots[ctx_start]["start"]
        win_end = shots[ctx_end]["end"]

        sel = {}
        for i in range(ctx_start, ctx_end + 1):
            sid = shots[i]["id"]
            idxs = list(range(len(shot_frames[sid])))
            if i != si:
                idxs = idxs[::max(1, int(p.target_fps / p.context_fps))]
            sel[i] = idxs
        if p.max_frames > 0 and sum(len(v) for v in sel.values()) > p.max_frames:
            tgt_budget = int(p.max_frames * 0.6)
            n_ctx = sum(1 for i in sel if i != si)
            per_ctx = max(2, (p.max_frames - tgt_budget) // max(n_ctx, 1))
            sel = {i: subsample(v, tgt_budget if i == si else per_ctx) for i, v in sel.items()}

        content = list(portrait_parts)
        for i in range(ctx_start, ctx_end + 1):
            cur = shots[i]
            if i == si:
                label = f"[TARGET SHOT | {p.target_fps} FPS]"
            elif i < si:
                summ = events_to_summary(cur.get("adlm_events", []))
                label = f"[CONTEXT SHOT BEFORE (t-{si - i}) | {p.context_fps} FPS]" + (f"\nEvents: {summ}" if summ else "")
            else:
                label = f"[CONTEXT SHOT AFTER (t+{i - si}) | {p.context_fps} FPS]"
            content.append({"type": "text", "text": label})
            for idx in sel[i]:
                img = frame_img(cur["id"], idx, offset)
                if i == si and not args.prep_root:
                    cv2.imwrite(os.path.join(circled_dir, os.path.basename(shot_frames[cur["id"]][idx][0])), img)
                content.append({"type": "image_url", "image_url": {"url": jpeg_data_url(img)}})

        tgt_faces = [(shot_frames[shot["id"]][k][1] - offset, faces[shot["id"]][k]["dets"]) for k in sel[si]]
        prompt = chunk_tmpl.format(
            start_timecode=format_mmss(shot["start"] - offset), end_timecode=format_mmss(shot["end"] - offset),
            face_summary=shot_face_summary(tgt_faces, cast, format_mmss),
            subs_text=get_shifted_subtitles(srt_path, offset, win_end, shot["start"], shot["end"], offset),
            recent_history="\n".join(history[-4:]) or "None (first shot)",
            character_registry=json.dumps(character_registry, indent=2),
            props_registry=json.dumps(props_registry, indent=2) if props_registry else "Empty — no props registered yet.",
            location_registry=json.dumps(location_registry, indent=2) if location_registry else "Empty — no locations registered yet.",
        )
        content.append({"type": "text", "text": prompt})
        system = system_base + PROMPT_VARIANTS[args.prompt_variant] + "\n\n## CAST LIST\n" + cast_text
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if not args.no_save_requests:
            json.dump(messages, open(os.path.join(req_dir, f"shot_{shot['id']:03d}.json"), "w"))
        build_s = time.perf_counter() - tb

        err = None
        for attempt in range(3):
            try:
                text, ptok, otok, dt = call(messages)
                data = json.loads(text)
                break
            except Exception as e:  # noqa: BLE001
                err = str(e)[:300]
                print(f"  shot {shot['id']} attempt {attempt + 1} failed: {err}")
        else:
            data, ptok, otok, dt = {"events": [], "new_characters": [], "new_props": [], "new_locations": []}, 0, 0, 0

        for reg, key in ((character_registry, "new_characters"), (props_registry, "new_props"), (location_registry, "new_locations")):
            for item in data.get(key, []):
                if not any(x["id"] == item["id"] for x in reg):
                    reg.append(item)
        events = []
        for e in data.get("events", []):  # never let the face-circle overlay leak into the text
            d = drop_circle_leaks(e.get("description", ""))
            if d:
                events.append({**e, "description": d})
        shot["adlm_events"] = events
        history.append(f"Shot {shot['id']} [{format_mmss(shot['start'] - offset)}-{format_mmss(shot['end'] - offset)}]: {events_to_summary(events)}")
        n_imgs = sum(1 for c in content if c["type"] == "image_url")
        per_call.append({"shot_id": shot["id"], "images": n_imgs, "prompt_tokens": ptok, "output_tokens": otok,
                         "latency_s": round(dt, 2), "build_s": round(build_s, 2), "events": len(events), "error": err})
        print(f"shot {shot['id']:>3}/{len(shots)}  imgs={n_imgs:>3}  in={ptok:>6}  out={otok:>5}  {dt:6.1f}s  events={len(events)}")
        results.append({
            "shot_id": shot["id"], "start_abs": shot["start"], "end_abs": shot["end"],
            "start_timecode": format_mmss(shot["start"] - offset), "end_timecode": format_mmss(shot["end"] - offset),
            "time_offset": offset, "events": events,
            "recognised_cast": sorted({cast[d["cast_idx"]]["id"] for f in faces[shot["id"]] for d in f["dets"]}),
            "new_characters": data.get("new_characters", []), "new_props": data.get("new_props", []),
            "new_locations": data.get("new_locations", []),
        })
    T.t["4_vlm_generation"] = sum(c["latency_s"] for c in per_call)
    T.t["4b_request_building"] = sum(c["build_s"] for c in per_call)

    # 5. outputs
    out = {"video": video_path, "cmd_filename": args.video, "imdbid": imdbid, "title": title, "model": args.model,
           "num_shots": len(results), "context_fps": p.context_fps, "target_fps": p.target_fps,
           "cast": [{k: v for k, v in c.items() if k not in ("bgr",)} for c in cast],
           "character_registry": character_registry, "props_registry": props_registry,
           "location_registry": location_registry, "shots": results}
    json.dump(out, open(os.path.join(out_dir, "adlm_descriptions.json"), "w"), indent=2, ensure_ascii=False)
    cfg.video.path = video_path
    cfg.model.name = args.model
    json.dump(build_merged_timeline(results, character_registry, cfg),
              open(os.path.join(out_dir, "adlm_timeline.json"), "w"), indent=2, ensure_ascii=False)
    timings = {"stages_s": {k: round(v, 2) for k, v in sorted(T.t.items())}, "n_shots": len(shots),
               "n_frames_extracted": n_frames, "n_raw_faces_detected": n_raw_faces,
               "face_fps": round(n_frames / T.t["3c_face_matching"], 1) if T.t.get("3c_face_matching") else None,
               "per_call": per_call}
    json.dump(timings, open(os.path.join(out_dir, "timings.json"), "w"), indent=2)
    print(json.dumps(timings["stages_s"], indent=1))

    if args.bench_concurrency:
        run_bench(call, req_dir, out_dir, args.bench_concurrency)


def load_prep(args, T):
    """Shots, frames, face matches and cast from a prep_charbank.py output dir (no GPU work)."""
    with T("0_load_prep"):
        d = os.path.join(args.prep_root, args.video)
        meta = json.load(open(os.path.join(d, "meta.json")))
        fj = json.load(open(os.path.join(d, "faces.json")))
        sub = "frames" if args.no_circles else "circled"
        shots = [{"id": s["id"], "start": s["start"], "end": s["end"]} for s in meta["shots"]]
        shot_frames = {s["id"]: [(os.path.join(d, sub, f["file"]), f["t"]) for f in s["frames"]] for s in meta["shots"]}
        faces = {int(k): v for k, v in fj["shots"].items()}
        cast = fj["cast"]
        for c in cast:
            c["portrait_crop"] = os.path.join(args.prep_root, c["portrait_crop"]) if c.get("portrait_crop") else None
    return (shots, shot_frames, faces, cast, meta["imdbid"], None, meta["n_frames"], meta["n_faces"])


def run_bench(call, req_dir, out_dir, levels):
    """Replay the saved per-shot requests concurrently (independent of each other) to measure throughput."""
    reqs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(req_dir, "shot_*.json")))]
    bench = []
    for c in levels:
        batch = reqs[:max(c, 1)] if c <= len(reqs) else (reqs * (c // len(reqs) + 1))[:c]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(call, batch))
        wall = time.perf_counter() - t0
        pin, pout = sum(r[1] for r in res), sum(r[2] for r in res)
        row = {"concurrency": c, "requests": len(batch), "wall_s": round(wall, 1),
               "req_per_min": round(len(batch) / wall * 60, 2), "prompt_tok_per_s": round(pin / wall),
               "output_tok_per_s": round(pout / wall), "mean_latency_s": round(sum(r[3] for r in res) / len(res), 1)}
        bench.append(row)
        print(row)
    json.dump(bench, open(os.path.join(out_dir, "bench.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
