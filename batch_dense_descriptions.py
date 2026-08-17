"""
Batch processing for ADLM dense descriptions using Vertex AI / Gemini batch API.

Processes multiple videos in rounds:
  Round 1: Shot 1 of each video (no prior context)
  Round 2: Shot 2 of each video (with context from round 1)
  ...
  Videos drop out as they complete all shots.

Usage:
    python batch_dense_descriptions.py
    python batch_dense_descriptions.py --config batch_config.yaml
    python batch_dense_descriptions.py --resume output/batch/checkpoint.json
"""

import os
import sys
import json
import time
import base64
import argparse
import subprocess
import glob
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from prompts_adlm import ADLM_SYSTEM_PROMPT, ADLM_CHUNK_PROMPT, ADLM_DESCRIPTION_SCHEMA

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def format_mmss(seconds):
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def mmss_to_seconds(mmss_str):
    parts = mmss_str.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def events_to_summary(events):
    if not events:
        return ""
    return " | ".join(f"[{e['timestamp']}] {e['description']}" for e in events)


def detect_shots(video_path, threshold=0.12, max_shot_duration=10.0):
    result = subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-"
    ], capture_output=True, text=True)

    dur_result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ], capture_output=True, text=True)
    video_duration = float(dur_result.stdout.strip())

    change_points = [0.0]
    for line in result.stderr.split("\n"):
        match = re.search(r'pts_time:([\d.]+)', line)
        if match:
            change_points.append(float(match.group(1)))
    change_points.append(video_duration)

    raw_shots = []
    for i in range(len(change_points) - 1):
        raw_shots.append((round(change_points[i], 3), round(change_points[i + 1], 3)))

    shots = []
    shot_id = 1
    for start, end in raw_shots:
        duration = end - start
        if duration < 0.1:
            continue
        if duration <= max_shot_duration:
            shots.append({"id": shot_id, "start": start, "end": end})
            shot_id += 1
        else:
            cursor = start
            while cursor < end:
                chunk_end = min(cursor + max_shot_duration, end)
                if chunk_end - cursor < 0.1:
                    break
                shots.append({"id": shot_id, "start": round(cursor, 3), "end": round(chunk_end, 3)})
                shot_id += 1
                cursor = chunk_end
    return shots


def extract_frames(video_path, start_sec, end_sec, fps, output_dir, prefix, frame_height=480):
    pattern = os.path.join(output_dir, f"{prefix}_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps},scale=-2:{frame_height}",
        "-q:v", "2", pattern
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = []
    i = 0
    while True:
        path = os.path.join(output_dir, f"{prefix}_{i+1:04d}.jpg")
        if not os.path.exists(path):
            break
        abs_time = start_sec + i / fps
        frames.append((path, abs_time))
        i += 1
    return frames


def burn_timestamp(frame_path, timestamp_str, output_path):
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    font_size = max(20, img.height // 20)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), timestamp_str, font=font)
    text_w = bbox[2] - bbox[0]
    margin = max(5, img.height // 100)
    x = img.width - text_w - margin
    y = margin

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), timestamp_str, font=font, fill="black")
    draw.text((x, y), timestamp_str, font=font, fill="white")
    img.save(output_path)


def subsample(frames, max_count):
    if len(frames) <= max_count:
        return frames
    indices = [int(i * (len(frames) - 1) / (max_count - 1)) for i in range(max_count)]
    return [frames[i] for i in indices]


def get_shifted_subtitles(srt_path, window_start, window_end, target_start, target_end, time_offset):
    try:
        import srt as srt_lib
    except ImportError:
        return "No srt module available."
    if not os.path.exists(srt_path):
        return "No dialogue track available."
    with open(srt_path, 'r', encoding='utf-8') as f:
        subs = list(srt_lib.parse(f.read()))
    result = []
    for sub in subs:
        s_start = sub.start.total_seconds()
        s_end = sub.end.total_seconds()
        if s_start <= window_end and s_end >= window_start:
            shifted_start = s_start - time_offset
            shifted_end = s_end - time_offset
            if s_end <= target_start:
                label = "[BEFORE TARGET]"
            elif s_start >= target_end:
                label = "[AFTER TARGET]"
            else:
                label = "[DURING TARGET]"
            content_clean = sub.content.replace('\n', ' ')
            result.append(
                f"{label} [{format_mmss(shifted_start)} - {format_mmss(shifted_end)}]: {content_clean}"
            )
    return "\n".join(result) if result else "No dialogue in this window."


class VideoState:
    """Tracks per-video processing state across rounds."""

    def __init__(self, video_id, video_path, dialogue_srt, shots, frames_dir, all_shots=None):
        self.video_id = video_id
        self.video_path = video_path
        self.dialogue_srt = dialogue_srt
        self.shots = shots
        self.frames_dir = frames_dir
        self.all_shots = all_shots or shots
        self.current_shot_idx = 0
        self.character_registry = []
        self.description_history = []
        self.results = []

    @property
    def is_done(self):
        return self.current_shot_idx >= len(self.shots)

    @property
    def current_shot(self):
        if self.is_done:
            return None
        return self.shots[self.current_shot_idx]

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "dialogue_srt": self.dialogue_srt,
            "shots": self.shots,
            "all_shots": self.all_shots,
            "frames_dir": self.frames_dir,
            "current_shot_idx": self.current_shot_idx,
            "character_registry": self.character_registry,
            "description_history": self.description_history,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, d):
        vs = cls(d["video_id"], d["video_path"], d["dialogue_srt"],
                 d["shots"], d["frames_dir"], d.get("all_shots", d["shots"]))
        vs.current_shot_idx = d["current_shot_idx"]
        vs.character_registry = d["character_registry"]
        vs.description_history = d["description_history"]
        vs.results = d["results"]
        return vs


def preprocess_video(video_cfg, pipeline_cfg, output_dir):
    """Detect shots and extract+burn all frames for a video."""
    video_id = video_cfg.id
    video_path = video_cfg.path
    dialogue_srt = video_cfg.get("dialogue_srt", "")

    max_shot_duration = pipeline_cfg.get("max_shot_duration", 10.0)
    print(f"\n[{video_id}] Detecting shots in {video_path}...")
    all_shots = detect_shots(video_path, threshold=pipeline_cfg.scene_threshold, max_shot_duration=max_shot_duration)
    num_shots = pipeline_cfg.get("num_shots", 0)
    shots = all_shots[:num_shots] if num_shots > 0 else all_shots
    print(f"[{video_id}] Found {len(all_shots)} shots, processing {len(shots)}")

    frames_dir = os.path.join(output_dir, video_id, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    context_fps = pipeline_cfg.context_fps
    target_fps = pipeline_cfg.target_fps
    frame_height = pipeline_cfg.frame_height
    context_window = pipeline_cfg.context_window
    max_frames = pipeline_cfg.max_frames

    print(f"[{video_id}] Extracting frames for {len(shots)} shots...")
    for idx, shot in enumerate(shots):
        ctx_start = max(0, idx - context_window)
        ctx_end = min(len(all_shots) - 1, idx + context_window)

        window_start_sec = shots[ctx_start]["start"]
        time_offset = window_start_sec

        shot_frame_dir = os.path.join(frames_dir, f"shot_{shot['id']:03d}")
        os.makedirs(shot_frame_dir, exist_ok=True)

        all_frame_paths = {}
        total_raw = 0

        for i in range(ctx_start, ctx_end + 1):
            curr = all_shots[i]
            is_target = (i == idx)
            fps = target_fps if is_target else context_fps
            prefix = f"ctx_{curr['id']:03d}" if not is_target else f"tgt_{curr['id']:03d}"

            raw_frames = extract_frames(
                video_path, curr["start"], curr["end"],
                fps, shot_frame_dir, prefix, frame_height
            )
            all_frame_paths[i] = {
                "frames": raw_frames, "fps": fps,
                "is_target": is_target, "shot": curr
            }
            total_raw += len(raw_frames)

        if max_frames > 0 and total_raw > max_frames:
            target_budget = int(max_frames * 0.6)
            ctx_budget = max_frames - target_budget
            n_ctx = sum(1 for sf in all_frame_paths.values() if not sf["is_target"])
            per_ctx = max(2, ctx_budget // max(n_ctx, 1))
            for sf in all_frame_paths.values():
                if sf["is_target"]:
                    sf["frames"] = subsample(sf["frames"], target_budget)
                else:
                    sf["frames"] = subsample(sf["frames"], per_ctx)

        burnt_frames = []
        for i in range(ctx_start, ctx_end + 1):
            sf = all_frame_paths[i]
            entry = {
                "shot_id": sf["shot"]["id"],
                "is_target": sf["is_target"],
                "fps": sf["fps"],
                "shot_idx_in_video": i,
                "target_idx": idx,
                "paths": [],
            }
            for frame_path, abs_time in sf["frames"]:
                norm_time = abs_time - time_offset
                ts_str = format_mmss(norm_time)
                burnt_path = frame_path.replace(".jpg", "_ts.jpg")
                burn_timestamp(frame_path, ts_str, burnt_path)
                entry["paths"].append(burnt_path)
            burnt_frames.append(entry)

        shot["_burnt_frames"] = burnt_frames
        shot["_ctx_start"] = ctx_start
        shot["_ctx_end"] = ctx_end
        shot["_time_offset"] = time_offset

    print(f"[{video_id}] Frame extraction complete")
    return VideoState(video_id, video_path, dialogue_srt, shots, frames_dir, all_shots)


def build_request_for_shot(vs, pipeline_cfg):
    """Build the content parts and prompt text for the current shot of a video.

    Returns (parts_for_genai, metadata_dict) where parts_for_genai is a list
    suitable for InlinedRequest.contents.
    """
    from google.genai import types

    idx = vs.current_shot_idx
    shot = vs.shots[idx]
    all_shots = vs.all_shots

    ctx_start = shot["_ctx_start"]
    ctx_end = shot["_ctx_end"]
    time_offset = shot["_time_offset"]

    start_sec = shot["start"]
    end_sec = shot["end"]
    norm_target_start = format_mmss(start_sec - time_offset)
    norm_target_end = format_mmss(end_sec - time_offset)

    content_parts = []

    for bf_entry in shot["_burnt_frames"]:
        is_target = bf_entry["is_target"]
        fps = bf_entry["fps"]
        bf_shot_idx = bf_entry["shot_idx_in_video"]

        if is_target:
            label = f"[TARGET SHOT | {fps} FPS]"
        elif bf_shot_idx < idx:
            summary = events_to_summary(vs.shots[bf_shot_idx].get("adlm_events", []))
            desc_text = f"\nEvents: {summary}" if summary else ""
            label = f"[CONTEXT SHOT BEFORE (t-{idx - bf_shot_idx}) | {fps} FPS]{desc_text}"
        else:
            label = f"[CONTEXT SHOT AFTER (t+{bf_shot_idx - idx}) | {fps} FPS]"

        content_parts.append(types.Part.from_text(text=label))

        for img_path in bf_entry["paths"]:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            content_parts.append(types.Part.from_bytes(
                data=img_bytes, mime_type="image/jpeg"
            ))

    subs_text = get_shifted_subtitles(
        vs.dialogue_srt,
        all_shots[ctx_start]["start"], all_shots[ctx_end]["end"],
        start_sec, end_sec, time_offset
    )

    history_lines = vs.description_history[-4:]
    recent_history = "\n".join(history_lines) if history_lines else "None (first shot)"

    if vs.character_registry:
        reg_text = json.dumps(vs.character_registry, indent=2)
    else:
        reg_text = "Empty — no characters registered yet."

    prompt_text = ADLM_CHUNK_PROMPT.format(
        start_timecode=norm_target_start,
        end_timecode=norm_target_end,
        subs_text=subs_text,
        recent_history=recent_history,
        character_registry=reg_text,
    )
    content_parts.append(types.Part.from_text(text=prompt_text))

    metadata = {
        "video_id": vs.video_id,
        "shot_id": str(shot["id"]),
        "shot_idx": str(idx),
    }

    return content_parts, metadata, norm_target_start, norm_target_end


def create_genai_client(batch_cfg):
    from google import genai

    if batch_cfg.backend == "vertex":
        return genai.Client(
            vertexai=True,
            project=batch_cfg.project,
            location=batch_cfg.location,
        )
    else:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)


def submit_batch(client, requests, model_name, batch_cfg, round_num, output_dir):
    """Submit a batch of InlinedRequest objects and return the batch job."""
    from google.genai import types

    display_name = f"adlm-dense-round-{round_num}"

    gcs_bucket = batch_cfg.get("gcs_bucket", "")

    if gcs_bucket:
        jsonl_lines = []
        for req in requests:
            req_dict = type(req).to_dict(req)
            jsonl_lines.append(json.dumps(req_dict))
        jsonl_content = "\n".join(jsonl_lines)

        gcs_input = f"{gcs_bucket}/round_{round_num:03d}_input.jsonl"
        gcs_output = f"{gcs_bucket}/round_{round_num:03d}_output/"

        from google.cloud import storage as gcs_storage
        bucket_name = gcs_input.replace("gs://", "").split("/")[0]
        blob_path = "/".join(gcs_input.replace("gs://", "").split("/")[1:])
        gcs_client = gcs_storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(jsonl_content)
        print(f"  Uploaded {len(requests)} requests to {gcs_input}")

        batch_job = client.batches.create(
            model=model_name,
            src=types.BatchJobSource(gcs_uri=gcs_input),
            config=types.CreateBatchJobConfig(
                display_name=display_name,
                dest=types.BatchJobDestination(gcs_uri=gcs_output),
            ),
        )
    else:
        batch_job = client.batches.create(
            model=model_name,
            src=types.BatchJobSource(inlined_requests=requests),
            config=types.CreateBatchJobConfig(
                display_name=display_name,
            ),
        )

    print(f"  Batch job submitted: {batch_job.name}")
    print(f"  State: {batch_job.state}")
    return batch_job


def wait_for_batch(client, batch_job, poll_interval=30):
    """Poll until batch job completes. Returns the completed job."""
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                       "JOB_STATE_CANCELLED", "SUCCEEDED", "FAILED", "CANCELLED"}
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state = str(batch_job.state)
        if hasattr(batch_job, 'completion_stats') and batch_job.completion_stats:
            stats = batch_job.completion_stats
            total = getattr(stats, 'total_count', '?')
            success = getattr(stats, 'success_count', '?')
            fail = getattr(stats, 'failed_count', '?')
            print(f"  State: {state} | {success}/{total} done, {fail} failed", flush=True)
        else:
            print(f"  State: {state}", flush=True)

        if state in terminal_states or any(t in state for t in terminal_states):
            break
        time.sleep(poll_interval)

    return batch_job


def parse_batch_results(batch_job, client, batch_cfg):
    """Extract results from the completed batch job.

    Returns a list of parsed JSON dicts (or None for failures), in request order.
    """
    results = []

    batch_job = client.batches.get(name=batch_job.name)
    dest = batch_job.dest

    if dest and getattr(dest, 'inlined_responses', None):
        for resp in dest.inlined_responses:
            if getattr(resp, 'error', None):
                print(f"  Request error: {resp.error}")
                results.append(None)
                continue
            body = getattr(resp, 'response', None)
            parsed = _extract_json_from_response(body)
            results.append(parsed)

    elif dest and getattr(dest, 'gcs_uri', None):
        from google.cloud import storage as gcs_storage
        gcs_uri = dest.gcs_uri
        bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
        prefix = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])

        gcs_client = gcs_storage.Client()
        bucket = gcs_client.bucket(bucket_name)
        blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                response = entry.get("response", {})
                text = ""
                candidates = response.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                parsed = _parse_json_text(text) if text else None
                results.append(parsed)

    return results


def _extract_json_from_response(resp_obj):
    if resp_obj is None:
        return None
    text = ""
    if hasattr(resp_obj, 'text'):
        text = resp_obj.text
    elif hasattr(resp_obj, 'candidates'):
        candidates = resp_obj.candidates
        if candidates:
            parts = candidates[0].content.parts if hasattr(candidates[0], 'content') else []
            if parts:
                text = parts[0].text if hasattr(parts[0], 'text') else str(parts[0])
    if not text:
        return None
    return _parse_json_text(text)


def _parse_json_text(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = text.find('{')
        if brace >= 0:
            bracket_count = 0
            for i in range(brace, len(text)):
                if text[i] == '{':
                    bracket_count += 1
                elif text[i] == '}':
                    bracket_count -= 1
                if bracket_count == 0:
                    try:
                        return json.loads(text[brace:i + 1])
                    except json.JSONDecodeError:
                        break
        return None


def apply_round_results(active_states, batch_results):
    """Apply batch results to video states by position, advancing each by one shot.

    active_states and batch_results must be in the same order.
    """
    for i, vs in enumerate(active_states):
        shot = vs.current_shot
        parsed = batch_results[i] if i < len(batch_results) else None

        if parsed is None:
            print(f"  WARNING: Failed result for [{vs.video_id}] shot {shot['id']}")
            vs.results.append({
                "shot_id": shot["id"],
                "start_abs": shot["start"],
                "end_abs": shot["end"],
                "events": [],
                "new_characters": [],
                "error": "parse_failed",
            })
            vs.current_shot_idx += 1
            continue

        events = parsed.get("events", [])
        new_chars = parsed.get("new_characters", [])

        for nc in new_chars:
            if not any(c["id"] == nc["id"] for c in vs.character_registry):
                vs.character_registry.append(nc)

        shot["adlm_events"] = events

        time_offset = shot["_time_offset"]
        norm_start = format_mmss(shot["start"] - time_offset)
        norm_end = format_mmss(shot["end"] - time_offset)
        summary = events_to_summary(events)
        vs.description_history.append(
            f"Shot {shot['id']} [{norm_start}-{norm_end}]: {summary}"
        )

        vs.results.append({
            "shot_id": shot["id"],
            "start_abs": shot["start"],
            "end_abs": shot["end"],
            "start_timecode": norm_start,
            "end_timecode": norm_end,
            "time_offset": time_offset,
            "events": events,
            "new_characters": new_chars,
        })

        print(f"  [{vs.video_id}] Shot {shot['id']}: {len(events)} events, "
              f"{len(new_chars)} new chars")

        vs.current_shot_idx += 1


def save_checkpoint(video_states, round_num, output_dir):
    checkpoint = {
        "round": round_num,
        "videos": [vs.to_dict() for vs in video_states],
    }
    path = os.path.join(output_dir, "checkpoint.json")
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False,
                  default=_checkpoint_serializer)
    return path


def _checkpoint_serializer(obj):
    """Handle non-serializable fields in shot dicts (drop frame paths)."""
    return str(obj)


def load_checkpoint(path):
    with open(path) as f:
        data = json.load(f)
    video_states = [VideoState.from_dict(v) for v in data["videos"]]
    return video_states, data["round"]


def save_final_outputs(video_states, cfg, output_dir):
    for vs in video_states:
        vid_dir = os.path.join(output_dir, vs.video_id, cfg.model.name)
        os.makedirs(vid_dir, exist_ok=True)

        desc_file = os.path.join(vid_dir, "adlm_descriptions.json")
        with open(desc_file, "w", encoding="utf-8") as f:
            json.dump({
                "video": vs.video_path,
                "model": cfg.model.name,
                "scene_threshold": cfg.pipeline.scene_threshold,
                "num_shots": len(vs.shots),
                "context_fps": cfg.pipeline.context_fps,
                "target_fps": cfg.pipeline.target_fps,
                "character_registry": vs.character_registry,
                "shots": vs.results,
            }, f, indent=2, ensure_ascii=False)

        timeline = []
        shot_boundaries = []
        for shot_result in vs.results:
            shot_boundaries.append({
                "shot_id": shot_result["shot_id"],
                "start_abs": format_mmss(shot_result["start_abs"]),
                "end_abs": format_mmss(shot_result["end_abs"]),
                "duration_s": round(shot_result["end_abs"] - shot_result["start_abs"], 1),
            })
            time_offset = shot_result.get("time_offset", 0)
            for event in shot_result.get("events", []):
                abs_seconds = mmss_to_seconds(event["timestamp"]) + time_offset
                timeline.append({
                    "timestamp_abs": format_mmss(abs_seconds),
                    "shot_id": shot_result["shot_id"],
                    "type": "event",
                    "description": event["description"],
                })

        timeline.sort(key=lambda e: mmss_to_seconds(e["timestamp_abs"]))

        timeline_file = os.path.join(vid_dir, "adlm_timeline.json")
        with open(timeline_file, "w", encoding="utf-8") as f:
            json.dump({
                "video": vs.video_path,
                "model": cfg.model.name,
                "total_shots": len(vs.results),
                "total_events": len(timeline),
                "character_registry": vs.character_registry,
                "shot_boundaries": shot_boundaries,
                "timeline": timeline,
            }, f, indent=2, ensure_ascii=False)

        print(f"[{vs.video_id}] Saved {desc_file}")
        print(f"[{vs.video_id}] Saved {timeline_file}")
        print(f"[{vs.video_id}] {len(vs.results)} shots, "
              f"{len(timeline)} events, "
              f"{len(vs.character_registry)} characters")


def main():
    parser = argparse.ArgumentParser(description="Batch ADLM dense descriptions")
    parser.add_argument("--config", default="batch_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint.json to resume from")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli([a for a in sys.argv[1:] if "=" in a])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    output_dir = cfg.output.dir
    os.makedirs(output_dir, exist_ok=True)

    from google.genai import types

    if args.resume:
        print(f"Resuming from {args.resume}")
        video_states, start_round = load_checkpoint(args.resume)
        start_round += 1
        print(f"  {len(video_states)} videos, resuming at round {start_round}")
        for vs in video_states:
            active = "DONE" if vs.is_done else f"shot {vs.current_shot_idx + 1}/{len(vs.shots)}"
            print(f"  [{vs.video_id}] {active}")
    else:
        print(f"Pre-processing {len(cfg.videos)} videos...")
        video_states = []
        for vcfg in cfg.videos:
            vs = preprocess_video(vcfg, cfg.pipeline, output_dir)
            video_states.append(vs)
            print(f"[{vs.video_id}] {len(vs.shots)} shots ready")
        start_round = 0

    client = create_genai_client(cfg.batch)

    max_rounds = max(len(vs.shots) for vs in video_states)
    print(f"\nStarting batch processing: {len(video_states)} videos, "
          f"max {max_rounds} rounds")
    print(f"Model: {cfg.model.name}")
    print(f"Backend: {cfg.batch.backend}\n")

    for round_num in range(start_round, max_rounds):
        active = [vs for vs in video_states if not vs.is_done]
        if not active:
            print("All videos complete.")
            break

        print(f"{'='*60}")
        print(f"ROUND {round_num + 1}/{max_rounds} — {len(active)} active videos")
        print(f"{'='*60}")

        requests = []
        request_order = []

        for vs in active:
            shot = vs.current_shot
            print(f"  [{vs.video_id}] Shot {shot['id']} "
                  f"[{format_mmss(shot['start'])} - {format_mmss(shot['end'])}] "
                  f"({shot['end'] - shot['start']:.1f}s)")

            content_parts, metadata, _, _ = build_request_for_shot(vs, cfg.pipeline)

            req = types.InlinedRequest(
                model=cfg.model.name,
                contents=[types.Content(role="user", parts=content_parts)],
                metadata=metadata,
                config=types.GenerateContentConfig(
                    system_instruction=ADLM_SYSTEM_PROMPT,
                    temperature=cfg.model.temperature,
                    response_mime_type="application/json",
                    response_schema=ADLM_DESCRIPTION_SCHEMA,
                ),
            )
            requests.append(req)
            request_order.append(vs.video_id)

        print(f"\n  Submitting batch of {len(requests)} requests...")
        batch_job = submit_batch(
            client, requests, cfg.model.name,
            cfg.batch, round_num, output_dir,
        )

        poll_interval = cfg.batch.get("poll_interval", 30)
        print(f"  Waiting for batch completion (polling every {poll_interval}s)...")
        batch_job = wait_for_batch(client, batch_job, poll_interval)

        state = str(batch_job.state)
        if "FAILED" in state or "CANCELLED" in state:
            print(f"  BATCH FAILED: {state}")
            save_checkpoint(video_states, round_num, output_dir)
            print(f"  Checkpoint saved. Resume with --resume {output_dir}/checkpoint.json")
            sys.exit(1)

        print(f"  Batch complete. Parsing results...")
        batch_results = parse_batch_results(batch_job, client, cfg.batch)
        print(f"  Got {len(batch_results)} results")

        apply_round_results(active, batch_results)

        for vs in active:
            if vs.is_done:
                print(f"  [{vs.video_id}] COMPLETED — all {len(vs.shots)} shots processed")
            else:
                shot = vs.current_shot
                prev = vs.results[-1] if vs.results else None
                n_events = len(prev["events"]) if prev else 0
                n_chars = len(vs.character_registry)
                print(f"  [{vs.video_id}] Shot {shot['id']} next | "
                      f"last had {n_events} events | {n_chars} chars total")

        cp_path = save_checkpoint(video_states, round_num, output_dir)
        print(f"  Checkpoint: {cp_path}\n")

    print(f"\n{'='*60}")
    print("All rounds complete. Saving final outputs...")
    print(f"{'='*60}\n")

    save_final_outputs(video_states, cfg, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
