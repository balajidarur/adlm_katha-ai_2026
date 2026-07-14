"""
ADLM dense description pipeline — timestamped visual events with character tracking.

Configuration via config.yaml (OmegaConf) with CLI overrides:
    python3 test_adlm_descriptions.py
    python3 test_adlm_descriptions.py pipeline.num_shots=5
    python3 test_adlm_descriptions.py model.provider=local model.name=google/gemma-4-E2B-it
    python3 test_adlm_descriptions.py model.name=gemma-4-26b-a4b-it model.temperature=0.3
"""

import os
import sys
import json
import glob
import subprocess
import re

sys.path.insert(0, "/home/dbalu/workspace/ad_gen/multi_language/srt_for_ads/pipeline")
sys.path.insert(0, "/tmp/adlm_dense_descriptions")

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from prompts_adlm import ADLM_SYSTEM_PROMPT, ADLM_CHUNK_PROMPT, ADLM_DESCRIPTION_SCHEMA
from model_client import create_client

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_config():
    file_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
    cli_cfg = OmegaConf.from_cli()
    return OmegaConf.merge(file_cfg, cli_cfg)


def format_mmss(seconds):
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def mmss_to_seconds(mmss_str):
    parts = mmss_str.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def detect_shots(video_path, threshold=0.12):
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

    shots = []
    for i in range(len(change_points) - 1):
        shots.append({
            "id": i + 1,
            "start": round(change_points[i], 3),
            "end": round(change_points[i + 1], 3)
        })
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


def get_shifted_subtitles(srt_path, window_start, window_end, target_start, target_end, time_offset):
    import srt as srt_lib
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


def events_to_summary(events):
    if not events:
        return ""
    return " | ".join(f"[{e['timestamp']}] {e['description']}" for e in events)


def build_merged_timeline(results, character_registry, cfg):
    timeline = []
    shot_boundaries = []

    for shot in results:
        shot_boundaries.append({
            "shot_id": shot["shot_id"],
            "start_abs": format_mmss(shot["start_abs"]),
            "end_abs": format_mmss(shot["end_abs"]),
            "duration_s": round(shot["end_abs"] - shot["start_abs"], 1)
        })
        time_offset = shot["time_offset"]
        for event in shot.get("events", []):
            abs_seconds = mmss_to_seconds(event["timestamp"]) + time_offset
            timeline.append({
                "timestamp_abs": format_mmss(abs_seconds),
                "shot_id": shot["shot_id"],
                "type": "event",
                "description": event["description"]
            })

    timeline.sort(key=lambda e: mmss_to_seconds(e["timestamp_abs"]))

    return {
        "video": cfg.video.path,
        "model": cfg.model.name,
        "provider": cfg.model.provider,
        "total_shots": len(results),
        "total_events": len(timeline),
        "character_registry": character_registry,
        "shot_boundaries": shot_boundaries,
        "timeline": timeline
    }


def main():
    cfg = load_config()

    video_path = cfg.video.path
    dialogue_srt = cfg.video.dialogue_srt
    output_dir = cfg.output.dir
    context_fps = cfg.pipeline.context_fps
    target_fps = cfg.pipeline.target_fps
    threshold = cfg.pipeline.scene_threshold
    num_shots = cfg.pipeline.num_shots
    frame_height = cfg.pipeline.frame_height

    os.makedirs(output_dir, exist_ok=True)

    print(f"Provider: {cfg.model.provider} | Model: {cfg.model.name}")
    client = create_client(cfg)

    print(f"Detecting shots in {video_path} (threshold={threshold})...")
    all_shots = detect_shots(video_path, threshold=threshold)
    print(f"Found {len(all_shots)} shots\n")

    for s in all_shots:
        dur = s["end"] - s["start"]
        print(f"  Shot {s['id']:3d}: [{format_mmss(s['start'])} - {format_mmss(s['end'])}] ({dur:.1f}s)")
    print()

    shots = all_shots[:num_shots] if num_shots > 0 else all_shots
    print(f"Processing {len(shots)} shots")
    print(f"Context: {context_fps} FPS | Target: {target_fps} FPS\n")

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    character_registry = []
    results = []
    description_history = []

    for idx, shot in enumerate(shots):
        shot_id = shot["id"]
        start_sec = shot["start"]
        end_sec = shot["end"]

        ctx_start = max(0, idx - 2)
        ctx_end = min(len(shots) - 1, idx + 2)

        window_start_sec = shots[ctx_start]["start"]
        window_end_sec = shots[ctx_end]["end"]
        time_offset = window_start_sec

        norm_target_start = format_mmss(start_sec - time_offset)
        norm_target_end = format_mmss(end_sec - time_offset)

        print(f"--- Shot {shot_id} [{format_mmss(start_sec)} - {format_mmss(end_sec)}] "
              f"| normalized [{norm_target_start} - {norm_target_end}] "
              f"| {end_sec - start_sec:.1f}s ---")

        for old in glob.glob(os.path.join(frames_dir, "*.jpg")):
            os.remove(old)

        parts = []
        frame_counts = {}
        total_frames = 0
        for i in range(ctx_start, ctx_end + 1):
            curr = shots[i]
            c_start = curr["start"]
            c_end = curr["end"]
            is_target = (i == idx)
            fps = target_fps if is_target else context_fps

            if is_target:
                label = f"[TARGET SHOT | {target_fps} FPS]"
            elif i < idx:
                summary = events_to_summary(curr.get("adlm_events", []))
                desc_text = f"\nEvents: {summary}" if summary else ""
                label = f"[CONTEXT SHOT BEFORE (t-{idx-i}) | {context_fps} FPS]{desc_text}"
            else:
                label = f"[CONTEXT SHOT AFTER (t+{i-idx}) | {context_fps} FPS]"

            parts.append({"type": "text", "text": label})

            prefix = f"shot_{curr['id']:03d}"
            raw_frames = extract_frames(video_path, c_start, c_end, fps, frames_dir, prefix, frame_height)

            role = "TARGET" if is_target else f"t-{idx-i}" if i < idx else f"t+{i-idx}"
            frame_counts[curr["id"]] = (len(raw_frames), fps, role)
            total_frames += len(raw_frames)

            for frame_path, abs_time in raw_frames:
                norm_time = abs_time - time_offset
                ts_str = format_mmss(norm_time)
                burnt_path = frame_path.replace(".jpg", "_ts.jpg")
                burn_timestamp(frame_path, ts_str, burnt_path)
                parts.append({"type": "image", "path": burnt_path})

        print(f"  Frames: {total_frames} total")
        for sid, (nf, fp, role) in frame_counts.items():
            print(f"    Shot {sid:3d} ({role:>6s}, {fp} FPS): {nf} frames")

        subs_text = get_shifted_subtitles(
            dialogue_srt, window_start_sec, window_end_sec,
            start_sec, end_sec, time_offset
        )

        history_lines = description_history[-4:]
        recent_history = "\n".join(history_lines) if history_lines else "None (first shot)"

        if character_registry:
            reg_text = json.dumps(character_registry, indent=2)
        else:
            reg_text = "Empty — no characters registered yet."

        prompt_text = ADLM_CHUNK_PROMPT.format(
            start_timecode=norm_target_start,
            end_timecode=norm_target_end,
            subs_text=subs_text,
            recent_history=recent_history,
            character_registry=reg_text
        )
        parts.append({"type": "text", "text": prompt_text})

        try:
            resp = client.generate(
                system_prompt=ADLM_SYSTEM_PROMPT,
                parts=parts,
                schema=ADLM_DESCRIPTION_SCHEMA
            )
            resp_data = resp["data"]
            events = resp_data.get("events", [])
            new_chars = resp_data.get("new_characters", [])

            for nc in new_chars:
                if not any(c["id"] == nc["id"] for c in character_registry):
                    character_registry.append(nc)
                    print(f"  NEW CHARACTER: {nc['name']} ({nc['id']}) — {nc.get('appearance', '')[:60]}")

            shot["adlm_events"] = events

            summary = events_to_summary(events)
            description_history.append(f"Shot {shot_id} [{norm_target_start}-{norm_target_end}]: {summary}")

            print(f"  Events: {len(events)}")
            for e in events:
                print(f"    [{e['timestamp']}] {e['description'][:100]}")
            print(f"  Tokens: prompt={resp['prompt_tokens']}, "
                  f"output={resp['output_tokens']}, "
                  f"thinking={resp['thinking_tokens']}")
            print()

            results.append({
                "shot_id": shot_id,
                "start_abs": start_sec,
                "end_abs": end_sec,
                "start_timecode": norm_target_start,
                "end_timecode": norm_target_end,
                "time_offset": time_offset,
                "events": events,
                "new_characters": new_chars
            })

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "shot_id": shot_id,
                "start_abs": start_sec,
                "end_abs": end_sec,
                "start_timecode": norm_target_start,
                "end_timecode": norm_target_end,
                "time_offset": time_offset,
                "events": [],
                "new_characters": []
            })

    output_file = os.path.join(output_dir, "adlm_descriptions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "video": video_path,
            "model": cfg.model.name,
            "provider": cfg.model.provider,
            "scene_threshold": threshold,
            "num_shots": len(shots),
            "context_fps": context_fps,
            "target_fps": target_fps,
            "character_registry": character_registry,
            "shots": results
        }, f, indent=2, ensure_ascii=False)

    merged = build_merged_timeline(results, character_registry, cfg)
    timeline_file = os.path.join(output_dir, "adlm_timeline.json")
    with open(timeline_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Per-shot results: {output_file}")
    print(f"Merged timeline:  {timeline_file}")
    print(f"Total shots: {len(results)}")
    print(f"Total events: {merged['total_events']}")
    print(f"Characters found: {len(character_registry)}")
    for c in character_registry:
        print(f"  - {c['name']} ({c['id']}): {c.get('appearance', 'N/A')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
