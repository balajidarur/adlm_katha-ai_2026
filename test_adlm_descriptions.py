"""
Test script: Generate ADLM dense descriptions for video shots using
timestamped frame extraction with normalized timelines.

Shot detection uses ffmpeg scene change detection (not pre-chunked files).

- Context shots (t-2, t-1, t+1, t+2) at 1 FPS
- Target shot at 3 FPS
- All timestamps normalized so t-2 shot start = 00:00.0
- Dialogues time-shifted to match
- Timestamps burnt on frames as MM:SS.m (white text, black border, top-right)

Produces:
  adlm_descriptions.json — per-shot events with character tracking
  adlm_timeline.json     — merged chronological timeline (absolute times restored)

Usage:
    cd /home/dbalu/workspace/ad_gen/multi_language/srt_for_ads/pipeline
    python3 /tmp/adlm_dense_descriptions/test_adlm_descriptions.py
    python3 /tmp/adlm_dense_descriptions/test_adlm_descriptions.py --num-shots 5
    python3 /tmp/adlm_dense_descriptions/test_adlm_descriptions.py --threshold 0.2
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import datetime
import re

sys.path.insert(0, "/home/dbalu/workspace/ad_gen/multi_language/srt_for_ads/pipeline")
sys.path.insert(0, "/tmp/adlm_dense_descriptions")

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from prompts_adlm import ADLM_SYSTEM_PROMPT, ADLM_CHUNK_PROMPT, ADLM_DESCRIPTION_SCHEMA

# --- Config ---
VIDEO_PATH = "output/rotary_3min/rotary_3min.mp4"
DIALOGUE_SRT = "output/rotary_3min/rotary_3min.srt"
OUTPUT_DIR = "/tmp/adlm_dense_descriptions/output"
MODEL = "gemini-3.1-pro-preview"

CONTEXT_FPS = 1
TARGET_FPS = 3
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEFAULT_SCENE_THRESHOLD = 0.12

os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_mmss(seconds):
    """Format seconds as MM:SS.m (1 decimal place)."""
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def mmss_to_seconds(mmss_str):
    """Parse MM:SS.m back to seconds."""
    parts = mmss_str.split(":")
    minutes = int(parts[0])
    secs = float(parts[1])
    return minutes * 60 + secs


def detect_shots(video_path, threshold=DEFAULT_SCENE_THRESHOLD):
    """Run ffmpeg scene detection and return list of shot boundaries as
    [{"id": 1, "start": 0.0, "end": 4.629}, ...]"""
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


def extract_frames(video_path, start_sec, end_sec, fps, output_dir, prefix):
    """Extract frames from video at given FPS, scaled to 480p height. Returns list of (path, absolute_time)."""
    pattern = os.path.join(output_dir, f"{prefix}_%04d.jpg")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps},scale=-2:480",
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
    """Burn MM:SS.m timestamp on top-right of frame (white text, black border)."""
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
    """Get subtitles in window with timestamps shifted by -time_offset, formatted as MM:SS.m."""
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
    """Compact summary of events for history context."""
    if not events:
        return ""
    parts = []
    for e in events:
        parts.append(f"[{e['timestamp']}] {e['description']}")
    return " | ".join(parts)


def build_merged_timeline(results, character_registry):
    """Merge all per-shot events into a single chronological timeline with absolute timestamps."""
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
        "video": VIDEO_PATH,
        "model": MODEL,
        "total_shots": len(results),
        "total_events": len(timeline),
        "character_registry": character_registry,
        "shot_boundaries": shot_boundaries,
        "timeline": timeline
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=0,
                        help="Number of shots to process (0 = all)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                        help="Scene change detection threshold (0.0-1.0)")
    args = parser.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Detect real shot boundaries
    print(f"Detecting shots in {VIDEO_PATH} (threshold={args.threshold})...")
    all_shots = detect_shots(VIDEO_PATH, threshold=args.threshold)
    print(f"Found {len(all_shots)} shots\n")

    for s in all_shots:
        dur = s["end"] - s["start"]
        print(f"  Shot {s['id']:3d}: [{format_mmss(s['start'])} - {format_mmss(s['end'])}] ({dur:.1f}s)")
    print()

    shots = all_shots[:args.num_shots] if args.num_shots > 0 else all_shots
    print(f"Processing {len(shots)} shots")
    print(f"Model: {MODEL}")
    print(f"Context shots: {CONTEXT_FPS} FPS | Target shot: {TARGET_FPS} FPS\n")

    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    character_registry = []
    results = []
    description_history = []

    for idx, shot in enumerate(shots):
        shot_id = shot["id"]
        start_sec = shot["start"]
        end_sec = shot["end"]

        # Context window: 2 shots before, 2 after
        ctx_start = max(0, idx - 2)
        ctx_end = min(len(shots) - 1, idx + 2)

        # Time offset: start of first context shot becomes 0
        window_start_sec = shots[ctx_start]["start"]
        window_end_sec = shots[ctx_end]["end"]
        time_offset = window_start_sec

        norm_target_start = format_mmss(start_sec - time_offset)
        norm_target_end = format_mmss(end_sec - time_offset)

        print(f"--- Shot {shot_id} [{format_mmss(start_sec)} - {format_mmss(end_sec)}] "
              f"| normalized [{norm_target_start} - {norm_target_end}] "
              f"| {end_sec - start_sec:.1f}s ---")

        # Clean up previous frames
        for old in glob.glob(os.path.join(frames_dir, "*.jpg")):
            os.remove(old)

        # Extract and burn frames for each shot in the window
        parts = []
        frame_counts = {}
        total_frames = 0
        for i in range(ctx_start, ctx_end + 1):
            curr = shots[i]
            c_start = curr["start"]
            c_end = curr["end"]
            is_target = (i == idx)
            fps = TARGET_FPS if is_target else CONTEXT_FPS

            if is_target:
                label = f"[TARGET SHOT | {TARGET_FPS} FPS]"
            elif i < idx:
                summary = events_to_summary(curr.get("adlm_events", []))
                desc_text = f"\nEvents: {summary}" if summary else ""
                label = f"[CONTEXT SHOT BEFORE (t-{idx-i}) | {CONTEXT_FPS} FPS]{desc_text}"
            else:
                label = f"[CONTEXT SHOT AFTER (t+{i-idx}) | {CONTEXT_FPS} FPS]"

            parts.append(types.Part(text=label))

            prefix = f"shot_{curr['id']:03d}"
            raw_frames = extract_frames(VIDEO_PATH, c_start, c_end, fps, frames_dir, prefix)

            role = "TARGET" if is_target else f"t-{idx-i}" if i < idx else f"t+{i-idx}"
            frame_counts[curr["id"]] = (len(raw_frames), fps, role)
            total_frames += len(raw_frames)

            for frame_path, abs_time in raw_frames:
                norm_time = abs_time - time_offset
                ts_str = format_mmss(norm_time)
                burnt_path = frame_path.replace(".jpg", "_ts.jpg")
                burn_timestamp(frame_path, ts_str, burnt_path)

                with open(burnt_path, "rb") as f:
                    parts.append(types.Part(
                        inline_data=types.Blob(data=f.read(), mime_type="image/jpeg")
                    ))

        # Log frame counts
        print(f"  Frames: {total_frames} total")
        for sid, (nf, fp, role) in frame_counts.items():
            print(f"    Shot {sid:3d} ({role:>6s}, {fp} FPS): {nf} frames")

        # Dialogue with shifted timestamps
        subs_text = get_shifted_subtitles(
            DIALOGUE_SRT, window_start_sec, window_end_sec,
            start_sec, end_sec, time_offset
        )

        # Recent history (last 4 shots)
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
        parts.append(types.Part(text=prompt_text))

        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=ADLM_DESCRIPTION_SCHEMA,
                    system_instruction=ADLM_SYSTEM_PROMPT
                )
            )
            resp_data = json.loads(resp.text)
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
            print(f"  Tokens: prompt={resp.usage_metadata.prompt_token_count}, "
                  f"output={resp.usage_metadata.candidates_token_count}, "
                  f"thinking={resp.usage_metadata.thoughts_token_count}")
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

    # Save per-shot results
    output_file = os.path.join(OUTPUT_DIR, "adlm_descriptions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "video": VIDEO_PATH,
            "model": MODEL,
            "scene_threshold": args.threshold,
            "num_shots": len(shots),
            "context_fps": CONTEXT_FPS,
            "target_fps": TARGET_FPS,
            "character_registry": character_registry,
            "shots": results
        }, f, indent=2, ensure_ascii=False)

    # Build and save merged timeline
    merged = build_merged_timeline(results, character_registry)
    timeline_file = os.path.join(OUTPUT_DIR, "adlm_timeline.json")
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
