"""
Audio-Visual Diarization Pipeline — Pure speaker identification.

Dialogue-based chunking (5 dialogues per chunk), dialogue-aware frame sampling
(3 FPS during dialogue, 1 FPS between, max 60 frames), audio + video input.

Usage:
    python3 /tmp/adlm_dense_descriptions/test_av_diarize.py
    python3 /tmp/adlm_dense_descriptions/test_av_diarize.py --num-chunks 3
    python3 /tmp/adlm_dense_descriptions/test_av_diarize.py --chunk-size 5
"""

import os
import sys
import json
import glob
import argparse
import subprocess
import re

sys.path.insert(0, "/home/dbalu/workspace/ad_gen/multi_language/srt_for_ads/pipeline")
sys.path.insert(0, "/tmp/adlm_dense_descriptions")

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types
from prompts_av_diarize import AV_DIARIZE_SYSTEM_PROMPT, AV_DIARIZE_CHUNK_PROMPT, AV_DIARIZE_SCHEMA

VIDEO_PATH = "output/rotary_3min/rotary_3min.mp4"
DIALOGUE_SRT = "output/rotary_3min/rotary_3min.srt"
OUTPUT_DIR = "/tmp/adlm_dense_descriptions/output_av_diarize"
MODEL = "gemini-3.1-pro-preview"

DIALOGUE_FPS = 3
GAP_FPS = 1
MAX_FRAMES_PER_CHUNK = 60
CHUNK_SIZE = 5
CHUNK_PADDING = 2.0
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_mmss(seconds):
    minutes = int(seconds) // 60
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def mmss_to_seconds(mmss_str):
    parts = mmss_str.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def get_video_duration(video_path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())


def load_subtitles(srt_path):
    import srt as srt_lib
    if not os.path.exists(srt_path):
        return []
    with open(srt_path, 'r', encoding='utf-8') as f:
        subs = list(srt_lib.parse(f.read()))
    return [(sub.start.total_seconds(), sub.end.total_seconds(), sub.content.replace('\n', ' ')) for sub in subs]


def build_dialogue_chunks(subs, chunk_size=5, padding=2.0, video_duration=None):
    """Group dialogues into chunks of chunk_size. Returns list of chunks, each:
    {"dialogues": [(start, end, text), ...], "start": float, "end": float}"""
    if not subs:
        return []
    chunks = []
    for i in range(0, len(subs), chunk_size):
        group = subs[i:i + chunk_size]
        chunk_start = max(0.0, group[0][0] - padding)
        chunk_end = group[-1][1] + padding
        if video_duration:
            chunk_end = min(chunk_end, video_duration)
        chunks.append({
            "dialogues": group,
            "start": round(chunk_start, 3),
            "end": round(chunk_end, 3)
        })
    return chunks


def extract_frames(video_path, start_sec, end_sec, fps, output_dir, prefix):
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


def extract_dialogue_aware_frames(video_path, chunk, output_dir, prefix, max_frames=60):
    """Extract frames: 3 FPS during dialogue, 1 FPS during gaps. Returns [(path, abs_time)]."""
    dialogues = chunk["dialogues"]
    chunk_start = chunk["start"]
    chunk_end = chunk["end"]

    # Build time segments: (start, end, fps)
    segments = []
    t = chunk_start
    for d_start, d_end, _ in dialogues:
        if t < d_start:
            segments.append((t, d_start, GAP_FPS))
        segments.append((d_start, d_end, DIALOGUE_FPS))
        t = d_end
    if t < chunk_end:
        segments.append((t, chunk_end, GAP_FPS))

    all_frames = []
    for seg_idx, (s_start, s_end, fps) in enumerate(segments):
        if s_end <= s_start:
            continue
        seg_prefix = f"{prefix}_seg{seg_idx:02d}"
        raw = extract_frames(video_path, s_start, s_end, fps, output_dir, seg_prefix)
        all_frames.extend(raw)

    # Subsample if over max_frames, keeping at least 1 frame per dialogue
    if len(all_frames) > max_frames:
        # Mark frames that fall within a dialogue window
        dialogue_frames = set()
        for path, abs_time in all_frames:
            for d_start, d_end, _ in dialogues:
                if d_start <= abs_time <= d_end:
                    dialogue_frames.add(path)
                    break

        # Keep all dialogue frames, subsample the rest
        kept = [(p, t) for p, t in all_frames if p in dialogue_frames]
        other = [(p, t) for p, t in all_frames if p not in dialogue_frames]

        remaining_slots = max_frames - len(kept)
        if remaining_slots > 0 and other:
            step = max(1, len(other) // remaining_slots)
            kept.extend(other[::step][:remaining_slots])

        kept.sort(key=lambda x: x[1])
        all_frames = kept

    return all_frames


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


def extract_window_audio(video_path, start_sec, end_sec, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-i", video_path,
        "-vn", "-q:a", "0", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return None


def get_dialogue_texts(dialogues, time_offset):
    if not dialogues:
        return "None"
    lines = []
    for s_start, s_end, text in dialogues:
        lines.append(f"[{format_mmss(s_start - time_offset)} - {format_mmss(s_end - time_offset)}]: {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-chunks", type=int, default=0,
                        help="Number of chunks to process (0 = all)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Dialogues per chunk")
    args = parser.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    video_duration = get_video_duration(VIDEO_PATH)

    all_subs = load_subtitles(DIALOGUE_SRT)
    print(f"Loaded {len(all_subs)} dialogue lines from {DIALOGUE_SRT}")

    all_chunks = build_dialogue_chunks(all_subs, chunk_size=args.chunk_size,
                                        padding=CHUNK_PADDING, video_duration=video_duration)
    print(f"Built {len(all_chunks)} chunks of {args.chunk_size} dialogues each\n")

    for i, ch in enumerate(all_chunks):
        print(f"  Chunk {i+1}: [{format_mmss(ch['start'])} - {format_mmss(ch['end'])}] "
              f"({ch['end'] - ch['start']:.1f}s, {len(ch['dialogues'])} dialogues)")
    print()

    chunks = all_chunks[:args.num_chunks] if args.num_chunks > 0 else all_chunks
    print(f"Processing {len(chunks)} chunks")
    print(f"Model: {MODEL}")
    print(f"Frame sampling: {DIALOGUE_FPS} FPS dialogue, {GAP_FPS} FPS gaps, max {MAX_FRAMES_PER_CHUNK}\n")

    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    character_registry = []
    results = []

    for idx, chunk in enumerate(chunks):
        chunk_id = idx + 1
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]

        # Context: 1 chunk before, 1 chunk after
        ctx_before = chunks[idx - 1] if idx > 0 else None
        ctx_after = chunks[idx + 1] if idx < len(chunks) - 1 else None

        window_start = ctx_before["start"] if ctx_before else chunk_start
        window_end = ctx_after["end"] if ctx_after else chunk_end
        time_offset = window_start

        norm_start = format_mmss(chunk_start - time_offset)
        norm_end = format_mmss(chunk_end - time_offset)

        print(f"--- Chunk {chunk_id} [{format_mmss(chunk_start)} - {format_mmss(chunk_end)}] "
              f"| normalized [{norm_start} - {norm_end}] "
              f"| {chunk_end - chunk_start:.1f}s | {len(chunk['dialogues'])} dialogues ---")

        # Clean up previous frames
        for old in glob.glob(os.path.join(frames_dir, "*.jpg")):
            os.remove(old)

        # Extract window audio
        audio_path = os.path.join(OUTPUT_DIR, f"window_audio_chunk{chunk_id}.mp3")
        audio_file = extract_window_audio(VIDEO_PATH, window_start, window_end, audio_path)

        parts = []

        if audio_file:
            layout_lines = ["[AUDIO — continuous for full context window]"]
            if ctx_before:
                layout_lines.append(f"  Context chunk (before): {format_mmss(ctx_before['start'] - time_offset)} - {format_mmss(ctx_before['end'] - time_offset)}")
            layout_lines.append(f"  Target chunk: {norm_start} - {norm_end}")
            if ctx_after:
                layout_lines.append(f"  Context chunk (after): {format_mmss(ctx_after['start'] - time_offset)} - {format_mmss(ctx_after['end'] - time_offset)}")
            parts.append(types.Part(text="\n".join(layout_lines)))

            with open(audio_file, "rb") as af:
                parts.append(types.Part(
                    inline_data=types.Blob(data=af.read(), mime_type="audio/mp3")
                ))
            print(f"  Audio: {os.path.getsize(audio_file) / 1024:.0f} KB")
        else:
            print("  Audio: extraction failed, proceeding without audio")

        # Context frames (1 FPS)
        total_frames = 0
        if ctx_before:
            parts.append(types.Part(text="[CONTEXT CHUNK BEFORE | 1 FPS]"))
            ctx_frames = extract_frames(VIDEO_PATH, ctx_before["start"], ctx_before["end"],
                                        GAP_FPS, frames_dir, f"ctx_before_{chunk_id}")
            for fp, abs_time in ctx_frames:
                ts_str = format_mmss(abs_time - time_offset)
                burnt = fp.replace(".jpg", "_ts.jpg")
                burn_timestamp(fp, ts_str, burnt)
                with open(burnt, "rb") as f:
                    parts.append(types.Part(inline_data=types.Blob(data=f.read(), mime_type="image/jpeg")))
            total_frames += len(ctx_frames)
            print(f"  Context before: {len(ctx_frames)} frames (1 FPS)")

        # Target frames (dialogue-aware)
        parts.append(types.Part(text=f"[TARGET CHUNK | {DIALOGUE_FPS} FPS dialogue, {GAP_FPS} FPS gaps]"))
        target_frames = extract_dialogue_aware_frames(
            VIDEO_PATH, chunk, frames_dir, f"target_{chunk_id}", MAX_FRAMES_PER_CHUNK)
        for fp, abs_time in target_frames:
            ts_str = format_mmss(abs_time - time_offset)
            burnt = fp.replace(".jpg", "_ts.jpg")
            burn_timestamp(fp, ts_str, burnt)
            with open(burnt, "rb") as f:
                parts.append(types.Part(inline_data=types.Blob(data=f.read(), mime_type="image/jpeg")))
        total_frames += len(target_frames)
        print(f"  Target: {len(target_frames)} frames (dialogue-aware)")

        if ctx_after:
            parts.append(types.Part(text="[CONTEXT CHUNK AFTER | 1 FPS]"))
            ctx_frames = extract_frames(VIDEO_PATH, ctx_after["start"], ctx_after["end"],
                                        GAP_FPS, frames_dir, f"ctx_after_{chunk_id}")
            for fp, abs_time in ctx_frames:
                ts_str = format_mmss(abs_time - time_offset)
                burnt = fp.replace(".jpg", "_ts.jpg")
                burn_timestamp(fp, ts_str, burnt)
                with open(burnt, "rb") as f:
                    parts.append(types.Part(inline_data=types.Blob(data=f.read(), mime_type="image/jpeg")))
            total_frames += len(ctx_frames)
            print(f"  Context after: {len(ctx_frames)} frames (1 FPS)")

        print(f"  Total frames: {total_frames}")

        # Dialogue texts
        target_subs_text = get_dialogue_texts(chunk["dialogues"], time_offset)
        context_dialogues = []
        if ctx_before:
            context_dialogues.extend(ctx_before["dialogues"])
        if ctx_after:
            context_dialogues.extend(ctx_after["dialogues"])
        context_subs_text = get_dialogue_texts(context_dialogues, time_offset)

        if character_registry:
            reg_text = json.dumps(character_registry, indent=2)
        else:
            reg_text = "Empty — no characters registered yet."

        prompt_text = AV_DIARIZE_CHUNK_PROMPT.format(
            start_timecode=norm_start,
            end_timecode=norm_end,
            target_subs_text=target_subs_text,
            context_subs_text=context_subs_text,
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
                    response_schema=AV_DIARIZE_SCHEMA,
                    system_instruction=AV_DIARIZE_SYSTEM_PROMPT
                )
            )
            resp_data = json.loads(resp.text)
            dialogues = resp_data.get("dialogues", [])
            new_chars = resp_data.get("new_characters", [])

            for nc in new_chars:
                if not any(c["id"] == nc["id"] for c in character_registry):
                    character_registry.append(nc)
                    print(f"  NEW CHARACTER: {nc['name']} ({nc['id']})")

            print(f"  Diarized: {len(dialogues)} dialogues")
            for d in dialogues:
                conf = d.get('confidence', '?')
                print(f"    [{d['timestamp_start']}-{d['timestamp_end']}] "
                      f"{d['speaker_name']} [{d['speaker_id']}] ({conf}): {d['text'][:80]}")
            print(f"  Tokens: prompt={resp.usage_metadata.prompt_token_count}, "
                  f"output={resp.usage_metadata.candidates_token_count}, "
                  f"thinking={resp.usage_metadata.thoughts_token_count}")
            print()

            results.append({
                "chunk_id": chunk_id,
                "start_abs": chunk_start,
                "end_abs": chunk_end,
                "start_timecode": norm_start,
                "end_timecode": norm_end,
                "time_offset": time_offset,
                "dialogues": dialogues,
                "new_characters": new_chars
            })

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "chunk_id": chunk_id,
                "start_abs": chunk_start,
                "end_abs": chunk_end,
                "start_timecode": norm_start,
                "end_timecode": norm_end,
                "time_offset": time_offset,
                "dialogues": [],
                "new_characters": []
            })

    # Save results
    output_file = os.path.join(OUTPUT_DIR, "av_diarize_descriptions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "video": VIDEO_PATH,
            "model": MODEL,
            "chunk_size": args.chunk_size,
            "num_chunks": len(results),
            "character_registry": character_registry,
            "chunks": results
        }, f, indent=2, ensure_ascii=False)

    # Build merged timeline
    all_dialogues = []
    for chunk_result in results:
        time_offset = chunk_result["time_offset"]
        for dlg in chunk_result.get("dialogues", []):
            abs_start = mmss_to_seconds(dlg["timestamp_start"]) + time_offset
            abs_end = mmss_to_seconds(dlg["timestamp_end"]) + time_offset
            all_dialogues.append({
                "timestamp_start_abs": format_mmss(abs_start),
                "timestamp_end_abs": format_mmss(abs_end),
                "chunk_id": chunk_result["chunk_id"],
                "text": dlg["text"],
                "speaker_id": dlg["speaker_id"],
                "speaker_name": dlg["speaker_name"],
                "confidence": dlg["confidence"]
            })
    all_dialogues.sort(key=lambda d: mmss_to_seconds(d["timestamp_start_abs"]))

    timeline_file = os.path.join(OUTPUT_DIR, "av_diarize_timeline.json")
    with open(timeline_file, "w", encoding="utf-8") as f:
        json.dump({
            "video": VIDEO_PATH,
            "model": MODEL,
            "total_dialogues": len(all_dialogues),
            "character_registry": character_registry,
            "dialogues": all_dialogues
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Per-chunk results: {output_file}")
    print(f"Merged timeline:  {timeline_file}")
    print(f"Total chunks: {len(results)}")
    print(f"Total dialogues diarized: {len(all_dialogues)}")
    print(f"Characters found: {len(character_registry)}")
    for c in character_registry:
        print(f"  - {c['name']} ({c['id']}): {c.get('appearance', 'N/A')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
