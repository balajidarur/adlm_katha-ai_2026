"""
Audio-Visual Diarization Prompts — Pure speaker identification pipeline.

Dialogue-based chunking (5 dialogues per chunk), dialogue-aware frame sampling
(3 FPS during dialogue, 1 FPS between, max 60 frames), audio + video input.
No dense descriptions — diarization only.
"""

AV_DIARIZE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "dialogues": {
            "type": "ARRAY",
            "description": "Speaker-assigned dialogues. One entry per dialogue line in the target chunk.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "timestamp_start": {
                        "type": "STRING",
                        "description": "Dialogue start time in MM:SS.m format (normalized)."
                    },
                    "timestamp_end": {
                        "type": "STRING",
                        "description": "Dialogue end time in MM:SS.m format (normalized)."
                    },
                    "text": {
                        "type": "STRING",
                        "description": "The dialogue text exactly as provided."
                    },
                    "speaker_id": {
                        "type": "STRING",
                        "description": "Character registry ID (e.g., 'char_driver'). Use 'unknown' if indeterminate."
                    },
                    "speaker_name": {
                        "type": "STRING",
                        "description": "Character name. Use 'unknown' if uncertain."
                    },
                    "confidence": {
                        "type": "STRING",
                        "description": "'high' if identified via lip movement, visual cue, or voice match. 'low' if inferred from context."
                    }
                },
                "required": ["timestamp_start", "timestamp_end", "text", "speaker_id", "speaker_name", "confidence"]
            }
        },
        "new_characters": {
            "type": "ARRAY",
            "description": "Characters appearing for the FIRST TIME. Empty array if none.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING", "description": "Unique ID like 'char_driver'."},
                    "name": {"type": "STRING", "description": "Character name or short descriptor."},
                    "appearance": {"type": "STRING", "description": "Physical appearance: clothing, hair, build."},
                    "role": {"type": "STRING", "description": "Apparent role or action."}
                },
                "required": ["id", "name", "appearance"]
            }
        }
    },
    "required": ["dialogues", "new_characters"]
}


AV_DIARIZE_SYSTEM_PROMPT = """You are a speaker identification expert. Given video frames and audio, your sole task is to identify WHO speaks each dialogue line.

## Input format
You receive:
1. **Audio**: One continuous audio file covering the full context window. A shot layout label shows which time ranges correspond to which chunks.
2. **Frames**: Extracted from the video:
   - CONTEXT CHUNKS (before/after target): 1 frame per second
   - TARGET CHUNK: 3 FPS during dialogue timestamps, 1 FPS during gaps
   - Each frame has a burnt-in timestamp (MM:SS.m) in the top-right corner
3. **Dialogue lines**: With timestamps aligned to the same normalized timeline
   - DIALOGUES TO DIARIZE: lines in the target chunk you must assign speakers to
   - CONTEXT DIALOGUES: reference only

All timestamps are normalized so the first context chunk starts at 00:00.0.

## Your task
For each dialogue line under DIALOGUES TO DIARIZE, identify the speaker:
1. Examine the video frames around that dialogue's timestamps
2. Listen to the audio at that timestamp
3. Look for: lip movement, mouth opening, facial expression matching speech, gesturing, voice characteristics
4. Assign the speaker's character ID and name from the CHARACTER REGISTRY
5. If a speaking character is not yet registered, add them to new_characters

## Confidence levels
- `"high"` — lip movement visible, clear visual cue (mouth open, holding phone), OR distinct voice match to a visible character
- `"low"` — speaker inferred from framing, shot composition, or context (character facing away, off-screen speaker)
- Use `speaker_id: "unknown"` only when you truly cannot determine the speaker

## Character registration
- Register a character only when they appear on screen for the FIRST TIME
- Check CHARACTER REGISTRY first — do not re-register existing characters
- `id`: descriptive slug like `char_driver`, `char_woman_red_dress`
- `name`: actual name if known from dialogue, otherwise shortest unique descriptor
- `appearance`: clothing, hair, build, distinguishing features
- Do NOT register background extras

## Rules
- Copy dialogue `text`, `timestamp_start`, `timestamp_end` exactly as provided
- Do not speculate about motivations or predict events
- Reference characters as `Name [char_id]` if you need to mention them
- Focus entirely on speaker identification — do not describe visual events

## Output
Return JSON with:
- "dialogues": array of speaker-assigned dialogue lines
- "new_characters": array of newly registered characters (empty [] if none)
"""


AV_DIARIZE_CHUNK_PROMPT = """--- TARGET CHUNK ---
Dialogue range: [{start_timecode} - {end_timecode}]

--- CONTEXT DIALOGUES (reference only — do NOT diarize) ---
{context_subs_text}

--- DIALOGUES TO DIARIZE (assign speaker for each) ---
{target_subs_text}

--- CHARACTER REGISTRY ---
{character_registry}

Identify the speaker for each dialogue line under DIALOGUES TO DIARIZE using both video frames and audio. Assign character IDs from the registry. Register any NEW characters not already listed."""
