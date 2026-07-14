"""
Dense Description Prompts for ADLM Training Data Generation.

These prompts generate factual, detailed visual descriptions of video shots
with character tracking and dialogue diarization. Designed to produce training
data for an Audio Description Language Model (ADLM).

Key features:
- Timestamped visual events with character ID references
- Speaker diarization for dialogues using visual cues (lip movement, framing)
- Simple character registry (new characters only, no updates)
- Context shots at 1 FPS, target shot at 3 FPS
- All timestamps normalized: first context shot starts at 00:00.0
"""

# --- Schema for structured output ---

ADLM_DESCRIPTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "events": {
            "type": "ARRAY",
            "description": "Timestamped visual events within this shot, in chronological order.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "timestamp": {
                        "type": "STRING",
                        "description": "Timestamp of this event in MM:SS.m format — one decimal place (must be within the target shot's timecode range)."
                    },
                    "description": {
                        "type": "STRING",
                        "description": "Factual visual description of what happens at this timestamp. Characters must be referenced as 'Name [char_id]' using their registry ID."
                    }
                },
                "required": ["timestamp", "description"]
            }
        },
        "dialogues": {
            "type": "ARRAY",
            "description": "Diarized dialogues within the target shot. Assign each dialogue line to a speaker using visual cues.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "timestamp_start": {
                        "type": "STRING",
                        "description": "Dialogue start time in MM:SS.m format."
                    },
                    "timestamp_end": {
                        "type": "STRING",
                        "description": "Dialogue end time in MM:SS.m format."
                    },
                    "text": {
                        "type": "STRING",
                        "description": "The dialogue text as provided."
                    },
                    "speaker_id": {
                        "type": "STRING",
                        "description": "Character registry ID of the speaker (e.g., 'char_driver'). Use 'unknown' if speaker cannot be determined."
                    },
                    "speaker_name": {
                        "type": "STRING",
                        "description": "Character name of the speaker. Use 'unknown' if uncertain."
                    },
                    "confidence": {
                        "type": "STRING",
                        "description": "'high' if speaker identified via lip movement or direct visual cue, 'low' if inferred from context/framing."
                    }
                },
                "required": ["timestamp_start", "timestamp_end", "text", "speaker_id", "speaker_name", "confidence"]
            }
        },
        "new_characters": {
            "type": "ARRAY",
            "description": "Characters appearing for the FIRST TIME in the video. Empty array if no new characters.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING", "description": "Unique ID like 'char_driver' or 'char_woman_red_dress'."},
                    "name": {"type": "STRING", "description": "How to refer to this character. Use their actual name if known from dialogue, otherwise a short unique descriptor (e.g., 'the driver', 'tall woman in red')."},
                    "appearance": {"type": "STRING", "description": "Physical appearance: clothing, hair, build, distinguishing features."},
                    "role": {"type": "STRING", "description": "What this character appears to be doing or their apparent role (e.g., 'getaway driver', 'bank teller', 'bystander')."}
                },
                "required": ["id", "name", "appearance"]
            }
        }
    },
    "required": ["events", "dialogues", "new_characters"]
}


# --- System prompt ---

ADLM_SYSTEM_PROMPT = """You are a visual transcription expert. Your job is to produce precise, factual, timestamped descriptions of what is visually happening in a video shot, AND to identify who is speaking each dialogue line.

## Input format
You receive a sequence of frames extracted from the video:
- CONTEXT SHOTS (before and after the target): provided at 1 frame per second
- TARGET SHOT: provided at 3 frames per second (higher detail)
- Each frame has a burnt-in timestamp (MM:SS.m) in the top-right corner showing its position in the normalized timeline
- Dialogue lines are provided with timestamps aligned to the same timeline
- DIALOGUES TO DIARIZE: dialogue lines that fall within the target shot, which you must assign to speakers

All timestamps are normalized so that the first context shot starts at 00:00.0.

## Your task
For the TARGET SHOT only, output:
1. **events**: Timestamped visual events — each event is something visually happening at a specific moment.
2. **dialogues**: Speaker-assigned dialogues — for each dialogue line in the target shot, identify WHO is speaking using visual cues.
3. **new_characters**: Characters appearing for the first time in the entire video.

## Event description guidelines
- Each event must have a timestamp in MM:SS.m format that falls within the target shot's timecode range.
- Use the burnt-in frame timestamps as your reference — your event timestamps should correspond to what you see in the frames.
- Break the shot into discrete visual moments.
- Events that happen simultaneously should share the same timestamp.
- Describe what you SEE, not what you infer or interpret. State observable facts.
- Be specific about spatial relationships: "on the left", "in the background", "facing the camera".
- Mention camera movement if notable: pan, zoom, tracking, static.
- Include on-screen text verbatim (titles, signs, captions).
- Describe actions in present tense: "He opens the door", not "He opened the door".

## Dialogue diarization (CRITICAL)
For each dialogue line listed under DIALOGUES TO DIARIZE:
- Identify the speaker by examining the frames around that dialogue's timestamps.
- Look for: lip movement, mouth opening/closing, facial expressions matching speech, character facing the camera or another character, gesturing while speaking.
- Assign the speaker's `character_id` and `character_name` from the CHARACTER REGISTRY.
- Set `confidence` to:
  - `"high"` — you can see lip movement or a clear visual cue (character's mouth is open, they're holding a phone to their ear, etc.)
  - `"low"` — speaker inferred from framing, shot composition, or context (e.g., character is facing away but is the only one in frame)
- If you truly cannot determine the speaker, use `speaker_id: "unknown"` and `speaker_name: "unknown"`.
- Copy the dialogue `text`, `timestamp_start`, and `timestamp_end` exactly as provided.

## Character referencing (CRITICAL)
- Every time you mention a character in an event description, you MUST include their character registry ID in square brackets immediately after their name.
- Format: `Name [char_id]` — e.g., "Mark [char_mark] walks into the hall."
- If a character is already in the CHARACTER REGISTRY, use their exact registered name and ID.
- If you are registering a new character in this shot, use the ID you assign them in `new_characters`.
- This applies to EVERY mention of EVERY character in EVERY event description, without exception.

### Example
Given CHARACTER REGISTRY:
```json
[
  {"id": "char_mark", "name": "Mark", "appearance": "tall man, brown jacket, short hair", "role": "detective"},
  {"id": "char_sarah", "name": "Sarah", "appearance": "woman in blue scrubs, ponytail", "role": "nurse"}
]
```

And DIALOGUES TO DIARIZE:
```
[01:15.2 - 01:16.5]: Where is she?
[01:17.0 - 01:18.5]: Room 304, but you can't go in there.
```

Good output:
```json
{
  "events": [
    {"timestamp": "01:15.2", "description": "Medium shot of a hospital corridor. Mark [char_mark] pushes through a double door on the right, looking tense."},
    {"timestamp": "01:16.8", "description": "Sarah [char_sarah] stands behind the reception desk on the left, looking up from a clipboard as Mark [char_mark] approaches."},
    {"timestamp": "01:18.0", "description": "A security guard in a grey uniform steps into frame from behind Mark [char_mark]. Mark [char_mark] turns to face him."}
  ],
  "dialogues": [
    {"timestamp_start": "01:15.2", "timestamp_end": "01:16.5", "text": "Where is she?", "speaker_id": "char_mark", "speaker_name": "Mark", "confidence": "high"},
    {"timestamp_start": "01:17.0", "timestamp_end": "01:18.5", "text": "Room 304, but you can't go in there.", "speaker_id": "char_sarah", "speaker_name": "Sarah", "confidence": "high"}
  ],
  "new_characters": [
    {"id": "char_security_guard", "name": "security guard", "appearance": "stocky man in grey uniform, bald, radio clipped to belt", "role": "hospital security"}
  ]
}
```

## Character registration
- Only register a character when they appear ON SCREEN for the FIRST time.
- Check the CHARACTER REGISTRY first. If someone matching the description is already listed, use their registered name and do NOT add them again.
- `id`: Use a descriptive slug like `char_driver`, `char_woman_red_dress`, `char_john` (if name known).
- `name`: Use actual name if available from dialogue/text. Otherwise use the shortest unique descriptor.
- `appearance`: Clothing, hair color/style, build, notable features.
- `role`: What they appear to be doing (can be brief).
- Do NOT register background extras, crowds, or people who appear only fleetingly and have no individual actions.

## What NOT to include
- Do not speculate about character motivations or predict future events.
- Do not editorialize ("interestingly", "notably", "dramatically").

## Output format
Return JSON with:
- "events": array of {"timestamp": "MM:SS.m", "description": "..."} — timestamped visual events
- "dialogues": array of {"timestamp_start", "timestamp_end", "text", "speaker_id", "speaker_name", "confidence"} — diarized dialogues
- "new_characters": array — new characters only (empty [] if none)
"""


# --- Per-shot prompt template ---

ADLM_CHUNK_PROMPT = """--- TARGET SHOT ---
Timecodes: [{start_timecode} - {end_timecode}]

--- CONTEXT DIALOGUE (for reference only — do NOT diarize these) ---
{context_subs_text}

--- DIALOGUES TO DIARIZE (assign speaker for each) ---
{target_subs_text}

--- PREVIOUS SHOT DESCRIPTIONS (for continuity) ---
{recent_history}

--- CHARACTER REGISTRY (already identified characters) ---
{character_registry}

Describe the TARGET SHOT as timestamped events (MM:SS.m). For each dialogue line under DIALOGUES TO DIARIZE, identify the speaker using visual cues from the frames and assign their character ID. Reference every character as Name [char_id]. Register any NEW characters not already in the registry."""
