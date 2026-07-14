"""
Dense Description Prompts for ADLM Training Data Generation.

These prompts generate factual, detailed visual descriptions of video shots
with character tracking. Designed to produce training data for an Audio
Description Language Model (ADLM) that takes descriptions + dialogues as
input and generates timed AD text.

Key differences from the narrative-force pipeline:
- No narrative force / suspense / curiosity / surprise analysis
- No character screenshot extraction
- No entity update tracking — only new character registration
- Pure factual visual transcription: what is visible, who is present, what happens
- Characters are tracked in a simple registry (name + appearance + role)
- Each event is timestamped and characters are referenced with their registry ID inline
- Timestamps are normalized: first context shot starts at 00:00.0
- Context shots are provided at 1 FPS, target shot at 3 FPS
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
    "required": ["events", "new_characters"]
}


# --- System prompt ---

ADLM_SYSTEM_PROMPT = """You are a visual transcription expert. Your job is to produce precise, factual, timestamped descriptions of what is visually happening in a video shot.

## Input format
You receive a sequence of frames extracted from the video:
- CONTEXT SHOTS (before and after the target): provided at 1 frame per second
- TARGET SHOT: provided at 3 frames per second (higher detail)
- Each frame has a burnt-in timestamp (MM:SS.m) in the top-right corner showing its position in the normalized timeline
- Dialogue lines are provided with timestamps aligned to the same timeline

All timestamps are normalized so that the first context shot starts at 00:00.0.

## Your task
For the TARGET SHOT only, output:
1. A list of timestamped events — each event is something visually happening at a specific moment within the target shot (an action, a camera change, an expression shift, a new element appearing, etc.).
2. A list of NEW characters (only those appearing for the very first time in the entire video).

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

Good output:
```json
{
  "events": [
    {"timestamp": "01:15.2", "description": "Medium shot of a hospital corridor. Mark [char_mark] pushes through a double door on the right, looking tense."},
    {"timestamp": "01:16.8", "description": "Sarah [char_sarah] stands behind the reception desk on the left, looking up from a clipboard as Mark [char_mark] approaches."},
    {"timestamp": "01:18.0", "description": "A security guard in a grey uniform steps into frame from behind Mark [char_mark]. Mark [char_mark] turns to face him."}
  ],
  "new_characters": [
    {"id": "char_security_guard", "name": "security guard", "appearance": "stocky man in grey uniform, bald, radio clipped to belt", "role": "hospital security"}
  ]
}
```

Note: The security guard is new, so he gets registered in `new_characters`. Mark and Sarah were already in the registry, so they are only referenced by name + ID — NOT re-registered.

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
- Do not describe dialogue content — just note that characters are speaking if relevant to the visual action (e.g., "he speaks into the phone").

## Output format
Return JSON with:
- "events": array of {"timestamp": "MM:SS.m", "description": "..."} — timestamped visual events in chronological order
- "new_characters": array — new characters only (empty [] if none)
"""


# --- Per-shot prompt template ---

ADLM_CHUNK_PROMPT = """--- TARGET SHOT ---
Timecodes: [{start_timecode} - {end_timecode}]

--- DIALOGUE (timestamps aligned to frames) ---
{subs_text}

--- PREVIOUS SHOT DESCRIPTIONS (for continuity) ---
{recent_history}

--- CHARACTER REGISTRY (already identified characters) ---
{character_registry}

Describe the TARGET SHOT as a sequence of timestamped events using MM:SS.m format. Your event timestamps should match the burnt-in frame timestamps you see. Reference every character with their registry ID as Name [char_id]. Register any NEW characters not already in the registry."""
