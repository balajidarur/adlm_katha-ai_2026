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
                        "description": "Factual visual description of what happens at this timestamp. Characters as 'Name [char_id]', props as 'Name [prop_id]', locations as 'Name [loc_id]'."
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
        },
        "new_props": {
            "type": "ARRAY",
            "description": "Props appearing for the FIRST TIME in the video. Empty array if no new props.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING", "description": "Unique ID like 'prop_duffel_bag' or 'prop_shotgun'."},
                    "name": {"type": "STRING", "description": "Short name for this prop (e.g., 'black duffel bag', 'silver coin', 'yellow car')."},
                    "description": {"type": "STRING", "description": "Visual appearance: color, size, distinguishing features."}
                },
                "required": ["id", "name", "description"]
            }
        },
        "new_locations": {
            "type": "ARRAY",
            "description": "Locations/settings appearing for the FIRST TIME in the video. Empty array if no new locations.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING", "description": "Unique ID like 'loc_warehouse' or 'loc_bank_entrance'."},
                    "name": {"type": "STRING", "description": "Short name for this location (e.g., 'warehouse interior', 'bank parking lot')."},
                    "description": {"type": "STRING", "description": "Visual appearance: lighting, notable features, spatial layout."}
                },
                "required": ["id", "name", "description"]
            }
        }
    },
    "required": ["events", "new_characters", "new_props", "new_locations"]
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
3. A list of NEW props (only those appearing for the very first time in the entire video).
4. A list of NEW locations (only those appearing for the very first time in the entire video).

## Event description guidelines
- Each event must have a timestamp in MM:SS.m format that falls within the target shot's timecode range.
- Use the burnt-in frame timestamps as your reference — your event timestamps should correspond to what you see in the frames.
- Each event must capture one atomic visual action — a single distinct change visible in the frames.
- Events that happen simultaneously should share the same timestamp.
- Describe ONLY what is physically visible in the frame. No inferences, no emotional labels, no interpretations of intent.
- Be specific about spatial relationships: "on the left", "in the background", "facing the camera".
- Open the first event of each shot with the shot type and camera angle:
  - Shot types: extreme close-up, close-up, medium close-up, medium, medium wide, wide, extreme wide, over-the-shoulder, POV
  - Camera angles: eye-level, low-angle, high-angle, overhead, dutch/tilted
  - Example: "Low-angle medium shot of the warehouse interior [loc_warehouse]."
- When the camera moves during the shot, describe the movement as its own event:
  - Movement types: pan (left/right), tilt (up/down), dolly/tracking (forward/backward/lateral), zoom (in/out), handheld shake, crane, static
  - Example: "Camera pans left to reveal the yellow car [prop_yellow_car] parked against the wall."
- Include on-screen text verbatim (titles, signs, captions).
- Describe actions in present tense: "He opens the door", not "He opened the door".

### Gestures and expressions (CRITICAL)
- Never use emotional labels like "angry", "happy", "tense", "concerned", "nervous". Describe the physical appearance instead:
  - BAD: "She looks angry"
  - GOOD: "Her eyebrows draw together, lips press into a thin line, jaw tightens"
- For gestures, describe the exact physical movement — limb positions, hand shapes, body angles:
  - BAD: "He gestures dismissively"
  - GOOD: "He raises his right hand palm-outward to shoulder height, fingers splayed, then drops it to his side"
- For body language, describe posture and position changes:
  - BAD: "She stands nervously"
  - GOOD: "She shifts her weight to her left foot, arms crossed tightly against her chest, chin lowered"

## Tagging — Characters, Props, and Locations (CRITICAL)
Every time you mention a character, prop, or location in an event description, you MUST tag it with its registry ID in square brackets.

### Character tags
- Format: `Name [char_id]` — e.g., "Mark [char_mark] walks into the hall."
- If already in the CHARACTER REGISTRY, use their exact registered name and ID.
- If new, use the ID you assign in `new_characters`.

### Prop tags
- Format: `Name [prop_id]` — e.g., "the black duffel bag [prop_duffel_bag] sits on the floor."
- Tag any object that a character interacts with, carries, uses, or that is visually prominent in the scene.
- Examples: bags, weapons, vehicles, phones, keys, coins, documents, tools, food/drinks.
- If already in the PROPS REGISTRY, reuse the existing ID. If new, register in `new_props`.
- Do NOT tag generic background furniture (walls, ceiling, floor) unless a character interacts with it.

### Location tags
- Format: `Name [loc_id]` — e.g., "inside the warehouse [loc_warehouse]."
- Tag the setting/location in the first event of a shot when the location is visible.
- If the location changes mid-shot (e.g., a character walks from one room to another), tag the new location.
- If already in the LOCATION REGISTRY, reuse the existing ID. If new, register in `new_locations`.

All three tag types apply to EVERY mention in EVERY event description, without exception.

### Example
Given registries:
- CHARACTER REGISTRY: Mark (char_mark), Sarah (char_sarah)
- PROPS REGISTRY: clipboard (prop_clipboard)
- LOCATION REGISTRY: hospital corridor (loc_hospital_corridor)

Good output:
```json
{
  "events": [
    {"timestamp": "01:15.2", "description": "Eye-level medium shot of the hospital corridor [loc_hospital_corridor]. Mark [char_mark] pushes through a double door [prop_double_door] on the right, eyebrows raised, mouth slightly open."},
    {"timestamp": "01:16.8", "description": "Sarah [char_sarah] stands behind the reception desk on the left, lifting her gaze from the clipboard [prop_clipboard] as Mark [char_mark] approaches."},
    {"timestamp": "01:18.0", "description": "A security guard in a grey uniform steps into frame from behind Mark [char_mark], right hand resting on the radio [prop_radio] clipped to his belt. Mark [char_mark] turns to face him, chin dropping slightly."}
  ],
  "new_characters": [
    {"id": "char_security_guard", "name": "security guard", "appearance": "stocky man in grey uniform, bald, radio clipped to belt", "role": "hospital security"}
  ],
  "new_props": [
    {"id": "prop_double_door", "name": "double door", "description": "pair of beige swinging doors with round windows"},
    {"id": "prop_radio", "name": "radio", "description": "black two-way radio clipped to belt"}
  ],
  "new_locations": []
}
```

Note: The security guard, double door, and radio are new — registered in their respective arrays. Mark, Sarah, clipboard, and hospital corridor were already in registries — referenced by tag only, NOT re-registered.

## Registration rules
For characters, props, and locations: only register when they appear ON SCREEN for the FIRST time. Always check the existing registry first — if already listed, reuse the ID and do NOT re-register.

### Characters
- `id`: slug like `char_driver`, `char_woman_red_dress`, `char_john`.
- `name`: actual name if known from dialogue/text, otherwise shortest unique descriptor.
- `appearance`: clothing, hair, build, notable features.
- `role`: what they appear to be doing (brief).
- Do NOT register background extras or crowds.

### Props
- `id`: slug like `prop_duffel_bag`, `prop_shotgun`, `prop_yellow_car`.
- `name`: short descriptor (e.g., "black duffel bag", "silver revolver").
- `description`: color, size, distinguishing features.
- Register props that characters interact with or that are visually prominent. Skip generic background objects.

### Locations
- `id`: slug like `loc_warehouse`, `loc_bank_parking_lot`, `loc_apartment_kitchen`.
- `name`: short descriptor (e.g., "warehouse interior", "bank entrance").
- `description`: lighting, spatial layout, notable features.

## What NOT to include
- Do not speculate about character motivations or predict future events.
- Do not editorialize ("interestingly", "notably", "dramatically").
- Do not describe dialogue content — just note that characters are speaking if relevant to the visual action (e.g., "he speaks into the phone").

## Output format
Return JSON with:
- "events": array of {"timestamp": "MM:SS.m", "description": "..."} — timestamped visual events in chronological order
- "new_characters": array — new characters only (empty [] if none)
- "new_props": array — new props only (empty [] if none)
- "new_locations": array — new locations only (empty [] if none)
"""


# --- Model-specific prompt additions ---

GEMMA_CONCISENESS_SUFFIX = """

## STRICT: No Redundancy (HIGHEST PRIORITY — OVERRIDES ALL OTHER RULES)
- NEVER produce two events that describe the same visual state. "Remains bright white", "stays bright white", "is still bright white" are THE SAME event — output it ONCE and stop.
- NEVER describe a frame where nothing has changed since the previous event. If the scene looks the same, skip that frame entirely.
- Do NOT create one event per frame. You receive frames at 3 FPS but most consecutive frames show NO visual change. Only create an event when something NEW and VISIBLY DIFFERENT happens.
- For static shots (logos, title cards, a character just standing or talking with no other action): maximum 2 events total.
- For a character speaking with no other physical action: ONE event noting they speak. Do NOT describe every mouth position or jaw movement across frames.
- BEFORE outputting, compare each event to the one before it. If they describe the same thing with different words, DELETE the duplicate.
- If your output has more than 8 events for a single shot, something is almost certainly wrong — go back and merge redundant events."""


# --- Per-shot prompt template ---

ADLM_CHUNK_PROMPT = """--- TARGET SHOT ---
Timecodes: [{start_timecode} - {end_timecode}]

--- DIALOGUE (timestamps aligned to frames) ---
{subs_text}

--- PREVIOUS SHOT DESCRIPTIONS (for continuity) ---
{recent_history}

--- CHARACTER REGISTRY (already identified characters) ---
{character_registry}

--- PROPS REGISTRY (already identified props) ---
{props_registry}

--- LOCATION REGISTRY (already identified locations) ---
{location_registry}

Describe the TARGET SHOT as a sequence of timestamped events using MM:SS.m format. Your event timestamps should match the burnt-in frame timestamps you see. Tag every character as Name [char_id], every prop as Name [prop_id], every location as Name [loc_id]. Register any NEW characters, props, or locations not already in their registries."""
