"""
Character-bank-aware prompts for ADLM dense descriptions.

Builds on prompts_adlm.py: same event / tagging / registry rules, plus an
AutoAD-Zero-style cast list with coloured face circles and portrait crops.
"""

from prompts_adlm import ADLM_SYSTEM_PROMPT, ADLM_DESCRIPTION_SCHEMA

ADLM_CHARBANK_SECTION = """

## CAST LIST, PORTRAITS AND FACE CIRCLES (CRITICAL — OVERRIDES the character naming rules above)
You are given the movie's main cast (top-billed characters). For each cast member you get:
- their character name and registry ID (already in the CHARACTER REGISTRY),
- the actor's portrait face crop,
- a unique circle colour.

In the frames, a face recognition system has drawn a coloured circle (ellipse) around faces it identified.
A circle of a given colour means that person is the cast member with that colour.

Rules:
1. When a face is circled, refer to that person by the matching cast name and ID, e.g. "Christine Brown [char_christine_brown]".
2. The same person stays the same character across frames and shots even when their face is not circled
   (turned away, too small, out of focus) — use continuity, clothing, the portraits and the dialogue to keep naming them.
3. A person who is never circled and does not match any portrait is NOT a cast member: register them in `new_characters`
   with a short descriptor (e.g. "the waiter") — do NOT guess a cast name for them.
4. Face recognition can be wrong. If a circle clearly contradicts what you see (e.g. a male face circled with a female
   character's colour, or the portrait looks nothing like the person), trust what you see.
5. The coloured circles are provided for character identification ONLY. NEVER mention circles, colours of circles,
   portraits, or the face recognition system in your descriptions.
6. Do NOT re-register cast members in `new_characters` — they are already registered.
"""

ADLM_CHARBANK_SYSTEM_PROMPT = ADLM_SYSTEM_PROMPT + ADLM_CHARBANK_SECTION

ADLM_CHARBANK_CHUNK_PROMPT = """--- TARGET SHOT ---
Timecodes: [{start_timecode} - {end_timecode}]

--- CAST FACES RECOGNISED IN THE TARGET SHOT (from face recognition; circled in the frames) ---
{face_summary}

--- DIALOGUE (timestamps aligned to frames) ---
{subs_text}

--- PREVIOUS SHOT DESCRIPTIONS (for continuity) ---
{recent_history}

--- CHARACTER REGISTRY (cast + already identified characters) ---
{character_registry}

--- PROPS REGISTRY (already identified props) ---
{props_registry}

--- LOCATION REGISTRY (already identified locations) ---
{location_registry}

Describe the TARGET SHOT as a sequence of timestamped events using MM:SS.m format. Your event timestamps should match the burnt-in frame timestamps you see. Name cast members by their cast name when you can identify them. Tag every character as Name [char_id], every prop as Name [prop_id], every location as Name [loc_id]. Register any NEW non-cast characters, props, or locations not already in their registries. Never mention the face circles."""

ADLM_CHARBANK_SCHEMA = ADLM_DESCRIPTION_SCHEMA


# Optional extra guidance appended to the system prompt (charbank_pipeline.py --prompt-variant).
# Generic writing rules only: no examples taken from any clip, character or event.
_PV_DESCRIPTOR = """
- Each time you name a person in an event, follow the name (on its first mention in that event) with a brief visible
  descriptor: apparent gender and age group plus one salient visual trait. People without a name get the descriptor alone."""
_PV_ACTION = """
- Focus on actions. Start each event with who does what, as a plain subject-verb-object sentence with a concrete action verb.
  Prefer describing what people do over describing states, poses or what is merely visible.
- State interactions explicitly: which person acts toward which person or object, and how.
- Describe visible facial expressions and emotional reactions in plain words.
- Avoid filler events that only report talking, lip movement, blinking or holding still, unless nothing else happens.
  Keep each event short (under 25 words)."""
_PV_ACTION_V2 = _PV_ACTION.split("- Avoid filler")[0] + """- Keep subtle moments too when they carry meaning: a pause, a held look, a blink, someone freezing or holding still.
  Give them their own event, but describe the actions first.
- Keep each event short (under 25 words)."""
_PV_DENSE = """
- Start a new event for every visible change (new action, gesture, expression, object handling, entrance or exit, cut),
  up to one event per frame timestamp. Do not merge several actions into one long event."""
_PV_ACTION_V3 = _PV_ACTION_V2.replace("""
- Keep each event short (under 25 words).""", "")

_PV_ACTION_V4 = _PV_ACTION_V3 + """
- Keep each event brief: one action or change per event, in a single short sentence."""

_PV_ACTION_V5 = _PV_ACTION_V3 + """
- Keep each event brief: one action or change per event, in a single short sentence (under 25 words)."""

PROMPT_VARIANTS = {
    "base": "",
    "desc": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR,
    "action": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_ACTION,
    "desc_action": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION,
    "desc_action_v2": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION_V2,
    "desc_action_v3": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION_V3,
    "desc_action_v4": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION_V4,
    "desc_action_v5": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION_V5,
    "desc_action_dense": "\n\n## WRITING STYLE (IMPORTANT)" + _PV_DESCRIPTOR + _PV_ACTION + _PV_DENSE,
}


# --- Character-only tagging variant -------------------------------------------------------------
def _strip_prop_loc(base):
    s = base
    s = s.replace("""2. A list of NEW characters (only those appearing for the very first time in the entire video).
3. A list of NEW props (only those appearing for the very first time in the entire video).
4. A list of NEW locations (only those appearing for the very first time in the entire video).""",
                  """2. A list of NEW characters (only those appearing for the very first time in the entire video).""")
    s = s.replace("""- Be specific about spatial relationships: "on the left", "in the background", "facing the camera".""",
                  """- Be specific about spatial relationships: "on the left", "in the background", "facing the camera".
- Describe the setting of each shot and the objects people handle or that stand out, with their visible details (colour, size, material, lighting, layout).""")
    s = s.replace(" of the warehouse interior [loc_warehouse].", " of the warehouse interior.")
    s = s.replace("reveal the yellow car [prop_yellow_car] parked", "reveal the yellow car parked")
    start, end = s.index("## Tagging — Characters, Props, and Locations"), s.index("## What NOT to include")
    s = s[:start] + """## Character tags (CRITICAL)
Every time you mention a character in an event description, tag them with their registry ID in square brackets: `Name [char_id]`, e.g. "Mark [char_mark] walks into the hall."
- If already in the CHARACTER REGISTRY, use their exact registered name and ID.
- If new, use the ID you assign in `new_characters`.

## Registration rules
Only register characters when they appear ON SCREEN for the FIRST time. Check the CHARACTER REGISTRY first — if already listed, reuse the ID and do NOT re-register.
- `id`: slug like `char_driver`, `char_woman_red_dress`, `char_john`.
- `name`: actual name if known from dialogue/text, otherwise shortest unique descriptor.
- `appearance`: clothing, hair, build, notable features.
- `role`: what they appear to be doing (brief).
- Do NOT register background extras or crowds.

""" + s[end:]
    s = s.replace("""- "new_characters": array — new characters only (empty [] if none)
- "new_props": array — new props only (empty [] if none)
- "new_locations": array — new locations only (empty [] if none)""", """- "new_characters": array — new characters only (empty [] if none)""")
    assert "prop_" not in s and "loc_" not in s and "new_props" not in s and "tags" not in s.lower().replace("character tags", ""), "leftovers"
    return s


ADLM_CHARBANK_SYSTEM_PROMPT_NOTAGS = _strip_prop_loc(ADLM_SYSTEM_PROMPT) + ADLM_CHARBANK_SECTION.replace(
    "   portraits, or the face recognition system in your descriptions.",
    "   portraits, or the face recognition system in your descriptions. Never describe any coloured circle, ring,\n"
    "   ellipse, outline or highlight around a face or head — they are not part of the film.")

ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS = ADLM_CHARBANK_CHUNK_PROMPT.replace("""--- PROPS REGISTRY (already identified props) ---
{props_registry}

--- LOCATION REGISTRY (already identified locations) ---
{location_registry}

""", "").replace("Tag every character as Name [char_id], every prop as Name [prop_id], every location as Name [loc_id]. Register any NEW non-cast characters, props, or locations not already in their registries.",
                 "Tag every character as Name [char_id]. Register any NEW non-cast characters not already in the registry.")
assert "{props_registry}" not in ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS and "[prop_id]" not in ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS

import copy as _copy
_ev = _copy.deepcopy(ADLM_CHARBANK_SCHEMA["properties"]["events"])
_ev["items"]["properties"]["description"]["description"] = (
    "Factual visual description of what happens at this timestamp. Characters as 'Name [char_id]'.")
ADLM_CHARBANK_SCHEMA_NOTAGS = {
    **ADLM_CHARBANK_SCHEMA,
    "properties": {**{k: v for k, v in ADLM_CHARBANK_SCHEMA["properties"].items() if k not in ("new_props", "new_locations")}, "events": _ev},
    "required": ["events", "new_characters"],
}

# Safety net: drop any sentence that describes the face-recognition overlay.
import re as _re
CIRCLE_LEAK_RE = _re.compile(
    r"face[- ]recognition|\b(red|green|blue|yellow|cyan|magenta|orange|purple|pink|brown|colou?red)\s+(circle|outline|ring|ellipse|oval)s?\b"
    r"|\b(circle|outline|ring|ellipse|oval)s?\b[^.]{0,40}\b(around|traces?|encircl\w*|surround\w*)\b[^.]{0,30}\b(face|head)\b", _re.I)


def drop_circle_leaks(text):
    sents = _re.split(r"(?<=[.!?])\s+", text)
    kept = [x for x in sents if not CIRCLE_LEAK_RE.search(x)]
    return " ".join(kept) if kept else ""
