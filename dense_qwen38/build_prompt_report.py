"""Local HTML: the desc_action prompt, the 20-clip prompt study results, and side-by-side examples."""
import html, json, random, re
from eval_entailment import load_ads, mmss, window, THRESH
from prompts_adlm_charbank import (PROMPT_VARIANTS, ADLM_CHARBANK_SYSTEM_PROMPT_NOTAGS, ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS,
                                   ADLM_CHARBANK_SCHEMA_NOTAGS, CIRCLE_LEAK_RE)

E = html.escape
OUT = "output_qwen_prompt/index.html"
clips = [l.strip() for l in open("output_entail/clips_prompt20.txt") if l.strip()]
S = json.load(open("output_entail/scores_prompt20.json"))
CL = json.load(open("output_entail/claims.json"))
ads = load_ads(clips)
RUNS = {"gemini35fl": ("Gemini 3.5 FL", "output/{v}/gemini-3.5-flash-lite"),
        "qwen_base": ("Qwen, current prompt", "output_qwen_dense/{v}/Qwen3.8-27B-FP8"),
        "desc": ("desc", "output_qwen_prompt/desc/{v}/Qwen3.8-27B-FP8"),
        "action": ("action", "output_qwen_prompt/action/{v}/Qwen3.8-27B-FP8"),
        "desc_action": ("desc_action", "output_qwen_prompt/desc_action/{v}/Qwen3.8-27B-FP8"),
        "desc_action_v2": ("desc_action_v2 (chosen)", "output_qwen_prompt/desc_action_v2/{v}/Qwen3.8-27B-FP8"),
        "desc_action_dense": ("desc_action_dense", "output_qwen_prompt/desc_action_dense/{v}/Qwen3.8-27B-FP8")}

def events(r, v, strip=True):
    tl = json.load(open(RUNS[r][1].format(v=v) + "/adlm_timeline.json"))["timeline"]
    return [(mmss(e["timestamp_abs"]), re.sub(r"\s*\[(?:char|prop|loc)_[a-z0-9_]+\]", "", e["description"]) if strip else e["description"])
            for e in tl if e.get("type", "event") == "event"]

rows = []
for r, (lab, _) in RUNS.items():
    sc = [s for c in S.values() for s in (c[r]["claim_scores"] or [])]
    n = w = 0
    for c in clips:
        ev = events(r, c.split("/")[1]); n += len(ev); w += sum(len(t.split()) for _, t in ev)
    rows.append((r, lab, 100 * sum(s >= THRESH for s in sc) / len(sc), sum(sc) / len(sc),
                 sum(c[r]["names_found"] for c in S.values()), sum(c[r]["names_mentioned"] for c in S.values()), n / len(clips), w / n))

# examples: claims where the best prompt flips a miss into a hit, plus a few where it still misses
flips, misses = [], []
for c in clips:
    v = c.split("/")[1]; EV = {r: events(r, v) for r in ("qwen_base", "desc_action", "gemini35fl")}
    k = 0
    for a in ads[c]:
        for x in [x for x in CL.get(a["key"], []) if x["visual"]]:
            sc = {r: S[c][r]["claim_scores"][k] for r in EV}; k += 1
            item = (c, a, x["text"], sc, {r: window(EV[r], a["t"]) for r in EV})
            if sc["desc_action"] >= THRESH > sc["qwen_base"]: flips.append(item)
            elif sc["desc_action"] < THRESH and sc["gemini35fl"] >= THRESH: misses.append(item)
random.seed(3)
flips, misses = random.sample(flips, min(8, len(flips))), random.sample(misses, min(4, len(misses)))

# one clip, first shots side by side (raw output with registry tags)
demo = clips[0]; dv = demo.split("/")[1]
D = {r: json.load(open(RUNS[r][1].format(v=dv) + "/adlm_descriptions.json"))["shots"] for r in ("qwen_base", "desc_action", "gemini35fl")}

css = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--card:#f5f5f7;--line:#ddd;--acc:#0a66c2;--good:#1a7f37;--bad:#b42318;--hl:#fff3c4}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121214;--fg:#eee;--mut:#aaa;--card:#1d1d21;--line:#333;--acc:#6aa9ff;--good:#4ac26b;--bad:#ff7b72;--hl:#3a3216}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1250px;padding:16px}
h2{margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}
tr.best td{font-weight:700;background:var(--hl)}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);padding:10px;border-radius:6px;font-size:12.5px;max-height:520px;overflow:auto}
.added{background:var(--hl);border-left:4px solid var(--acc)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.grid>div{font-size:13px}.mut{color:var(--mut)}.g{color:var(--good);font-weight:700}.b{color:var(--bad);font-weight:700}
details{margin:6px 0}summary{cursor:pointer;font-weight:600}@media (max-width:800px){.grid{grid-template-columns:1fr}}"""
P = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Qwen Prompt Study</title><style>{css}</style></head><body>",
     "<h1>Qwen3.8-27B prompt study: entailment of GT audio descriptions</h1>",
     f"<p class='mut'>20 random finished clips, {sum(len(c['qwen27b'] if False else c['qwen_base']['claim_scores'] or []) for c in S.values())} visual claims from {sum(len(ads[c]) for c in clips)} GT ADs. "
     "Claim = Gemini 2.5 Flash split of each AD, names neutralised. Entailed = DeBERTa-xlarge-MNLI P ≥ 0.57 against the description events within ±8 s. "
     "Names = cast names in the AD that appear in the description window. Same frames (360p, circled, 3/1 FPS), same model and settings for all Qwen rows; only the system prompt differs.</p>",
     "<h2>Results</h2><table><tr><th>Prompt</th><th>Entailed</th><th>Mean P</th><th>Names found</th><th>Events / clip</th><th>Words / event</th></tr>"]
for r, lab, ent, mp, nf, nm, epc, wpe in rows:
    P.append(f"<tr class='{'best' if r == 'desc_action_v2' else ''}'><td>{E(lab)}</td><td>{ent:.1f}%</td><td>{mp:.3f}</td><td>{nf}/{nm}</td><td>{epc:.0f}</td><td>{wpe:.1f}</td></tr>")
P.append("</table><p class='mut'>desc = name + visible descriptor; action = action-first writing rules; v2 = action rules but keeps meaningful pauses/blinks instead of banning them; dense = one event per visible change. "
         "The study ran with prop/location tags still in the prompt; the full run (below) uses the same writing rules with those tags removed.</p>")

P.insert(2, "<div class='card'><b>Final prompt used for the full run (qwen_dense_v3)</b>: base system prompt with character-only tags "
         "+ cast/face-circle section + desc_action_v3 writing style. Flags: <code>--prompt-variant desc_action_v3 --no-prop-loc-tags</code>. "
         "<a href='#prompt'>Jump to prompt ↓</a></div>")
P.append("<h2 id='prompt'>Final prompt (full run, v3)</h2>")
P.append("<div class='card'><b>Writing-style block (desc_action_v3)</b><pre class='added'>" + E(PROMPT_VARIANTS["desc_action_v3"].strip()) + "</pre></div>")
P.append("<details open><summary>Full system prompt, exactly as sent (the cast list is appended per movie)</summary><pre>" + E(ADLM_CHARBANK_SYSTEM_PROMPT_NOTAGS) +
         "<span class='added'>" + E(PROMPT_VARIANTS["desc_action_v3"]) + "</span>\n\n## CAST LIST\n- Name [char_id] — played by Actor — circle colour: RED\n- ...</pre></details>")
P.append("<details><summary>Per-shot user message</summary><pre>[CAST PORTRAITS] name + colour + face crop, for each cast member\n"
         "[CONTEXT SHOT BEFORE (t-2) | 1 FPS] + events summary of that shot, frames...\n[CONTEXT SHOT BEFORE (t-1) | 1 FPS] ...\n[TARGET SHOT | 3 FPS] frames (circled, timestamp burnt in)\n"
         "[CONTEXT SHOT AFTER (t+1) | 1 FPS] ...\n[CONTEXT SHOT AFTER (t+2) | 1 FPS] ...\n\n" + E(ADLM_CHARBANK_CHUNK_PROMPT_NOTAGS) + "</pre></details>")
P.append("<details><summary>Output JSON schema</summary><pre>" + E(json.dumps(ADLM_CHARBANK_SCHEMA_NOTAGS, indent=1)) + "</pre></details>")
P.append("<details><summary>Circle-leak safety filter (applied to every event)</summary><p>Any sentence matching this pattern is deleted before saving, "
         "so the face-circle overlay can never appear in the text:</p><pre>" + E(CIRCLE_LEAK_RE.pattern) + "</pre></details>")

def ex(item, title):
    c, a, claim, sc, win = item
    cls = lambda s: "g" if s >= THRESH else "b"
    P.append(f"<div class='card'><div class='mut'>{E(c)} · AD at {a['t']:.1f}s · {E(title)}</div><div><b>GT AD:</b> {E(a['text'])}</div>"
             f"<div><b>Claim:</b> {E(claim)} — base <span class='{cls(sc['qwen_base'])}'>{sc['qwen_base']:.2f}</span>, "
             f"desc_action <span class='{cls(sc['desc_action'])}'>{sc['desc_action']:.2f}</span>, Gemini <span class='{cls(sc['gemini35fl'])}'>{sc['gemini35fl']:.2f}</span></div>"
             "<details><summary>Description windows (±8 s)</summary><div class='grid'>" +
             "".join(f"<div><b>{n}</b><br>" + "<br>".join("• " + E(t) for t in win[r]) + "</div>"
                     for r, n in (("qwen_base", "Qwen current"), ("desc_action", "Qwen desc_action"), ("gemini35fl", "Gemini"))) + "</div></details></div>")

P.append(f"<h2>Examples: claims the new prompt now entails ({len(flips)} random)</h2>")
for it in flips: ex(it, "missed by current prompt, entailed by desc_action")
P.append(f"<h2>Examples: still missed but Gemini gets ({len(misses)} random)</h2>")
for it in misses: ex(it, "missed by desc_action, entailed by Gemini")

P.append(f"<h2>Side by side: {E(demo)}, first 6 shots (raw output)</h2>")
for i in range(min(6, len(D["qwen_base"]))):
    s = D["qwen_base"][i]
    P.append(f"<div class='card'><b>Shot {s['shot_id']}</b> <span class='mut'>{s['start_abs']:.1f}–{s['end_abs']:.1f}s</span><div class='grid'>" +
             "".join(f"<div><b>{n}</b><br>" + "<br>".join(f"<span class='mut'>{E(e.get('time',''))}</span> {E(e.get('description',''))}" for e in D[r][i]["events"]) + "</div>"
                     for r, n in (("qwen_base", "Qwen current"), ("desc_action", "Qwen desc_action"), ("gemini35fl", "Gemini"))) + "</div></div>")
P.append("</body></html>")
open(OUT, "w").write("\n".join(P))
print("wrote", OUT, len(flips), len(misses))
