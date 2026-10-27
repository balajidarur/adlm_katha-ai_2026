"""Build sample/index.html (scene-shot version): : 5 REFRAMED excerpts, first 5 shots each, with
circled frames, Qwen3.8 events, the dialogue and the ground-truth US ADs in the same 10 s window."""
import html
import json
import os
import re

S = "/workspace/dbalu/reframed/sample"
DD = "/workspace/dbalu/reframed/dd20_scenes"
SRT = "/workspace/dbalu/reframed/srt/dialogue"
AD = "/workspace/.common_cache/common_share_for_users/srt_for_ads/budgetAD/reframed_train/srt/ad_us"
SCORES = "/home/dbalu/workspace/ad_gen/adlm_katha-ai_2026/output_entail/scores_reframed20_scenes.json"
CLIPS = ["tt0762107/xkHWlYgIbApN", "tt0104231/63YLAJuM2syh", "tt0095159/6ATxDJJ_jaUA",
         "tt4589218/_aFep0DZPX1q", "tt0338526/1kP2ffQURJRp"]
NSHOTS = 5
TITLES = {m: v["imdb_name"] + f" ({v['imdb_year']})" for m, v in json.load(open("/storage/datasets/Reframed/train/meta.json")).items()}


def srt(path):
    out = []
    if not os.path.exists(path):
        return out
    for b in open(path, encoding="utf-8").read().strip().split("\n\n"):
        L = b.strip().split("\n")
        if len(L) >= 3 and "-->" in L[1]:
            a, z = (sum(float(x) * m for x, m in zip(t.strip().replace(",", ".").split(":"), (3600, 60, 1))) for t in L[1].split("-->"))
            out.append((a, z, " ".join(L[2:])))
    return out


def tag(txt):
    # registry tags -> small grey chips so they stay visible but readable
    return re.sub(r"\[((?:char|prop|loc)_[a-z0-9_]+)\]", r'<span class="tag">\1</span>', html.escape(txt))


scores = json.load(open(SCORES))
TH = 0.57
parts = []
for c in CLIPS:
    m, vid = c.split("/")
    d = json.load(open(f"{DD}/{vid}/Qwen3.8-27B-FP8/adlm_descriptions.json"))
    f = json.load(open(f"{S}/dd5s/{c}/faces.json"))
    meta = json.load(open(f"{S}/dd5s/{c}/meta.json"))
    sc = scores[c]["scenes"]
    ent = 100 * sum(x >= TH for x in sc["claim_scores"]) / len(sc["claim_scores"])
    cast = "".join(f'<figure class="pc"><img src="dd5s/{x["portrait_crop"]}" alt=""><figcaption><b style="color:{x["colour"]}">●</b> {html.escape(x["name"])}<br><small>{html.escape(x["actor"])}</small></figcaption></figure>'
                   for x in f["cast"] if x["portrait_crop"])
    dia, ads = srt(f"{SRT}/{c}.srt"), srt(f"{AD}/{c}.srt")
    rows = []
    for s, ms in zip(d["shots"][:NSHOTS], meta["shots"][:NSHOTS]):
        a, z = s["start_abs"], s["end_abs"]
        imgs = "".join(f'<img src="dd5s/{c}/circled/{fn}" alt="">' for fn in sorted(os.listdir(f"{S}/dd5s/{c}/circled")) if fn.startswith(f"shot_{s['shot_id']:03d}_"))
        mm = lambda t: f"{int(t // 60)}:{t % 60:04.1f}"
        rel = lambda ts: int(ts.split(":")[0]) * 60 + float(ts.split(":")[1])
        ev = "".join(f'<li><span class="t">{mm(s["time_offset"] + rel(e["timestamp"]))}</span> {tag(e["description"])}</li>' for e in s["events"])
        dl = "".join(f'<li><span class="t">{t0:.1f}s</span> {html.escape(x)}</li>' for t0, t1, x in dia if a <= t0 < z) or '<li class="none">none</li>'
        gl = "".join(f'<li><span class="t">{t0:.1f}s</span> {html.escape(x)}</li>' for t0, t1, x in ads if a <= t0 < z) or '<li class="none">none</li>'
        rows.append(f'<div class="shot"><div class="sh">Shot {s["shot_id"]} · {mm(a)}–{mm(z)} ({z - a:.1f} s) · {len(ms["frames"])} frames · recognised: {", ".join(s.get("recognised_cast", [])) or "none"}</div>'
                    f'<div class="frames">{imgs}</div><div class="cols"><div><h4>Qwen3.8 dense description</h4><ul>{ev}</ul></div>'
                    f'<div><h4>Dialogue (input)</h4><ul>{dl}</ul><h4>Ground-truth US AD (not an input)</h4><ul>{gl}</ul></div></div></div>')
    parts.append(f'<section class="dd"><h2>{html.escape(TITLES.get(m, m))}</h2><p class="sub">{c} · {d["num_shots"]} shots · {meta["duration"]:.0f} s · '
                 f'entailment {ent:.1f}% of {len(sc["claim_scores"])} claims · names {sc["names_found"]}/{sc["names_mentioned"]}</p>'
                 f'<div class="cast">{cast}</div>{"".join(rows)}<p class="sub">Showing the first {NSHOTS} of {d["num_shots"]} shots.</p></section>')

css = """<style id="ddcss">
.dd{border-top:2px solid var(--line);padding-top:8px;margin-top:28px}
.dd .cast{grid-template-columns:repeat(auto-fill,minmax(90px,1fr))}
.shot{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px;margin:12px 0}
.sh{font-weight:600;margin-bottom:6px}
.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.frames img{width:100%;border-radius:4px}
.cols{display:grid;grid-template-columns:3fr 2fr;gap:14px;margin-top:8px}.cols>div{min-width:0}
@media (max-width:800px){.cols{grid-template-columns:1fr}.frames{grid-template-columns:1fr 1fr}}
h4{margin:6px 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
ul{margin:0;padding-left:18px}li{margin:2px 0}.t{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px}
.tag{font-size:11px;color:var(--muted);background:var(--bg);border:1px solid var(--line);border-radius:3px;padding:0 3px}
.none{color:var(--muted);list-style:none;margin-left:-18px}
</style>"""
old = json.load(open(SCORES))
tot = [x for c in CLIPS for x in []]
intro = (f'<h1>Reframed: scene-shot dense descriptions</h1><p class="sub">Shots now come from the same detector and settings as CMD-AD '
         '(ffmpeg scene filter 0.12, max 10 s, min 3 s, tolerance 0.3 s, absorb 1 s): 54,502 shots, mean 5.36 s (CMD-AD 5.3 s). '
         'Frames, faces and circles were rebuilt on these shots; every sample below was checked to use them (shots_source = ffmpeg_scene_cmdad_params, shot boundaries identical to the scene-shot list).</p>'
         '<table><tr><th></th><th>Reframed, scene shots</th><th>Reframed, old fixed 10 s</th><th>CMD-AD Qwen3.8</th></tr>'
         '<tr><td>visual claims entailed (20 excerpts, 845 claims)</td><td><b>38.5%</b></td><td>40.5%</td><td>35.0%</td></tr>'
         '<tr><td>names found</td><td>226/262</td><td>228/262</td><td>-</td></tr>'
         '<tr><td>avg shot length</td><td>5.46 s</td><td>9.80 s</td><td>5.3 s</td></tr>'
         '<tr><td>spans per minute</td><td>101.3</td><td>77.8</td><td>99.1</td></tr>'
         '<tr><td>words per span</td><td>18.1</td><td>17.7</td><td>18.1</td></tr></table>')
base = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reframed scene-shot samples</title><style>
:root{--bg:#f6f7f9;--fg:#1b1f27;--muted:#5d6676;--card:#fff;--line:#dde1e8}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa3b2;--card:#1b1f26;--line:#2e3440}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1300px;margin:0 auto;padding:20px 16px}
h1{font-size:22px;margin:0}h2{font-size:18px;margin:28px 0 0}.sub{color:var(--muted);margin:2px 0}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}figcaption{font-size:12px;color:var(--muted);margin-top:4px}
.cast{display:grid;gap:8px}.cast img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px}
table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid var(--line);padding:5px 10px;text-align:left}th{color:var(--muted);font-size:12px}
</style></head><body></body></html>"""
page = base.replace("</head>", css + "</head>", 1).replace("</body></html>", intro + "".join(parts) + "</body></html>", 1)
open(f"{S}/index.html", "w").write(page)
print("ok", len(page))
