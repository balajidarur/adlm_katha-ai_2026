"""Side-by-side HTML of circled frames: 2 REFRAMED excerpts + 1 CMD-AD clip (same prep code)."""
import json, os, shutil
S = "/workspace/dbalu/reframed/sample"
CMD = "/storage/users/dbalu/dd_inputs/prep_360p"
CMD_CLIP = "2011/009tNfQRd4o"
NAMES = {"tt7668870": "Searching (2018)", "tt9686708": "The King of Staten Island (2020)"}

# CMD-AD example: copy 6 multi-face circled frames + crops
f = json.load(open(f"{CMD}/{CMD_CLIP}/faces.json"))
fr = [x for s in f["shots"].values() for x in s if len(x["dets"]) >= 2] or [x for s in f["shots"].values() for x in s if x["dets"]]
os.makedirs(f"{S}/cmdad/{CMD_CLIP}/circled", exist_ok=True)
for x in fr[::max(1, len(fr) // 6)][:6]:
    shutil.copy(f"{CMD}/{CMD_CLIP}/circled/{x['frame']}", f"{S}/cmdad/{CMD_CLIP}/circled/")
shutil.copy(f"{CMD}/{CMD_CLIP}/faces.json", f"{S}/cmdad/{CMD_CLIP}/")
os.makedirs(f"{S}/cmdad/_crops", exist_ok=True)
for c in f["cast"]:
    if c["portrait_crop"] and os.path.exists(f"{CMD}/{c['portrait_crop']}"):
        shutil.copy(f"{CMD}/{c['portrait_crop']}", f"{S}/cmdad/_crops/")


def block(title, root, clip):
    f = json.load(open(f"{S}/{root}{clip}/faces.json"))
    cast = "".join(
        f'<figure><img src="{root}{c["portrait_crop"]}" alt=""><figcaption><b style="color:{c["colour"]}">●</b> '
        f'{c["name"]}<br><small>{c["actor"]}</small></figcaption></figure>' if c["portrait_crop"] else
        f'<figure><div class="noimg">no face</div><figcaption>{c["name"]}<br><small>{c["actor"]}</small></figcaption></figure>'
        for c in f["cast"])
    labels = {c["id"]: c["name"] for c in f["cast"]}
    det = {x["frame"]: x for s in f["shots"].values() for x in s}
    imgs = ""
    for fn in sorted(os.listdir(f"{S}/{root}{clip}/circled")):
        x = det[fn]
        who = ", ".join(f'{f["cast"][d["cast_idx"]]["name"]} ({d["score"]:.2f})' for d in x["dets"]) or "no match"
        imgs += f'<figure><img src="{root}{clip}/circled/{fn}" alt=""><figcaption>{fn} · t={x["t"]}s · faces={x["n_faces"]}<br>{who}</figcaption></figure>'
    return f'<section><h2>{title}</h2><p class="sub">{clip}</p><h3>Cast bank (top 10)</h3><div class="cast">{cast}</div><h3>Circled frames</h3><div class="grid">{imgs}</div></section>'


body = block("REFRAMED · " + NAMES["tt7668870"], "", "tt7668870/bZMXNyydj4WX")
body += block("REFRAMED · " + NAMES["tt9686708"], "", "tt9686708/dFvqK9HbW1_X")
body += block("CMD-AD reference (same code)", "cmdad/", CMD_CLIP)
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reframed circle check</title><style>
:root{{--bg:#f6f7f9;--fg:#1b1f27;--muted:#5d6676;--card:#fff;--line:#dde1e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171c;--fg:#e6e9ef;--muted:#9aa3b2;--card:#1b1f26;--line:#2e3440}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:1300px;margin:0 auto;padding:20px 16px}}
h1{{font-size:22px;margin:0}}h2{{font-size:18px;margin:28px 0 0}}h3{{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:14px 0 6px}}
.sub{{color:var(--muted);margin:2px 0}}figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px}}
figcaption{{font-size:12px;color:var(--muted);margin-top:4px}}figcaption b{{font-size:14px}}
.cast{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}}.cast img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px}}
.noimg{{aspect-ratio:1;display:grid;place-items:center;color:var(--muted);border:1px dashed var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:10px}}.grid img{{width:100%;border-radius:4px}}
</style></head><body><h1>Reframed circle check</h1>
<p class="sub">Frames from prep_charbank.py on box A: InsightFace buffalo_l, ArcFace match to the top-10 cast portraits (threshold 0.2), shot smoothing, one colour per character. Number in brackets is the match score.</p>
{body}</body></html>"""
open(f"{S}/index.html", "w").write(html)
print("ok")
