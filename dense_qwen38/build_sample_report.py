"""Build a local HTML page showing, per shot, exactly what was sent to Qwen (frames, portraits, prompt) and what came back.

    python build_sample_report.py output_qwen_samples   # expects {root}/{vid}/{model}/requests/*.json + adlm_descriptions.json
"""
import base64
import glob
import html
import json
import os
import sys

root = sys.argv[1]
page = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Qwen Input Check</title><style>",
        ":root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--card:#f5f5f7;--line:#ddd;--acc:#0a66c2}",
        "@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#eee;--mut:#aaa;--card:#1d1d21;--line:#333;--acc:#6aa9ff}}",
        "body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:16px}",
        "details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:8px 12px}",
        "summary{cursor:pointer;font-weight:600}.imgs{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0}",
        ".imgs img{height:110px;border-radius:3px}.por img{height:80px}.lab{font-weight:600;margin-top:10px;color:var(--acc)}",
        "pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);padding:8px;border-radius:6px;max-height:420px;overflow:auto;font-size:12px}",
        "table{border-collapse:collapse}td,th{border-bottom:1px solid var(--line);padding:3px 8px;text-align:left;vertical-align:top}.mut{color:var(--mut)}",
        "</style></head><body><h1>Qwen3.8-27B input/output check</h1>",
        "<p class='mut'>Each shot shows the exact request sent to vLLM (portraits, labelled target/context frames from <code>circled/</code> with burnt timestamps, system + user prompt) and the parsed JSON reply.</p>"]
for desc_path in sorted(glob.glob(os.path.join(root, "*", "*", "adlm_descriptions.json"))):
    d = os.path.dirname(desc_path)
    vid = os.path.basename(os.path.dirname(d))
    out = json.load(open(desc_path))
    timings = json.load(open(os.path.join(d, "timings.json")))
    img_dir = os.path.join(d, "img")
    os.makedirs(img_dir, exist_ok=True)
    rel = os.path.relpath(img_dir, root)
    page.append(f"<h2>{html.escape(out['cmd_filename'])} — {out['num_shots']} shots</h2>")
    page.append("<p>Cast: " + ", ".join(f"{html.escape(c['name'])} ({c['colour']})" for c in out["cast"]) + "</p>")
    by_id = {s["shot_id"]: s for s in out["shots"]}
    calls = {c["shot_id"]: c for c in timings["per_call"]}
    for rp in sorted(glob.glob(os.path.join(d, "requests", "shot_*.json"))):
        sid = int(os.path.basename(rp)[5:8])
        msgs = json.load(open(rp))
        s, c = by_id[sid], calls[sid]
        page.append(f"<details><summary>Shot {sid} [{s['start_abs']:.1f}–{s['end_abs']:.1f}s] — {c['images']} images, "
                    f"{c['prompt_tokens']} in / {c['output_tokens']} out tokens, {c['latency_s']}s, {len(s['events'])} events</summary>")
        page.append("<div class='lab'>Output events</div><table>" + "".join(
            f"<tr><td>{html.escape(e.get('time',''))}</td><td>{html.escape(e.get('description',''))}</td></tr>" for e in s["events"]) + "</table>")
        texts, block, k = [], [], 0
        for part in msgs[1]["content"]:
            if part["type"] == "text":
                if block:
                    page.append("<div class='imgs" + (" por" if "PORTRAIT" in label else "") + "'>" + "".join(block) + "</div>")
                    block = []
                label = part["text"]
                if len(label) < 400:
                    page.append(f"<div class='lab'>{html.escape(label)}</div>")
                else:
                    texts.append(label)
            else:
                fn = f"s{sid:03d}_{k:03d}.jpg"; k += 1
                open(os.path.join(img_dir, fn), "wb").write(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
                block.append(f"<img loading='lazy' src='{rel}/{fn}'>")
        if block:
            page.append("<div class='imgs'>" + "".join(block) + "</div>")
        page.append("<div class='lab'>User prompt text</div><pre>" + html.escape("\n\n".join(texts)) + "</pre>")
        page.append("<details><summary>System prompt</summary><pre>" + html.escape(msgs[0]["content"]) + "</pre></details>")
        page.append("<div class='lab'>Raw reply (new registries)</div><pre>" + html.escape(json.dumps(
            {k2: s[k2] for k2 in ("new_characters", "new_props", "new_locations")}, indent=1, ensure_ascii=False)) + "</pre></details>")
page.append("</body></html>")
open(os.path.join(root, "index.html"), "w").write("\n".join(page))
print("wrote", os.path.join(root, "index.html"))
