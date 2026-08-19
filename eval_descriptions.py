"""
ADLM Description Evaluator — rate each event sentence against a looping video segment.

Keyboard:
    A = time & sentence both correct
    S = sentence has hallucination / partial mistake
    D = sentence correct but time prediction wrong
    F = sentence / prediction wrong
    J / K or arrow keys = move without rating
    Space = replay current segment
    C = jump to the comment box (Esc to leave it)

Saved to output/eval_ratings/<video>__<model>_ratings.json as
{"<event idx>": {"rating": "A", "comment": "..."}}. Older files that stored a
bare {"<idx>": "A"} are read as-is and upgraded on the next save.
Autosaves every 10 ratings, on comment blur, and on tab close.

Usage:
    python eval_descriptions.py --port 7860
"""

import argparse
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# The runs to evaluate: <output subdir> -> friendly video name.
VIDEOS = {
    "8fyjCygRaPc_3min": "robbery",
    "bc-TQzKoPQo_3min": "sunflower",
}

# Only these models are surfaced, in this order.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemma-4-31b-it"]


def parse_mmss(ts):
    parts = ts.split(":")
    return int(parts[0]) * 60 + float(parts[1])


def resolve_video(path):
    """Prefer the size-reduced copy — the originals are 200MB+ and seek slowly."""
    small = path.replace(".mp4", "_small.mp4")
    return small if os.path.exists(small) else path


def discover_runs(base_dir):
    """Build {run_key: {...}} for each (video, model) pair that has descriptions."""
    runs = {}
    for subdir, label in VIDEOS.items():
        for model in MODELS:
            desc = os.path.join(base_dir, subdir, model, "adlm_descriptions.json")
            if not os.path.exists(desc):
                continue
            with open(desc) as f:
                video = json.load(f).get("video", "")
            runs[f"{label} · {model}"] = {
                "desc": desc,
                "video": resolve_video(video),
                "label": label,
                "model": model,
                # ratings are keyed per (video, model) so the two never collide
                "slug": f"{label}__{model}",
            }
    return runs


def load_events(desc_path):
    with open(desc_path) as f:
        data = json.load(f)

    events = []
    for shot in data["shots"]:
        time_offset = shot.get("time_offset", 0)
        shot_events = shot.get("events", [])
        shot_end_abs = shot["end_abs"]

        for i, ev in enumerate(shot_events):
            abs_start = parse_mmss(ev["timestamp"]) + time_offset
            if i + 1 < len(shot_events):
                abs_end = parse_mmss(shot_events[i + 1]["timestamp"]) + time_offset
            else:
                abs_end = shot_end_abs
            abs_end = max(abs_end, abs_start + 0.5)

            events.append({
                "idx": len(events),
                "shot_id": shot["shot_id"],
                "t1": round(abs_start, 2),
                "t2": round(abs_end, 2),
                "timestamp": ev["timestamp"],
                "description": ev["description"],
            })
    return events


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ADLM Description Evaluator</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif;
         background: #14161a; color: #e6e8eb; }
  header { display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
           padding: 10px 16px; background: #1c1f26; border-bottom: 1px solid #2c313b; }
  h1 { font-size: 15px; margin: 0; font-weight: 600; }
  select { background: #262b34; color: #e6e8eb; border: 1px solid #3a4150;
           border-radius: 6px; padding: 6px 10px; font-size: 13px; }
  .stat { font-size: 13px; color: #9aa4b2; }
  .stat b { color: #e6e8eb; }
  #saveState { font-size: 12px; padding: 3px 9px; border-radius: 10px;
               background: #2a3140; color: #9aa4b2; }
  #saveState.dirty { background: #4a3a12; color: #f0c674; }
  #saveState.saved { background: #16391f; color: #7ddb92; }

  main { display: flex; padding: 14px; align-items: flex-start; }
  .left { flex: 0 0 44%; position: sticky; top: 14px; min-width: 240px; }
  .right { flex: 1; min-width: 0; }

  /* drag to rebalance video vs. table */
  .splitter { flex: 0 0 14px; align-self: stretch; cursor: col-resize; position: relative; }
  .splitter::before { content: ''; position: absolute; left: 6px; top: 0; bottom: 0;
                      width: 2px; background: #2c313b; border-radius: 1px; }
  .splitter:hover::before, .splitter.drag::before { background: #5b9dd9; width: 3px; left: 5px; }

  video { width: 100%; border-radius: 8px; background: #000; display: block; }
  .seg { margin-top: 10px; padding: 12px 14px; background: #1c1f26;
         border: 1px solid #2c313b; border-radius: 8px; }
  .seg .meta { font-size: 12px; color: #7f8b9c; margin-bottom: 6px;
               font-variant-numeric: tabular-nums; }
  .seg .desc { font-size: 15px; line-height: 1.5; }

  .keys { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
  .keys button { padding: 10px 6px; border-radius: 8px; cursor: pointer;
                 border: 1px solid #3a4150; background: #262b34; color: #e6e8eb;
                 font-size: 12px; line-height: 1.35; text-align: center; }
  .keys button:hover { background: #303743; }
  .keys button kbd { display: block; font-size: 15px; font-weight: 700; margin-bottom: 2px; }
  .keys button.k-a { border-color: #2f6b3d; }
  .keys button.k-s { border-color: #7a6220; }
  .keys button.k-d { border-color: #7a4a1e; }
  .keys button.k-f { border-color: #7a2c2c; }
  .hint { margin-top: 8px; font-size: 12px; color: #6c7787; }

  /* fixed layout keeps the rating column visible no matter how long the text is */
  table { width: 100%; border-collapse: collapse; font-size: 13px; table-layout: fixed; }
  td.desc { overflow-wrap: anywhere; }
  td.desc.clip { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  th { position: sticky; top: 0; background: #1c1f26; text-align: left;
       padding: 7px 9px; border-bottom: 1px solid #2c313b; font-size: 12px;
       color: #9aa4b2; z-index: 1; }
  /* drag handle sitting on each column's right edge */
  .colres { position: absolute; right: -3px; top: 0; bottom: 0; width: 7px;
            cursor: col-resize; z-index: 2; }
  .colres::after { content: ''; position: absolute; left: 3px; top: 5px; bottom: 5px;
                   width: 1px; background: #3a4150; }
  .colres:hover::after, .colres.drag::after { background: #5b9dd9; width: 2px; top: 0; bottom: 0; }
  th.spacer { width: auto; }
  td { padding: 7px 9px; border-bottom: 1px solid #23272f; vertical-align: top; }
  tr.row { cursor: pointer; }
  tr.row:hover td { background: #1e2229; }
  tr.row.active td { background: #26303f; }
  tr.row.active td:first-child { box-shadow: inset 3px 0 0 #5b9dd9; }
  td.time { white-space: nowrap; color: #8e9aab; font-variant-numeric: tabular-nums; }
  td.n { color: #6c7787; font-variant-numeric: tabular-nums; }
  td.rate { text-align: center; font-weight: 700; }
  td.cmt { color: #c9a86a; overflow-wrap: anywhere; font-size: 12px; }
  #commentBox { width: 100%; margin-top: 8px; min-height: 62px; resize: vertical;
                background: #14171d; color: #e6e8eb; border: 1px solid #3a4150;
                border-radius: 6px; padding: 8px 10px; font: inherit; font-size: 13px; }
  #commentBox:focus { outline: none; border-color: #5b9dd9; background: #171b22; }
  #commentBox::placeholder { color: #5d6775; }
  .speed { display: inline-block; padding: 1px 7px; border-radius: 9px;
           background: #2b3f5c; color: #9dc4ea; font-weight: 600; font-size: 11px; }
  .r-A { color: #7ddb92; } .r-S { color: #f0c674; }
  .r-D { color: #f0a35e; } .r-F { color: #ef7b7b; }
  .scroll { max-height: calc(100vh - 100px); overflow: auto;
            border: 1px solid #2c313b; border-radius: 8px; }
</style>
</head>
<body>

<header>
  <h1>ADLM Description Evaluator</h1>
  <select id="modelSel"></select>
  <span class="stat">Rated <b id="ratedN">0</b> / <b id="totalN">0</b></span>
  <span class="stat">A <b id="cA">0</b> · S <b id="cS">0</b> · D <b id="cD">0</b> · F <b id="cF">0</b></span>
  <span id="saveState">no changes</span>
  <button id="saveBtn" style="margin-left:auto;background:#2b3f5c;color:#dbe7f5;
          border:1px solid #3f5b80;border-radius:6px;padding:6px 12px;cursor:pointer;">
    Save now
  </button>
</header>

<main>
  <div class="left" id="leftPane">
    <video id="vid" preload="auto" playsinline muted></video>

    <div class="seg">
      <div class="meta" id="segMeta">Select an event to begin</div>
      <div class="desc" id="segDesc">—</div>
      <textarea id="commentBox" spellcheck="false"
                placeholder="Comment on this event — press C to jump here, Esc to go back"></textarea>
    </div>

    <div class="keys">
      <button class="k-a" data-r="A"><kbd>A</kbd>time + text<br>correct</button>
      <button class="k-s" data-r="S"><kbd>S</kbd>hallucination /<br>partial</button>
      <button class="k-d" data-r="D"><kbd>D</kbd>text ok,<br>time wrong</button>
      <button class="k-f" data-r="F"><kbd>F</kbd>wrong</button>
    </div>
    <div class="hint">J / K or ↑ ↓ to move · Space to replay · C to comment · autosaves every 10</div>
    <div class="hint">Drag the divider or any column edge to resize · double-click to reset ·
                      speed scales with segment length</div>
  </div>

  <div class="splitter" id="splitter" title="Drag to resize"></div>

  <div class="right">
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th data-col="n">#<span class="colres"></span></th>
            <th data-col="time">Time<span class="colres"></span></th>
            <th data-col="desc">Description<span class="colres"></span></th>
            <th data-col="rate">R<span class="colres"></span></th>
            <th data-col="cmt">Comment<span class="colres"></span></th>
            <th class="spacer"></th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</main>

<script>
const vid       = document.getElementById('vid');
const tbody     = document.getElementById('tbody');
const modelSel  = document.getElementById('modelSel');
const segMeta   = document.getElementById('segMeta');
const segDesc   = document.getElementById('segDesc');
const saveState = document.getElementById('saveState');

const commentBox = document.getElementById('commentBox');

let events = [];
let ratings = {};
let comments = {};
let cur = -1;
let model = '';
let unsaved = 0;   // ratings since last save, drives the every-10 autosave
let dirty = false; // any unpersisted edit, including comments

// ---------- looping ----------
let loopA = 0, loopB = 0, looping = false;

// 1x up to 3s, 2x up to 6s, 3x up to 9s, 4x from 12s on
function speedFor(dur) {
  return Math.min(4, Math.max(1, Math.ceil(dur / 3)));
}

function startLoop(a, b) {
  loopA = a; loopB = b; looping = true;
  vid.playbackRate = speedFor(b - a);
  vid.currentTime = a;
  vid.play().catch(() => {});
}

// rAF gives far tighter loop boundaries than the 'timeupdate' event (~4Hz)
function tick() {
  if (looping && vid.currentTime >= loopB) {
    vid.currentTime = loopA;
    if (vid.paused) vid.play().catch(() => {});
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---------- rendering ----------
function render() {
  tbody.innerHTML = '';
  events.forEach(ev => {
    const tr = document.createElement('tr');
    tr.className = 'row' + (ev.idx === cur ? ' active' : '');
    tr.dataset.idx = ev.idx;
    const r = ratings[ev.idx] || '';
    tr.innerHTML =
      '<td class="n">' + ev.idx + '</td>' +
      '<td class="time">' + ev.t1.toFixed(1) + '–' + ev.t2.toFixed(1) + '</td>' +
      '<td class="desc">' + escapeHtml(ev.description) + '</td>' +
      '<td class="rate ' + (r ? 'r-' + r : '') + '">' + r + '</td>' +
      '<td class="cmt">' + escapeHtml(comments[ev.idx] || '') + '</td>' +
      '<td></td>';
    tbody.appendChild(tr);
  });
  updateStats();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function updateStats() {
  const vals = Object.values(ratings);
  document.getElementById('ratedN').textContent = vals.length;
  document.getElementById('totalN').textContent = events.length;
  ['A','S','D','F'].forEach(k => {
    document.getElementById('c' + k).textContent = vals.filter(v => v === k).length;
  });
}

function select(i, scroll) {
  if (i < 0 || i >= events.length) return;
  cur = i;
  const ev = events[i];

  const dur = ev.t2 - ev.t1;
  segMeta.innerHTML =
    'Event ' + i + '  ·  shot ' + ev.shot_id +
    '  ·  ' + ev.t1.toFixed(1) + 's – ' + ev.t2.toFixed(1) + 's' +
    '  ·  ' + dur.toFixed(1) + 's  ' +
    '<span class="speed">' + speedFor(dur) + '×</span>';
  segDesc.textContent = ev.description;

  commentBox.value = comments[i] || '';

  [...tbody.children].forEach(tr =>
    tr.classList.toggle('active', +tr.dataset.idx === i));

  if (scroll !== false) {
    const row = tbody.querySelector('tr[data-idx="' + i + '"]');
    if (row) row.scrollIntoView({ block: 'nearest' });
  }
  startLoop(ev.t1, ev.t2);
}

function setRatingCell(i) {
  const row = tbody.querySelector('tr[data-idx="' + i + '"]');
  if (!row) return;
  const cell = row.children[3];
  const r = ratings[i] || '';
  cell.textContent = r;
  cell.className = 'rate ' + (r ? 'r-' + r : '');
}

function setCommentCell(i) {
  const row = tbody.querySelector('tr[data-idx="' + i + '"]');
  if (row) row.children[4].textContent = comments[i] || '';
}

// comments are free text, so persist on blur rather than on every keystroke
commentBox.addEventListener('input', () => {
  if (cur < 0) return;
  const v = commentBox.value;
  if (v.trim()) comments[cur] = v; else delete comments[cur];
  setCommentCell(cur);
  markDirty();
});
commentBox.addEventListener('blur', () => { if (dirty) save(); });
commentBox.addEventListener('keydown', e => {
  if (e.key === 'Escape') { e.preventDefault(); commentBox.blur(); }
});

function rate(r) {
  if (cur < 0) return;
  ratings[cur] = r;
  setRatingCell(cur);
  updateStats();
  unsaved++;
  markDirty();
  if (unsaved >= 10) save();
  if (cur < events.length - 1) select(cur + 1);
}

function markDirty() {
  dirty = true;
  saveState.className = 'dirty';
  saveState.textContent = unsaved > 0 ? unsaved + ' unsaved' : 'unsaved edits';
}

// ---------- persistence ----------
async function save() {
  if (!model) return;
  await fetch('/api/ratings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, ratings, comments })
  });
  unsaved = 0;
  dirty = false;
  const nc = Object.keys(comments).length;
  saveState.className = 'saved';
  saveState.textContent = 'saved · ' + Object.keys(ratings).length +
                          (nc ? ' + ' + nc + ' notes' : '');
}

async function loadModel(name) {
  // flush anything pending before switching away
  if (dirty) await save();

  model = name;
  const res  = await fetch('/api/events?model=' + encodeURIComponent(name));
  const data = await res.json();
  events   = data.events;
  ratings  = data.ratings;
  comments = data.comments || {};
  cur = -1;
  unsaved = 0;
  dirty = false;
  saveState.className = '';
  saveState.textContent = 'no changes';
  render();

  // swap the video only when the run actually points at a different file
  const ready = new Promise(res2 => {
    if (vid.getAttribute('src') === data.video) return res2();
    looping = false;
    vid.setAttribute('src', data.video);
    vid.addEventListener('loadedmetadata', () => res2(), { once: true });
    vid.load();
  });

  // resume at the first unrated event
  const first = events.findIndex(e => !(e.idx in ratings));
  const start = first === -1 ? 0 : first;
  await ready;
  select(start);
}

// ---------- events ----------
tbody.addEventListener('click', e => {
  const tr = e.target.closest('tr.row');
  if (tr) select(+tr.dataset.idx, false);
});

document.querySelectorAll('.keys button').forEach(b => {
  b.addEventListener('click', () => rate(b.dataset.r));
});

document.getElementById('saveBtn').addEventListener('click', save);
modelSel.addEventListener('change', () => loadModel(modelSel.value));

// ---------- resizable split ----------
const splitter = document.getElementById('splitter');
const leftPane = document.getElementById('leftPane');
let dragging = false;

const savedW = localStorage.getItem('evalLeftWidth');
if (savedW) leftPane.style.flex = '0 0 ' + savedW + 'px';

splitter.addEventListener('mousedown', e => {
  dragging = true;
  splitter.classList.add('drag');
  document.body.style.userSelect = 'none';
  document.body.style.cursor = 'col-resize';
  e.preventDefault();
});

document.addEventListener('mousemove', e => {
  if (!dragging) return;
  const rect = document.querySelector('main').getBoundingClientRect();
  const w = Math.max(240, Math.min(rect.width - 300, e.clientX - rect.left));
  leftPane.style.flex = '0 0 ' + w + 'px';
  localStorage.setItem('evalLeftWidth', Math.round(w));
  if (!colW.desc) applyCols();
});

document.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  splitter.classList.remove('drag');
  document.body.style.userSelect = '';
  document.body.style.cursor = '';
});

// double-click the divider to reset
splitter.addEventListener('dblclick', () => {
  leftPane.style.flex = '0 0 44%';
  localStorage.removeItem('evalLeftWidth');
  applyCols();
});

// ---------- resizable table columns ----------
// A trailing spacer column soaks up any leftover width, so every sized column
// is honoured exactly and dragging one never disturbs the others.
const COL_DEF = { n: 42, time: 92, rate: 40, cmt: 190 };
const ths = {};
document.querySelectorAll('th[data-col]').forEach(th => { ths[th.dataset.col] = th; });

let colW = {};
try { colW = JSON.parse(localStorage.getItem('evalColWidths') || '{}'); } catch (e) { colW = {}; }
const saveCols = () => localStorage.setItem('evalColWidths', JSON.stringify(colW));

function applyCols() {
  const sized = ['n', 'time', 'rate', 'cmt'];
  sized.forEach(k => { ths[k].style.width = (colW[k] || COL_DEF[k]) + 'px'; });
  // description fills the remaining space until the user sizes it by hand
  let d = colW.desc;
  if (!d) {
    const avail = document.querySelector('.scroll').clientWidth;
    const used = sized.reduce((s, k) => s + (colW[k] || COL_DEF[k]), 0);
    d = Math.max(160, avail - used - 4);
  }
  ths.desc.style.width = d + 'px';
}
applyCols();
window.addEventListener('resize', () => { if (!colW.desc) applyCols(); });

let colDrag = null;
document.querySelectorAll('th[data-col] .colres').forEach(h => {
  h.addEventListener('mousedown', e => {
    const th = h.parentElement;
    colDrag = { key: th.dataset.col, th, x: e.clientX, w: th.getBoundingClientRect().width };
    h.classList.add('drag');
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
    e.stopPropagation();
  });
  // double-click a handle to reset just that column
  h.addEventListener('dblclick', e => {
    delete colW[h.parentElement.dataset.col];
    saveCols();
    applyCols();
    e.stopPropagation();
  });
});

document.addEventListener('mousemove', e => {
  if (!colDrag) return;
  const w = Math.max(28, Math.round(colDrag.w + (e.clientX - colDrag.x)));
  colW[colDrag.key] = w;
  colDrag.th.style.width = w + 'px';
});

document.addEventListener('mouseup', () => {
  if (!colDrag) return;
  saveCols();
  document.querySelectorAll('.colres').forEach(h => h.classList.remove('drag'));
  colDrag = null;
  document.body.style.userSelect = '';
  document.body.style.cursor = '';
});

document.addEventListener('keydown', e => {
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  const k = e.key.toLowerCase();
  if (k === 'c') { e.preventDefault(); commentBox.focus(); }
  else if ('asdf'.includes(k)) { e.preventDefault(); rate(k.toUpperCase()); }
  else if (k === 'j' || e.key === 'ArrowDown') { e.preventDefault(); select(cur + 1); }
  else if (k === 'k' || e.key === 'ArrowUp')   { e.preventDefault(); select(cur - 1); }
  else if (e.key === ' ') { e.preventDefault(); if (cur >= 0) select(cur, false); }
});

window.addEventListener('beforeunload', () => {
  if (dirty) {
    navigator.sendBeacon('/api/ratings',
      new Blob([JSON.stringify({ model, ratings, comments })], { type: 'application/json' }));
  }
});

// ---------- boot ----------
(async () => {
  const models = await (await fetch('/api/models')).json();
  models.forEach(m => {
    const o = document.createElement('option');
    o.value = m; o.textContent = m;
    modelSel.appendChild(o);
  });
  if (models.length) await loadModel(models[0]);
})();
</script>
</body>
</html>
"""


def read_annotations(path):
    """Read {idx: {rating, comment}}, tolerating the older {idx: "A"} layout."""
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        raw = json.load(f)

    ratings, comments = {}, {}
    for k, v in raw.items():
        if isinstance(v, str):          # legacy file: idx -> "A"
            if v:
                ratings[k] = v
        elif isinstance(v, dict):
            if v.get("rating"):
                ratings[k] = v["rating"]
            if v.get("comment"):
                comments[k] = v["comment"]
    return ratings, comments


def write_annotations(path, ratings, comments):
    """Merge into {idx: {rating, comment}} and swap in atomically."""
    merged = {}
    for k in sorted(set(ratings) | set(comments), key=lambda s: int(s)):
        entry = {}
        if ratings.get(k):
            entry["rating"] = ratings[k]
        if comments.get(k):
            entry["comment"] = comments[k]
        if entry:
            merged[k] = entry

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)          # atomic: a crash mid-write can't truncate the real file
    return len(merged)


def build_app(base_dir, ratings_dir):
    runs = discover_runs(base_dir)
    if not runs:
        raise SystemExit(f"No runs found under {base_dir} for {list(VIDEOS)} x {MODELS}")

    app = FastAPI()

    # Mount each distinct video directory. StaticFiles honours HTTP Range
    # requests, which the <video> element needs in order to seek.
    mounted = {}
    for run in runs.values():
        vdir = os.path.dirname(os.path.abspath(run["video"]))
        if vdir not in mounted:
            mid = f"v{len(mounted)}"
            app.mount(f"/media/{mid}", StaticFiles(directory=vdir), name=f"media_{mid}")
            mounted[vdir] = mid
        run["url"] = f"/media/{mounted[vdir]}/{os.path.basename(run['video'])}"

    def ratings_path(key):
        return os.path.join(ratings_dir, f"{runs[key]['slug']}_ratings.json")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/models")
    def models():
        return list(runs.keys())

    @app.get("/api/events")
    def events(model: str):
        if model not in runs:
            return JSONResponse({"error": "unknown run"}, status_code=400)
        ratings, comments = read_annotations(ratings_path(model))
        return {
            "events": load_events(runs[model]["desc"]),
            "ratings": ratings,
            "comments": comments,
            "video": runs[model]["url"],
        }

    @app.post("/api/ratings")
    async def put_ratings(request: Request):
        body = await request.json()
        model = body["model"]
        if model not in runs:
            return JSONResponse({"error": "unknown run"}, status_code=400)
        n = write_annotations(
            ratings_path(model),
            body.get("ratings") or {},
            body.get("comments") or {},
        )
        return {"ok": True, "n": n}

    return app, runs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="output")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    ratings_dir = os.path.join(args.base_dir, "eval_ratings")
    os.makedirs(ratings_dir, exist_ok=True)

    app, runs = build_app(args.base_dir, ratings_dir)
    for key, run in runs.items():
        n = len(load_events(run["desc"]))
        mb = os.path.getsize(run["video"]) / 1e6
        print(f"  {key:38s} {n:4d} events   {os.path.basename(run['video'])} ({mb:.1f} MB)")
    print(f"Saving to {ratings_dir}/<video>__<model>_ratings.json")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
