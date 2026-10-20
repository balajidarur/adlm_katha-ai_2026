"""
How much of the ground-truth AD do the dense descriptions entail?

Same recipe as ../entail_val (eval2.py, the validated setting):
  1. Each GT AD is split into atomic claims by Gemini 2.5 Flash; only visual claims are kept,
     with names replaced by neutral descriptors ("Christine grabs it" -> "A woman grabs it").
  2. Premise = dense events within +-8 s of the AD start (nearest 24), registry tags stripped,
     split into sentences and grouped 3 at a time.
  3. DeBERTa-xlarge-MNLI P(entailment) for (group, claim); a claim's score is the max over groups.
     A claim counts as entailed at >= 0.57 (best-accuracy threshold vs Gemini yes/no claim labels:
     79% accuracy, 82% yes-vs-no ranking).
Also reported: name hit rate = of the cast names an AD mentions, the share that appear in the
description window (the visual claims are name-free, so step 1-3 can't see naming).

  python3 eval_entailment.py claims --clips clips.txt                 # Vertex Gemini, cached
  $LF_PY  eval_entailment.py score  --clips clips.txt --run gemini=output/{vid}/gemini-3.5-flash-lite/adlm_timeline.json
"""

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

CSV_PATH = "/storage/datasets/CMD-AD/data/cmd_ad_anno_v1.csv"
META_PATH = "/storage/datasets/CMD-AD/data/character_bank/metadata.json"
CLAIMS_CACHE = "output_entail/claims.json"
WIN, MAXEV, CHUNK, THRESH = 8.0, 24, 3, 0.57
NLI_MODEL = "microsoft/deberta-xlarge-mnli"

DECOMPOSE_SYS = """You split film audio descriptions into atomic claims.

For each numbered audio description, output the smallest standalone factual claims it asserts.
Rewrite each claim as a complete sentence that can be understood on its own.

Tag each claim visual=true ONLY if it could be verified by watching ~10 seconds of silent video
with no prior knowledge of the film: physical actions, objects, spatial relations, camera framing,
facial expression, gaze, clothing, setting.

Tag visual=false for anything requiring outside knowledge or inference:
- proper names ("Sarah grabs it" -> claim "A woman grabs it" is visual;
  a separate claim "Her name is Sarah" is visual=false)
- intentions, emotions stated as fact, plot knowledge, causation
- events happening off-screen or earlier/later

Replace named people with a neutral visual descriptor in the visual claims.
Keep each claim under 15 words."""

DECOMPOSE_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "id": {"type": "integer"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "visual": {"type": "boolean"}},
            "required": ["text", "visual"]}}},
    "required": ["id", "claims"]}}}, "required": ["items"]}


def load_ads(clips):
    ads = {c: [] for c in clips}
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["cmd_filename"] in ads:
                ads[r["cmd_filename"]].append({"t": float(r["scaled_start"]), "end": float(r["scaled_end"]),
                                               "text": r["text"], "imdbid": r["imdbid"]})
    for c in ads:
        ads[c].sort(key=lambda a: a["t"])
        for i, a in enumerate(ads[c]):
            a["key"] = f"{c}:{i}"
    return ads


def cmd_claims(args):
    from google import genai
    from google.genai import types
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_file(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/home/dbalu/tmp/creds.json"),
        scopes=["https://www.googleapis.com/auth/cloud-platform"])
    client = genai.Client(vertexai=True, credentials=creds,
                          project=os.environ.get("VERTEX_PROJECT", "bigls-cf-vi-qa-scoring-494618"),
                          location=os.environ.get("VERTEX_LOCATION", "us-central1"))
    os.makedirs(os.path.dirname(CLAIMS_CACHE), exist_ok=True)
    done = json.load(open(CLAIMS_CACHE)) if os.path.exists(CLAIMS_CACHE) else {}
    todo = [a for c in load_ads(read_clips(args.clips)).values() for a in c if a["key"] not in done]
    batches = [todo[i:i + 20] for i in range(0, len(todo), 20)]

    def run(batch):
        r = client.models.generate_content(
            model="gemini-2.5-flash", contents="\n".join(f"{i}. {a['text']}" for i, a in enumerate(batch)),
            config=types.GenerateContentConfig(system_instruction=DECOMPOSE_SYS, temperature=0.0,
                                               response_mime_type="application/json",
                                               response_schema=DECOMPOSE_SCHEMA))
        return {batch[it["id"]]["key"]: it["claims"] for it in json.loads(r.text)["items"]
                if 0 <= it["id"] < len(batch)}

    with ThreadPoolExecutor(16) as ex:
        for res in ex.map(run, batches):
            done.update(res)
    json.dump(done, open(CLAIMS_CACHE, "w"), indent=1)
    n_vis = sum(x["visual"] for v in done.values() for x in v)
    print(f"ADs with claims: {len(done)} (new {len(todo)}), visual claims: {n_vis}")


def read_clips(path):
    return [l.strip() for l in open(path) if l.strip()]


def mmss(s):
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def load_events(path, cast_names):
    """[(abs_time, text)] with registry tags stripped (cast names kept as written)."""
    tl = json.load(open(path))
    out = []
    for e in tl["timeline"]:
        if e.get("type", "event") != "event":
            continue
        txt = re.sub(r"\s*\[(?:char|prop|loc)_[a-z0-9_]+\]", "", e["description"]).strip()
        out.append((mmss(e["timestamp_abs"]), txt))
    return out


def window(events, t):
    idx = [i for i, (et, _) in enumerate(events) if t - WIN <= et <= t + WIN]
    idx.sort(key=lambda i: abs(events[i][0] - t))
    return [events[i][1] for i in sorted(idx[:MAXEV])]


def cast_names(imdbid, meta):
    names = set()
    for i, e in enumerate(meta.get(imdbid, {}).values()):
        if i >= 10:
            break
        for ch in e.get("characters", [])[:1]:
            for part in re.split(r"[\s/]+", ch):
                part = part.strip(".,\"'()")
                if len(part) >= 3 and part[0].isupper() and part.lower() not in {"the", "mrs", "mr", "dr", "himself", "herself"}:
                    names.add(part)
    return names


def cmd_score(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    clips = read_clips(args.clips)
    ads = load_ads(clips)
    claims = json.load(open(CLAIMS_CACHE))
    meta = json.load(open(META_PATH))
    runs = dict(r.split("=", 1) for r in args.run)

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, dtype=torch.float32).to(args.device).eval()
    ENT = [int(k) for k, v in mdl.config.id2label.items() if v.upper() == "ENTAILMENT"][0]

    results = {}
    for run, pattern in runs.items():
        work, meta_w, name_rows = [], [], []
        for c in clips:
            path = pattern.format(vid=c.split("/")[1], clip=c)
            if not os.path.exists(path):
                continue
            names = cast_names(ads[c][0]["imdbid"], meta) if ads[c] else set()
            events = load_events(path, names)
            for a in ads[c]:
                prem_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(window(events, a["t"]))) if len(s.strip()) > 8][:21]
                groups = [" ".join(prem_sents[i:i + CHUNK]) for i in range(0, len(prem_sents), CHUNK)] or [""]
                vis = [x["text"] for x in claims.get(a["key"], []) if x["visual"]]
                for j, h in enumerate(vis):
                    for g in groups:
                        work.append((g, h))
                        meta_w.append((c, a["key"], j))
                mentioned = {n for n in names if re.search(rf"\b{re.escape(n)}\b", a["text"])}
                if mentioned:
                    prem = " ".join(window(events, a["t"]))
                    name_rows.append((c, len(mentioned), sum(bool(re.search(rf"\b{re.escape(n)}\b", prem)) for n in mentioned)))
        order = sorted(range(len(work)), key=lambda k: len(work[k][0]) + len(work[k][1]))
        scores = [0.0] * len(work)
        for b in range(0, len(order), args.bs):
            ch = [work[k] for k in order[b:b + args.bs]]
            enc = tok([x[0] for x in ch], [x[1] for x in ch], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(args.device)
            with torch.no_grad():
                p = torch.softmax(mdl(**enc).logits.float(), -1)[:, ENT].cpu().tolist()
            for k, v in zip(order[b:b + args.bs], p):
                scores[k] = v
        best = {}
        for (c, key, j), s in zip(meta_w, scores):
            best[(c, key, j)] = max(best.get((c, key, j), 0.0), s)
        per_clip = {}
        for (c, key, j), s in best.items():
            per_clip.setdefault(c, []).append(s)
        names_by_clip = {}
        for c, n, hit in name_rows:
            a, b = names_by_clip.get(c, (0, 0))
            names_by_clip[c] = (a + n, b + hit)
        results[run] = {"per_clip": per_clip, "names": names_by_clip}

    rows_out = {}
    print(f"\nVisual-claim entailment (DeBERTa-xlarge-MNLI, max over 3-sentence groups in +-{WIN:.0f} s; entailed = P >= {THRESH})\n")
    hdr = f"{'clip':<20}{'ADs':>5}{'claims':>8}" + "".join(f"{r + ' mean P':>16}{r + ' %entailed':>18}{r + ' names':>14}" for r in runs)
    print(hdr)
    print("-" * len(hdr))
    allc = {r: [] for r in runs}
    alln = {r: [0, 0] for r in runs}
    for c in clips:
        line = f"{c:<20}{len(ads[c]):>5}"
        ncl = max((len(results[r]["per_clip"].get(c, [])) for r in runs), default=0)
        line += f"{ncl:>8}"
        for r in runs:
            v = results[r]["per_clip"].get(c)
            nm = results[r]["names"].get(c)
            if v:
                allc[r] += v
                line += f"{sum(v) / len(v):>16.3f}{100 * sum(x >= THRESH for x in v) / len(v):>17.1f}%"
            else:
                line += f"{'–':>16}{'–':>18}"
            if nm:
                alln[r][0] += nm[0]
                alln[r][1] += nm[1]
                line += f"{f'{nm[1]}/{nm[0]}':>14}"
            else:
                line += f"{'–':>14}"
            rows_out.setdefault(c, {})[r] = {"claim_scores": v, "names_mentioned": nm[0] if nm else 0,
                                             "names_found": nm[1] if nm else 0}
        print(line)
    print("-" * len(hdr))
    tot = f"{'ALL':<20}{sum(len(ads[c]) for c in clips):>5}{max(len(v) for v in allc.values()):>8}"
    for r in runs:
        v = allc[r]
        tot += (f"{sum(v) / len(v):>16.3f}{100 * sum(x >= THRESH for x in v) / len(v):>17.1f}%" if v else f"{'–':>16}{'–':>18}")
        tot += f"{f'{alln[r][1]}/{alln[r][0]}':>14}"
    print(tot)
    os.makedirs("output_entail", exist_ok=True)
    json.dump(rows_out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("claims")
    a.add_argument("--clips", required=True)
    b = sub.add_parser("score")
    b.add_argument("--clips", required=True)
    b.add_argument("--run", action="append", required=True, help="name=path pattern with {vid}")
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--bs", type=int, default=64)
    b.add_argument("--out", default="output_entail/scores.json")
    args = ap.parse_args()
    cmd_claims(args) if args.cmd == "claims" else cmd_score(args)


if __name__ == "__main__":
    main()
