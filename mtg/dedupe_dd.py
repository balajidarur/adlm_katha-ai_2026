"""
Post-process MTG dense descriptions: drop exact-duplicate events inside a shot (repetition loops) and report loop stats.

For every {root}/{vid}/{model}/adlm_descriptions.json: the raw files are kept as *_raw.json (once), events whose
normalised text (registry tags, case, punctuation and whitespace removed) already occurred earlier in the same shot
are dropped, and adlm_descriptions.json / adlm_timeline.json are rewritten from the deduped shots.

    python dedupe_dd.py --root /scratch/mtg/dd_full --model Qwen3.8-27B-FP8 --stats stats.json
    python dedupe_dd.py --root ... --dry-run     # stats only
A shot counts as a loop when it had >= LOOP_DUPS duplicate events or >= LOOP_EVENTS events.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_adlm_descriptions import format_mmss, mmss_to_seconds  # noqa: E402

LOOP_DUPS, LOOP_EVENTS = 3, 30


def norm(t):
    t = re.sub(r"\[(?:char|prop|loc)_[^\]]*\]", "", t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def process(d, dry):
    raw_p = os.path.join(d, "adlm_descriptions_raw.json")
    desc = json.load(open(raw_p if os.path.exists(raw_p) else os.path.join(d, "adlm_descriptions.json")))
    tl_raw_p = os.path.join(d, "adlm_timeline_raw.json")
    tl = json.load(open(tl_raw_p if os.path.exists(tl_raw_p) else os.path.join(d, "adlm_timeline.json")))
    st = {"shots": 0, "empty_shots": 0, "events_raw": 0, "events": 0, "dups": 0, "loop_shots": 0, "words": 0}
    timeline = []
    for s in desc["shots"]:
        seen, kept = set(), []
        for e in s["events"]:
            k = norm(e.get("description", ""))
            if k and k not in seen:
                seen.add(k)
                kept.append(e)
        n_dup = len(s["events"]) - len(kept)
        st["shots"] += 1
        st["empty_shots"] += not kept
        st["events_raw"] += len(s["events"])
        st["events"] += len(kept)
        st["dups"] += n_dup
        st["loop_shots"] += (n_dup >= LOOP_DUPS or len(s["events"]) >= LOOP_EVENTS)
        st["words"] += sum(len(e["description"].split()) for e in kept)
        s["events"] = kept
        s["n_events_raw"] = n_dup + len(kept)
        for e in kept:
            try:
                t = mmss_to_seconds(e["timestamp"]) + s["time_offset"]
            except Exception:  # noqa: BLE001
                t = s["start_abs"]
            timeline.append({"timestamp_abs": format_mmss(t), "shot_id": s["shot_id"], "type": "event",
                             "description": e["description"]})
    if not dry:
        if not os.path.exists(raw_p):
            os.replace(os.path.join(d, "adlm_descriptions.json"), raw_p)
            os.replace(os.path.join(d, "adlm_timeline.json"), tl_raw_p)
        timeline.sort(key=lambda e: mmss_to_seconds(e["timestamp_abs"]))
        desc["dedup"] = {"rule": "exact normalised duplicate within shot", **{k: st[k] for k in ("events_raw", "events", "dups")}}
        tl.update({"timeline": timeline, "total_events": len(timeline)})
        json.dump(desc, open(os.path.join(d, "adlm_descriptions.json"), "w"), indent=2, ensure_ascii=False)
        json.dump(tl, open(os.path.join(d, "adlm_timeline.json"), "w"), indent=2, ensure_ascii=False)
    dur = desc["shots"][-1]["end_abs"] if desc["shots"] else 0
    return {**st, "dur_s": dur}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="Qwen3.8-27B-FP8")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    per = {}
    for p in sorted(glob.glob(os.path.join(a.root, "*", a.model, "adlm_descriptions*.json"))):
        d = os.path.dirname(p)
        vid = d.split(os.sep)[-2]
        if vid not in per:
            per[vid] = process(d, a.dry_run)
    tot = {k: sum(v[k] for v in per.values()) for k in ("shots", "empty_shots", "events_raw", "events", "dups", "loop_shots", "words", "dur_s")}
    tot["videos"] = len(per)
    tot["dup_rate"] = round(tot["dups"] / max(tot["events_raw"], 1), 4)
    tot["loop_shot_rate"] = round(tot["loop_shots"] / max(tot["shots"], 1), 4)
    tot["videos_with_loop"] = sum(v["loop_shots"] > 0 for v in per.values())
    tot["events_per_min"] = round(tot["events"] / max(tot["dur_s"], 1) * 60, 1)
    tot["words_per_event"] = round(tot["words"] / max(tot["events"], 1), 1)
    print(json.dumps(tot, indent=1))
    if a.stats:
        json.dump({"total": tot, "per_video": per}, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
