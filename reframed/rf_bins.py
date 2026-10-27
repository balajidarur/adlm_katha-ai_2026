"""Add the 9 budget-bin tokens to the REFRAMED gap-frames datasets with the CMD-AD script's own
convert() (add_bins_to_gapframes.py), then cap frames at 150 and keep clips <= 40k tokens (mk150 cap/est),
exactly as for the CMD-AD Qwen3.8 run (bins_f150_40k).

Budget source: CMD-AD uses Piper prefix durations rescaled to the WhisperX span; REFRAMED train has the
narrator's measured word times, so prefix_duration[k] = start of word k+1 - AD start (last = AD end).
Clip-end rule (user, 2026-10-27): an AD with no dialogue line after it is budgeted against the end of
the clip (content_end) instead of being left without budget tokens, so clips whose last lines are ADs
are kept and the model learns to write up to the end.
Val/test have no word times: their ADs stay as plain text, but every record gets the bin legend so the
prompt matches training, and no clip is dropped.

  python rf_bins.py --dir /scratch/rf_datasets/frame_variants_448x252
"""

import argparse
import json
import os
import sys

import numpy as np

CODE = "/scratch2/codeA"
WORDS = "/scratch2/rf_train/ad_word_times/ad_us"
sys.path.insert(0, CODE)

ab = {}
exec(open(f"{CODE}/add_bins_to_gapframes.py").read().split('if __name__ == "__main__":')[0], ab)
mk = {}
exec(open(f"{CODE}/mk150.py").read().split("d = json.load")[0], mk)


def load_budget_reframed():
    """video_id -> [(start, words, spoken)] like add_bins_to_gapframes.load_budget()."""
    out = {}
    for movie in os.listdir(WORDS):
        for f in os.listdir(f"{WORDS}/{movie}"):
            vid = f[:-5]
            for a in json.load(open(f"{WORDS}/{movie}/{f}"))["ads"]:
                ws = a.get("words") or []
                if not ws:
                    continue
                t0 = a["start"]
                spoken = [w["start"] - t0 for w in ws[1:]] + [a["end"] - t0]
                out.setdefault(vid, []).append((float(t0), [w["word"] for w in ws], spoken))
    for v in out:
        out[v].sort()
    return out


def convert_clip_end(rec, vid, budget, clip_end):
    """add_bins_to_gapframes.convert(), except the budget of an AD with no later dialogue line runs
    to clip_end (the clip-end rule) instead of the AD staying unbinned."""
    AD_RE, DLG_RE, BIN_TOKENS = ab["AD_RE"], ab["DLG_RE"], ab["BIN_TOKENS"]
    parse_ts, lookup, bin_of = ab["parse_ts"], ab["lookup"], ab["bin_of"]
    lines = rec["messages"][1]["content"].split("\n")
    dlg_starts = sorted(parse_ts(m.group(1)) for m in (DLG_RE.match(l) for l in lines) if m)
    out, n_ok, n_words, n_end = [], 0, 0, 0
    for line in lines:
        m = AD_RE.match(line)
        if not m:
            out.append(line)
            continue
        t, text = parse_ts(m.group(1)), m.group(2)
        ent = lookup(budget, vid, t)
        if ent is None:
            out.append(line)
            continue
        words, spoken = ent
        if " ".join(words).strip() != text.strip():
            words = text.split()
            if len(words) != len(spoken):
                out.append(line)
                continue
        nxt = [a for a in dlg_starts if a >= t]
        gap = (nxt[0] - t) if nxt else (clip_end - t)
        n_end += not nxt
        body = "".join(f"{BIN_TOKENS[bin_of(gap - p)]}{w} " for w, p in zip(words, spoken)).strip()
        out.append(f"{m.group(1)} [AD] {body}")
        n_ok += 1
        n_words += len(words)
    rec["messages"][1]["content"] = "\n".join(out)
    rec["messages"][0]["content"] = ab["BIN_LEGEND"] + "\n\n" + rec["messages"][0]["content"]
    return n_ok, n_words, n_end


def even_cap(rec, n):
    """Fallback when mk150.cap() cannot thin: already-resampled clips have frames > 1.5 s apart, so every
    frame is its own run and keeps its 1-frame minimum. Keep n frames evenly spread instead."""
    lines = rec["messages"][0]["content"].split("\n")
    idx = [i for i, l in enumerate(lines) if mk["FRAME_LINE"].match(l.strip())]
    keep = set(np.unique(np.linspace(0, len(idx) - 1, n).round().astype(int)).tolist())
    drop = {idx[k] for k in range(len(idx)) if k not in keep}
    rec["messages"][0]["content"] = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    rec["images"] = [im for k, im in enumerate(rec["images"]) if k in keep]
    assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--budget", type=int, default=40000)
    a = ap.parse_args()
    mk["MAX_F"] = a.max_frames
    budget = load_budget_reframed()
    print(f"budget: {sum(len(v) for v in budget.values()):,} ADs with word times over {len(budget):,} excerpts")
    info = json.load(open(f"{a.dir}/dataset_info.json"))
    for split in ("train", "validation", "test"):
        src = f"{a.dir}/cmdad_dds_gapframes_{split}.json"
        data, meta = json.load(open(src)), json.load(open(src.replace(".json", "_meta.json")))
        assert len(data) == len(meta)
        out, stats = [], {"clips": 0, "ads_binned": 0, "ads_binned_to_clip_end": 0, "words": 0, "no_binned_ad": 0}
        for rec, mt in zip(data, meta):
            n, w, n_end = convert_clip_end(rec, mt["video_id"], budget, mt["content_end"])   # also prepends the bin legend
            stats["ads_binned_to_clip_end"] += n_end
            assert rec["messages"][0]["content"].count("<image>") == len(rec["images"])
            stats["ads_binned"] += n
            stats["words"] += w
            stats["no_binned_ad"] += int(n == 0 and split == "train")
            if split == "train" and n == 0:
                continue            # same rule as CMD-AD: a train clip needs >= 1 binned AD
            out.append(rec)
        for r in out:
            mk["cap"](r)
            if len(r["images"]) > a.max_frames:
                even_cap(r, a.max_frames)
        est = np.array([mk["est"](r) for r in out])
        keep = [r for r, x in zip(out, est) if x <= a.budget]
        name = f"reframed_bins_f{a.max_frames}_{a.budget // 1000}k_{split}"
        json.dump(keep, open(f"{a.dir}/{name}.json", "w"), ensure_ascii=False)
        info[name] = dict(info[f"cmdad_dds_gapframes_{split}"], file_name=f"{name}.json")
        ke = np.array([mk["est"](r) for r in keep])
        stats.update(clips=len(keep), dropped_over_budget=len(out) - len(keep),
                     frames_max=int(max(len(r["images"]) for r in keep)), tokens_max=int(ke.max()))
        print(name, stats)
    json.dump(info, open(f"{a.dir}/dataset_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
