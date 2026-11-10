"""Budget-bin tokens for the MTG gap-frames datasets (reframed/rf_bins.py logic, CMD-AD convert()):
per-word spoken prefix = Piper(1.2x) AD duration x cumulative-character share (MTG has no word times);
an AD with no later dialogue line is budgeted against the clip end (clip-end rule). Every split is binned
(GT durations exist for all); train keeps clips with >= 1 binned AD. Frames capped at --max-frames.

  python mtg_bins.py --dir $MTG/ds --max-frames 250 --budget 60000
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.abspath(os.path.join(HERE, "..", "..", "dataset"))
RF = os.path.abspath(os.path.join(HERE, "..", "..", "reframed"))
MTG = os.environ.get("MTG", "/scratch/mtg")
src = open(f"{RF}/rf_bins.py").read().replace('CODE = "/scratch2/codeA"', f'CODE = "{DS}"')
rf = {"__name__": "rf_bins"}
exec(src.split("def main():")[0], rf)


def load_budget_mtg():
    tts = json.load(open(f"{MTG}/gt_tts.json"))
    split = json.load(open(f"{MTG}/split.json"))
    out = {}
    for f in ("validation", "test"):
        for l in open(f"{MTG}/hf/data/{f}.jsonl"):
            r = json.loads(l)
            for a in r["gt_ad_segments"]:
                t = a["text"].strip()
                if not t:
                    continue
                words = t.split(); D = tts[t]
                cum = np.cumsum([len(w) + 1 for w in words], dtype=float); cum /= cum[-1]
                out.setdefault(r["video_id"], []).append((float(a["start_seconds"]), words, list(D * cum)))
    for v in out:
        out[v].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--budget", type=int, default=60000)
    a = ap.parse_args()
    rf["mk"]["MAX_F"] = a.max_frames
    budget = load_budget_mtg()
    info = json.load(open(f"{a.dir}/dataset_info.json"))
    for split in ("train", "val10", "test"):
        p = f"{a.dir}/cmdad_dds_gapframes_{split}.json"
        data, meta = json.load(open(p)), json.load(open(p.replace(".json", "_meta.json")))
        out, st = [], {"clips_in": len(data), "ads_binned": 0, "ads_to_clip_end": 0, "no_binned_ad": 0}
        ids = []
        for rec, mt in zip(data, meta):
            n, w, n_end = rf["convert_clip_end"](rec, mt["video_id"], budget, mt["content_end"])
            st["ads_binned"] += n; st["ads_to_clip_end"] += n_end
            if n == 0:
                st["no_binned_ad"] += 1
                if split == "train":
                    continue
            rf["mk"]["cap"](rec)
            if len(rec["images"]) > a.max_frames:
                rf["even_cap"](rec, a.max_frames)
            out.append(rec); ids.append(mt["video_id"])
        est = np.array([rf["mk"]["est"](r) for r in out])
        keep = [i for i, x in enumerate(est) if split != "train" or x <= a.budget]
        name = f"mtg_bins_f{a.max_frames}_{split}"
        json.dump([out[i] for i in keep], open(f"{a.dir}/{name}.json", "w"), ensure_ascii=False)
        json.dump([ids[i] for i in keep], open(f"{a.dir}/{name}_ids.json", "w"))
        info[name] = dict(info[f"cmdad_dds_gapframes_{split}"], file_name=f"{name}.json")
        st.update(clips=len(keep), dropped_over_budget=len(out) - len(keep),
                  frames_max=int(max(len(out[i]["images"]) for i in keep)),
                  tokens_est_max=int(est[keep].max()), tokens_est_median=int(np.median(est[keep])))
        print(name, st)
    json.dump(info, open(f"{a.dir}/dataset_info.json", "w"), indent=2)


if __name__ == "__main__":
    main()
