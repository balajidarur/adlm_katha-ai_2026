"""Dialogue SRTs for REFRAMED validation/test, in the layout of reframed_train/srt/dialogue.

Applies steps 1-2 of the train cleaning (reframed_train/README.md section 4):
  1. "- Go.\n- No." -> one entry per speaker, cut by character count, first part ends 0.04 s before
  2. line breaks -> spaces
Step 3 (pulling subtitle ends back to the next AD start) is NOT applied: it uses ground-truth AD
times, which must not leak into val/test inputs.

Out: {out}/dialogue/{movie}/{excerpt}.srt   (train SRTs are symlinked in, so one dir covers all splits)
"""

import argparse
import glob
import json
import os
import re

R = "/storage/datasets/Reframed"
TRAIN_SRT = "/workspace/.common_cache/common_share_for_users/srt_for_ads/budgetAD/reframed_train/srt/dialogue"


def ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def entries(path):
    out = []
    for e in sorted(json.load(open(path)).values(), key=lambda x: x["start"]):
        parts = [p for p in re.split(r"\n(?=-)", e["text"].strip()) if p.strip()]
        if len(parts) > 1 and all(p.lstrip().startswith("-") for p in parts):
            texts = [re.sub(r"^-\s*", "", p.strip()).replace("\n", " ") for p in parts]
            tot, t = sum(len(x) for x in texts), e["start"]
            for i, x in enumerate(texts):
                end = e["end"] if i == len(texts) - 1 else t + (e["end"] - e["start"]) * len(x) / tot
                out.append((t, end - (0.04 if i < len(texts) - 1 else 0), x))
                t = end
        else:
            out.append((e["start"], e["end"], e["text"].replace("\n", " ").strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/dbalu/reframed/srt")
    args = ap.parse_args()
    n = {"train": 0, "validation": 0, "test": 0}
    for p in glob.glob(f"{TRAIN_SRT}/*/*.srt"):
        dst = f"{args.out}/dialogue/{p.split('/')[-2]}/{os.path.basename(p)}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(p, dst)
        n["train"] += 1
    for split in ("validation", "test"):
        for v in glob.glob(f"{R}/{split}/excerpt/video/*/*.mp4"):
            movie, exc = v.split("/")[-2], os.path.basename(v)[:-4]
            src = f"{R}/{split}/excerpt/subtitles/{movie}/{exc}.json"
            ents = entries(src) if os.path.exists(src) else []
            dst = f"{args.out}/dialogue/{movie}/{exc}.srt"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write("".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(ents, 1)))
            n[split] += 1
    print(n)


if __name__ == "__main__":
    main()
