#!/usr/bin/env python3
"""Per-system CSVs for LLM-AD-Eval and R@k/N on the CMD-AD eval split (7,316 GT ADs, 591 clips).
Input: the Action-score input CSVs (anno_idx = row of AutoAD-Zero's public prediction CSV = official 7,316-AD order;
text_gt identical across systems). Adds cmd_filename / start / end from that public CSV for ordering.
  python make_e1x_csvs.py <public_ref_csv> <out_dir> name=action_input.csv [name=...]
"""
import os, sys

import pandas as pd

ref = pd.read_csv(sys.argv[1], keep_default_na=False); assert len(ref) == 7316
out = sys.argv[2]; os.makedirs(out, exist_ok=True)
for spec in sys.argv[3:]:
    name, path = spec.split("=", 1)
    d = pd.read_csv(path, keep_default_na=False)
    assert len(d) == 7316 and (d.anno_idx.values == range(7316)).all() and (d.text_gt == ref.text_gt).all(), name
    assert (d.text_gen.astype(str).str.strip() != "").all(), f"{name}: empty predictions"
    pd.DataFrame({"anno_idx": d.anno_idx, "imdbid": ref.vid, "cmd_filename": ref.cmd_filename, "start": ref.start, "end": ref.end,
                  "text_gt": d.text_gt, "text_gen": d.text_gen}).to_csv(f"{out}/{name}.csv", index=False)
    print(name, len(d))
