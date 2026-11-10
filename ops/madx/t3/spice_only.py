"""SPICE (pycocoevalcap) for a preds CSV with text_gt / text_gen (one row per GT AD). Usage: spice_only.py CSV OUT.json"""
import json, sys

import pandas as pd
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

df = pd.read_csv(sys.argv[1], keep_default_na=False)
tk = PTBTokenizer()
gts = tk.tokenize({str(i): [{"caption": t}] for i, t in enumerate(df.text_gt)})
res = tk.tokenize({str(i): [{"caption": t}] for i, t in enumerate(df.text_gen)})
s, _ = Spice().compute_score(gts, res)
json.dump({"SPICE": round(100 * s, 2), "n": len(df), "csv": sys.argv[1]}, open(sys.argv[2], "w"))
print("SPICE", round(100 * s, 2))
