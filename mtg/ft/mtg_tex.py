"""LaTeX table (paper tab:mtg) of MTG test metrics: mean over videos per official aggregate.py rules.
  python mtg_tex.py > tables/mtg.tex"""
import glob, os, sys, types
sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
sys.path.insert(0, "/workspace/dbalu/tasks_nov8/T4/mtg-ad")
import matplotlib; matplotlib.use("Agg")
import numpy as np
from analysis import aggregate as A
E = "/storage/users/dbalu/mtg_eval/sys"
ROWS = [("human_gt", r"\textcolor{gray}{Human AD}", "gt"),
        ("gemini25_timing_asr", "Gemini 2.5 Flash", "b"), ("gemini35_timing_asr", "Gemini 3.5 Flash", "b"),
        ("qwen_vl_max", "Qwen-VL-Max", "b"), ("qwen3p5_omni_flash", "Qwen 3.5 Omni Flash", "b"), ("qwen3p5_omni_plus", "Qwen 3.5 Omni Plus", "b"),
        ("gpt_4_1_mini", "GPT-4.1 mini", "b"), ("gpt_5_2", "GPT-5.2", "b"), ("claude_4", "Claude 4", "b"), ("claude_4_5", "Claude 4.5", "b"),
        ("zs440", "Ours, zero-shot (CMD-AD)", "o"), ("ft102", "Ours, fine-tuned on MTG", "o")]
COLS = [("metric_G1.1_precision_avg_ad_in_visual_info", "up", 2), ("metric_2_1_does_not_restate", "up", 2),
        ("metric_1_2_narrates_onscreen_text", "up", 2), ("metric_g_2_avg_diff_time", "down", 2),
        ("total_overlap", "down", 2), ("ad_total_speaking_time_per_silent_second", "up_to_1", 2),
        ("avg_silence_duration_w_ad_tts", "down", 2), ("videoa11y_descriptive", "up", 2), ("videoa11y_objective", "up", 2),
        ("videoa11y_accurate", "up", 2), ("videoa11y_clear", "up", 2)]
def load(s):
    fs = glob.glob(f"{E}/{s}_test/eval/metrics/output_*.json")
    return [A.load_video_metrics(p) for p in fs] if len(fs) >= 90 else None
data = {s: load(s) for s, _, _ in ROWS}
rows = [(s, n, k) for s, n, k in ROWS if data[s]]
mean = {(s, c): np.mean([r[c] for r in data[s] if r and c in r]) for s, _, _ in rows for c, _, _ in COLS}
best = {}
for c, d, _ in COLS:   # best among models (not human)
    v = {s: mean[(s, c)] for s, _, k in rows if k != "gt"}
    best[c] = (max(v, key=v.get) if d == "up" else min(v, key=v.get) if d == "down" else min(v, key=lambda s: abs(1 - v[s])))
print(r"""% MTG (Mind the Gap, Venugopalan et al. 2026) test split, 97 available of 101 videos, end-to-end (models predict AD times + text).
% All rows scored by us with the official run_ad_eval.py + aggregate.py rules (mean over videos; capped values dropped); AD length from
%   Piper en_US-lessac-medium at 1.2x (paper: Cloud TTS en-US-Neural2-C 1.2x; Piper ~5% shorter); LLM judge = repo default gemini-3.8-flash
%   (scores higher than the paper's on VideoA11y, so rows are only comparable within this table). Baselines = the paper's released outputs.
% Ours: Qwen3.8 DDs + Qwen3.5-4B writer with budget tokens. Zero-shot = CMD-AD M3 ckpt-440; on 26/97 videos it emits no AD (means over 71).
%   Fine-tuned = from ckpt-440 on 279 MTG validation videos, 3 epochs (ckpt-102 picked on a 34-video val split); ADs on 96/97 videos.
% Source: /workspace/dbalu/tasks_nov8/T5_MTG_RESULTS.md, code mtg/ft/ (mtg_score.sh, mtg_tex.py).
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Content} & \multicolumn{4}{c}{Timing} & \multicolumn{4}{c}{VideoA11y-Ref} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(l){9-12}
Method & Prior.$\uparrow$ & No Red.$\uparrow$ & OCR$\uparrow$ & Timel.\,(s)$\downarrow$ & Ovl.\,(s)$\downarrow$ & Cov.\,($\le$1) & Sil.\,(s)$\downarrow$ & Desc.$\uparrow$ & Obj.$\uparrow$ & Acc.$\uparrow$ & Clear$\uparrow$ \\
\midrule""")
prev = None
for s, n, k in rows:
    if prev and k != prev:
        print(r"\midrule")
    prev = k
    cells = []
    for c, d, p in COLS:
        x = f"{mean[(s, c)]:.{p}f}"
        if k == "gt":
            x = r"\textcolor{gray}{" + x + "}"
        elif best[c] == s:
            x = r"\textbf{" + x + "}"
        cells.append(x)
    print(f"{n} & " + " & ".join(cells) + r" \\")
print(r"""\bottomrule
\end{tabular}
\caption{\textbf{Mind the Gap test set} (97 YouTube videos), end-to-end inline AD with the official metrics: prioritization, no redundancy and on-screen text (Gemini judge), content timeliness, overlap with dialogue, audio coverage and average silence (ADs rendered with TTS at 1.2$\times$), and reference-based VideoA11y scores (1--5). All rows are scored by us with the same pipeline from the released outputs of~\cite{venugopalan2026mindgap}. Zero-shot: our CMD-AD model; fine-tuned: further trained on MTG validation videos. The zero-shot model produces no ADs for 26 of the 97 videos; its means are over the other 71.}
\label{tab:mtg}
\end{table*}""")
