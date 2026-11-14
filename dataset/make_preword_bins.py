#!/usr/bin/env python3
"""Pre-word budget labels: rewrite ONLY the AD budget tokens of an existing M3-style JSON.

M3 (add_bins_to_gapframes.py convert()) put before word k the bin of the time left AFTER word k is spoken:
    old token_k = bin(gap - prefix_duration_s[k])          (k = 1..n)
The legend says the token is the silence that remains before the next dialogue, i.e. at the moment the word starts:
    new token_k = bin(gap - prefix_duration_s[k-1])        (prefix_duration_s[0] = 0, so token_1 = bin(gap))

Method, per AD line that carries budget tokens:
  1. look the AD up in the parquet exactly as convert() does (speed 0.67-1.5, nearest scaled_start within 0.07 s,
     parquet words or, on a text mismatch, the line's words if the counts agree; gap = next [DIALOGUE] start - AD start)
  2. rebuild the line with the ORIGINAL rule and assert it equals the existing line byte for byte
  3. write the line with the pre-word rule
AD lines without tokens are left alone (and we assert convert() would not have binned them either).
Nothing else in a record changes: prompt (legend included), images, dialogue lines, AD words, record order.

  python make_preword_bins.py --src gapframes_bins_train_f150_40k.json --out qwen38_bins_preword_f150_40k.json \
      [--stats stats.json] [--examples N]
The output keeps the source's JSON encoding (ensure_ascii is chosen so the source round-trips byte-identically).
"""
import argparse, collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ab = {}
exec(open(f"{HERE}/add_bins_to_gapframes.py").read().split('if __name__ == "__main__":')[0], ab)
BIN_TOKENS, bin_of, parse_ts, AD_RE, DLG_RE = ab["BIN_TOKENS"], ab["bin_of"], ab["parse_ts"], ab["AD_RE"], ab["DLG_RE"]
TOK = re.compile(r"<\|budget_(\d+)\|>")
FRAME_ROOT = re.compile(r"/1fps_448x252/([^/]+)/")


def load_budget(path):
    """Same filter/sort as add_bins_to_gapframes.load_budget, but keeps the full prefix list (incl. [0] = 0)."""
    import pandas as pd
    df = pd.read_parquet(path)
    df = df[df.speed_factor.between(ab["SPEED_LO"], ab["SPEED_HI"])]
    out = {}
    for r in df.itertuples(index=False):
        vid = r.cmd_filename.split("/")[-1]
        pref = [float(x) for x in r.prefix_duration_s]
        assert pref[0] == 0.0 and len(pref) == len(r.words) + 1
        out.setdefault(vid, []).append((float(r.scaled_start), list(r.words), pref))
    for v in out:
        out[v].sort(key=lambda x: (x[0], x[1], x[2]))   # identical order to convert()'s tuple sort
    return out


def lookup(budget, vid, t, tol=0.07):
    """add_bins_to_gapframes.lookup, returning the full prefix list."""
    rows = budget.get(vid)
    if not rows:
        return None
    best, bd = None, tol
    for st, words, pref in rows:
        d = abs(st - t)
        if d <= bd:
            best, bd = (words, pref), d
    return best


def label(line, vid, dlg_starts, budget):
    """-> (old_line, new_line, old_bins, new_bins) exactly as convert() would build it, or None if convert() skips."""
    m = AD_RE.match(line)
    t, text = parse_ts(m.group(1)), m.group(2)
    ent = lookup(budget, vid, t)
    if ent is None:
        return None
    words, pref = ent
    if " ".join(words).strip() != text.strip():
        words = text.split()
        if len(words) != len(pref) - 1:
            return None
    nxt = [a for a in dlg_starts if a >= t]
    if not nxt:
        return None
    gap = nxt[0] - t
    ob = [bin_of(gap - p) for p in pref[1:]]          # original: after word k
    nb = [bin_of(gap - p) for p in pref[:-1]]         # pre-word: before word k
    ob, nb = ob[:len(words)], nb[:len(words)]         # zip() truncation in convert()
    old = "".join(f"{BIN_TOKENS[b]}{w} " for w, b in zip(words, ob)).strip()
    new = "".join(f"{BIN_TOKENS[b]}{w} " for w, b in zip(words, nb)).strip()
    return f"{m.group(1)} [AD] {old}", f"{m.group(1)} [AD] {new}", ob, nb


def vid_of(rec):
    vs = {m.group(1) for i in rec["images"] for m in [FRAME_ROOT.search(i)] if m}
    return vs.pop() if len(vs) == 1 else None


def infer_vid(rec, budget):
    """Records without frames: the clip whose parquet ADs match every tokened AD line (must be unique)."""
    lines = rec["messages"][1]["content"].split("\n")
    dl = sorted(parse_ts(m.group(1)) for m in (DLG_RE.match(l) for l in lines) if m)
    ads = [l for l in lines if AD_RE.match(l) and TOK.search(l)]
    cands = [v for v in budget if all((r := label(f"{AD_RE.match(l).group(1)} [AD] {TOK.sub('', l.split(' [AD] ', 1)[1])}",
                                                     v, dl, budget)) and r[0] == l for l in ads)]
    assert len(cands) == 1, cands
    return cands[0]


def convert(rec, budget, st, examples):
    u0, imgs0 = rec["messages"][0]["content"], list(rec["images"])
    vid = vid_of(rec) or infer_vid(rec, budget)
    lines = rec["messages"][1]["content"].split("\n")
    dlg_starts = sorted(parse_ts(m.group(1)) for m in (DLG_RE.match(l) for l in lines) if m)
    out = []
    for line in lines:
        m = AD_RE.match(line)
        if not m:
            out.append(line)
            continue
        body = m.group(2)
        if not TOK.search(body):
            st["ad_no_tokens"] += 1
            plain = label(line, vid, dlg_starts, budget)
            assert plain is None, (vid, line)      # convert() would have binned it -> source isn't what we think
            out.append(line)
            continue
        st["ad_with_tokens"] += 1
        text = TOK.sub("", body)
        r = label(f"{m.group(1)} [AD] {text}", vid, dlg_starts, budget)
        if r is None or r[0] != line:
            st["ad_unreproduced"] += 1
            st["unreproduced"].append((vid, line))
            out.append(line)
            continue
        old, new, ob, nb = r
        assert [int(x) for x in TOK.findall(line)] == ob
        assert TOK.sub("", new) == TOK.sub("", line)
        for a, b in zip(ob, nb):
            st["shift"][b - a] += 1
            st["tok_old"][a] += 1
            st["tok_new"][b] += 1
        st["tok_changed"] += sum(a != b for a, b in zip(ob, nb))
        st["tok_total"] += len(ob)
        st["ad_changed"] += old != new
        st["first_b8_old"] += ob[0] == 8
        st["first_b8_new"] += nb[0] == 8
        if len(examples) < 400:
            examples.append({"vid": vid, "old": old, "new": new})
        out.append(new)
    rec["messages"][1]["content"] = "\n".join(out)
    assert rec["messages"][0]["content"] == u0 and rec["images"] == imgs0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="")
    ap.add_argument("--parquet", default=os.environ.get("BUDGET_PARQUET", ab["BUDGET_PARQUET"]))
    ap.add_argument("--max-unreproduced", type=float, default=0.001, help="max share of tokened ADs allowed unreproduced")
    a = ap.parse_args()

    raw = open(a.src, "rb").read()
    data = json.loads(raw)
    ea = next((e for e in (False, True) if json.dumps(data, ensure_ascii=e).encode() == raw), None)
    assert ea is not None, "source does not round-trip through json.dumps; cannot guarantee byte-identity"
    src = json.loads(raw)
    budget = load_budget(a.parquet)
    st = {k: 0 for k in ("ad_no_tokens", "ad_with_tokens", "ad_unreproduced", "ad_changed",
                         "tok_changed", "tok_total", "first_b8_old", "first_b8_new")}
    st.update(shift=collections.Counter(), tok_old=collections.Counter(), tok_new=collections.Counter(), unreproduced=[])
    ex = []
    out = [convert(r, budget, st, ex) for r in data]

    # record-level check: everything but the AD budget tokens is identical to the source
    assert len(out) == len(src)
    for x, y in zip(out, src):
        assert x["images"] == y["images"] and x["messages"][0] == y["messages"][0]
        assert len(x["messages"]) == len(y["messages"]) == 2 and x["messages"][1]["role"] == y["messages"][1]["role"]
        assert TOK.sub("", x["messages"][1]["content"]) == TOK.sub("", y["messages"][1]["content"])
        lx, ly = x["messages"][1]["content"].split("\n"), y["messages"][1]["content"].split("\n")
        assert len(lx) == len(ly)
        for p, q in zip(lx, ly):
            if not (AD_RE.match(q) and TOK.search(q)):
                assert p == q
            else:
                assert len(TOK.findall(p)) == len(TOK.findall(q))
    blob = json.dumps(out, ensure_ascii=ea).encode()
    open(a.out, "wb").write(blob)

    n = st["ad_with_tokens"]
    frac = st["ad_unreproduced"] / max(n, 1)
    rep = {"src": a.src, "out": a.out, "ensure_ascii": ea, "records": len(out),
           "ad_with_tokens": n, "ad_without_tokens": st["ad_no_tokens"], "ad_unreproduced": st["ad_unreproduced"],
           "ad_changed": st["ad_changed"], "tokens_total": st["tok_total"], "tokens_changed": st["tok_changed"],
           "tokens_unchanged": st["tok_total"] - st["tok_changed"],
           "shift_new_minus_old": dict(sorted(st["shift"].items())),
           "first_token_b8_old": st["first_b8_old"], "first_token_b8_new": st["first_b8_new"],
           "tok_hist_old": dict(sorted(st["tok_old"].items())), "tok_hist_new": dict(sorted(st["tok_new"].items())),
           "unreproduced_examples": st["unreproduced"][:20], "examples": ex[:400]}
    if a.stats:
        json.dump(rep, open(a.stats, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("examples", "unreproduced_examples")}, indent=1))
    if frac > a.max_unreproduced:
        sys.exit(f"STOP: {st['ad_unreproduced']} / {n} tokened ADs ({frac:.4%}) not reproducible with the original rule")


if __name__ == "__main__":
    main()
