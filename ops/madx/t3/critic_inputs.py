"""CRITIC on MAD-Eval with the official AutoAD-III critic_metric.py, unmodified: only its two data files are extended.
  cast_list_for_eval.json += {imdbid: top-10 IMDb character names} from the community MAD-Eval character bank
                             (AutoAD-Zero resources/charbanks/madeval_charbank.pkl, charbank_roles; same top-10 rule as CMD-AD)
  cmd_fn_to_imdb.json     += {imdbid: imdbid}   (our 'vid' column is the movie's IMDb id)
Input rows: madeval_anno order (temporal within each movie), vid = imdbid.
  python critic_inputs.py --metrics DIR(official metrics dir copy) --roles madeval_charbank_roles.json --anno madeval_anno.csv \
      --preds preds_tf.csv --out critic_input.csv
"""
import argparse, csv, json

ap = argparse.ArgumentParser()
for k in ("metrics", "roles", "anno", "preds", "out"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
raw = open(a.anno, "rb").read()
try:
    txt = raw.decode("utf-8")
except UnicodeDecodeError:
    txt = raw.decode("cp1252")
anno = list(csv.DictReader(txt.splitlines()))
movie2imdb = {r["movie"]: r["imdb"] for r in anno}
roles = json.load(open(a.roles))
cast = json.load(open(f"{a.metrics}/cast_list_for_eval.json"))
fn = json.load(open(f"{a.metrics}/cmd_fn_to_imdb.json"))
for mv, rl in roles.items():
    cast[movie2imdb[mv]] = rl
    fn[movie2imdb[mv]] = movie2imdb[mv]
json.dump(cast, open(f"{a.metrics}/cast_list_for_eval.json", "w"))
json.dump(fn, open(f"{a.metrics}/cmd_fn_to_imdb.json", "w"))
rows = list(csv.DictReader(open(a.preds, newline="")))
assert [int(r["anno_idx"]) for r in rows] == list(range(len(anno)))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["vid", "text_gt", "text_gen"])
    for r in rows:
        w.writerow([r["imdbid"], r["text_gt"], r["text_gen"]])
print(len(rows), "rows;", len(roles), "movies added to the cast list")
