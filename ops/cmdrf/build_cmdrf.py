"""CMD-AD + REFRAMED combined budget training set: M3's CMD-AD bins_f150_40k (Qwen3.8 dense descriptions, 0.5 s bins) and
REFRAMED's reframed_bins_f150_40k_train (same build), concatenated and shuffled (seed 0) so both datasets mix inside every
epoch. Rows are copied unchanged (image paths are absolute). Writes <out>/cmdrf_bins_f150_40k.json + dataset_info.json.
  python3 build_cmdrf.py CMDAD_JSON REFRAMED_JSON OUT_DIR"""
import json, os, random, sys
cmd_f, rf_f, out = sys.argv[1:4]
cmd, rf = json.load(open(cmd_f)), json.load(open(rf_f))
def head(r):   # prompt header up to the clip-specific part: must match so both use the same instructions/tokens
    return r['messages'][0]['content'][:1500]
print('PROMPT_HEADER_MATCH', head(cmd[0]) == head(rf[0]))
for name, d in (('cmdad', cmd), ('reframed', rf)):
    roles = {tuple(m['role'] for m in r['messages']) for r in d}
    print(f'{name}: {len(d)} rows, {sum(len(r["images"]) for r in d):,} frames, role patterns {roles}')
rows = cmd + rf
random.Random(0).shuffle(rows)
os.makedirs(out, exist_ok=True)
json.dump(rows, open(f'{out}/cmdrf_bins_f150_40k.json', 'w'))
info = {'cmdrf_bins_f150_40k': {'file_name': 'cmdrf_bins_f150_40k.json', 'formatting': 'sharegpt',
        'columns': {'messages': 'messages', 'images': 'images'},
        'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}}}
json.dump(info, open(f'{out}/dataset_info.json', 'w'), indent=2)
first = ['rf' if '/rf_frames/' in r['images'][0] else 'cmd' for r in rows[:20]]
print(f'combined: {len(rows)} rows; first 20 sources {first}')
