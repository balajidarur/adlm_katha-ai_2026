# usage: slimlist.py clips.txt > files.txt   (paths relative to /prep/prep_360p: only what the Qwen stage reads)
import json, os, sys
R = "/prep/prep_360p"
crops = set()
for c in open(sys.argv[1]):
    c = c.strip()
    if not c or not os.path.exists(f"{R}/{c}/meta.json"):
        continue
    print(f"{c}/meta.json"); print(f"{c}/faces.json")
    m = json.load(open(f"{R}/{c}/meta.json"))
    for s in m["shots"]:
        for f in s["frames"]:
            print(f"{c}/circled/{f['file']}")
    for x in json.load(open(f"{R}/{c}/faces.json"))["cast"]:
        if x.get("portrait_crop"):
            crops.add(x["portrait_crop"])
for p in sorted(crops):
    print(p)
