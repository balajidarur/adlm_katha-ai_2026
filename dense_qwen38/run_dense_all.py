"""
Run charbank_pipeline.py over many clips against several vLLM servers (one per GPU).

Each clip is sequential inside (shot N sees shot N-1), so throughput comes from running
--per-server clips at once on every server. Finished clips (adlm_descriptions.json) are skipped.

    python run_dense_all.py --clips clips_all.txt --prep-root /prep/prep_360p \
        --out-root /prep/qwen_dense --ports 8000 8001 8002 8003 --per-server 8
"""
import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--prep-root", required=True)
ap.add_argument("--out-root", required=True)
ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
ap.add_argument("--ports", type=int, nargs="+", default=[8000])
ap.add_argument("--per-server", type=int, default=8)
ap.add_argument("--prompt-variant", default="base")
ap.add_argument("--no-prop-loc-tags", action="store_true")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
tag = args.model.split("/")[-1]
clips = [c.strip() for c in open(args.clips) if c.strip()]
todo = [c for c in clips if os.path.exists(os.path.join(args.prep_root, c, "meta.json"))
        and not os.path.exists(os.path.join(args.out_root, c.split("/")[-1], tag, "adlm_descriptions.json"))]
print(f"{len(clips)} clips, {len(todo)} to do", flush=True)
os.makedirs(os.path.join(args.out_root, "_logs"), exist_ok=True)
log = open(os.path.join(args.out_root, "run_log.jsonl"), "a")
lock = threading.Lock()
q = queue.Queue()
for c in todo:
    q.put(c)


def worker(port):
    while True:
        try:
            c = q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        lp = os.path.join(args.out_root, "_logs", c.replace("/", "_") + ".log")
        with open(lp, "w") as lf:
            rc = subprocess.call([sys.executable, os.path.join(here, "charbank_pipeline.py"), "--video", c,
                                  "--server", f"http://localhost:{port}/v1", "--model", args.model,
                                  "--prep-root", args.prep_root, "--out-root", args.out_root, "--no-save-requests",
                                  "--prompt-variant", args.prompt_variant] + (["--no-prop-loc-tags"] if args.no_prop_loc_tags else []),
                                 stdout=lf, stderr=subprocess.STDOUT)
        with lock:
            log.write(json.dumps({"clip": c, "port": port, "rc": rc, "wall_s": round(time.time() - t0, 1),
                                  "end": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            log.flush()


threads = [threading.Thread(target=worker, args=(p,)) for p, _ in itertools.product(args.ports, range(args.per_server))]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("ALL_DONE", flush=True)
