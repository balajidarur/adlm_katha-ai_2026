"""MTG baseline generation with open-weight VLMs served by vLLM (OpenAI-compatible): the paper's prompt (default timing_asr,
"Gd. + Timing ASR", the setting of its model tables) with the same transcript file, the whole video as one video_url
(vLLM samples it at --fps, max --max-frames; Qwen-VL gets frame timestamps). Output: OUT/output_{vid}.txt (raw response;
run_ad_eval parses it like the released baselines).
  python open_gen.py --ids ids.json --videos DIR --transcripts DIR --out OUT --ports 8000,8001 --model NAME [--workers 16]"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import openai
sys.path.insert(0, os.environ.get("MTG_REPO", "/scratch/mtg-ad"))
from generation import gemini_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True); ap.add_argument("--videos", required=True); ap.add_argument("--transcripts", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--ports", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--prompt", default="timing_asr"); ap.add_argument("--fps", type=float, default=1.0)
ap.add_argument("--max-frames", type=int, default=256); ap.add_argument("--max-tokens", type=int, default=8192)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
P = gemini_prompt.PROMPTS[a.prompt]
clients = [openai.OpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=3600) for p in a.ports.split(",")]
lock = threading.Lock(); k = [0]


def run(vid):
    o = f"{a.out}/output_{vid}.txt"
    if os.path.exists(o) and os.path.getsize(o):
        return vid, "skip"
    tr = open(f"{a.transcripts}/output_{vid}.txt").read()
    with lock:
        c = clients[k[0] % len(clients)]; k[0] += 1
    try:
        r = c.chat.completions.create(model=a.model, max_tokens=a.max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{a.videos}/{vid}.mp4"}},
                {"type": "text", "text": P.replace("ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE", tr)}]}],
            extra_body={"mm_processor_kwargs": {"fps": a.fps, "max_frames": a.max_frames},
                        "chat_template_kwargs": {"enable_thinking": False}})
        open(o, "w").write(r.choices[0].message.content or "")
        return vid, "ok"
    except Exception as e:
        return vid, f"ERR {type(e).__name__}: {str(e)[:200]}"


ids = json.load(open(a.ids))
with ThreadPoolExecutor(a.workers) as ex:
    for vid, st in ex.map(run, ids):
        print(vid, st, flush=True)
