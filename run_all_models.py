"""Run ADLM pipeline on multiple videos × multiple models — ALL IN PARALLEL."""

import os
import sys
import json
import time
import subprocess
import re
import threading
import filelock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEOS = [
    {
        "path": "/home/dbalu/workspace/ad_gen/visual_cues_per_shot/8fyjCygRaPc/8fyjCygRaPc_3min.mp4",
        "srt": "/home/dbalu/workspace/ad_gen/demo/v2_voice_transcribe_idea/english/three_minute_srts/8fyjCygRaPc_part01.srt",
    },
    {
        "path": "/home/dbalu/workspace/ad_gen/visual_cues_per_shot/bc-TQzKoPQo/bc-TQzKoPQo_3min.mp4",
        "srt": "",
    },
]

MODELS = [
    {"name": "gemini-3.5-flash-lite", "provider": "api", "temp": "0.2"},
    {"name": "gemma-4-31b-it",        "provider": "api", "temp": "0.2"},
    {"name": "qwen.qwen3-vl-235b-a22b", "provider": "bedrock", "temp": "0.2", "region": "us-east-1"},
    {"name": "gemini-3.6-flash",      "provider": "api", "temp": "0.2"},
]

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "output", "run_progress.json")
LOCK_FILE = PROGRESS_FILE + ".lock"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

progress_lock = threading.Lock()
progress = {}


def get_video_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def count_shots(video_path):
    from test_adlm_descriptions import detect_shots
    shots = detect_shots(video_path, max_shot_duration=10.0)
    return len(shots)


def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def update_run(key, **kwargs):
    with progress_lock:
        progress["runs"][key].update(kwargs)
        save_progress()


def run_single(video, model):
    video_id = get_video_id(video["path"])
    model_name = model["name"]
    key = f"{video_id}/{model_name}"

    update_run(key, status="running", started_at=time.time())

    cmd = [
        sys.executable, "-u",
        os.path.join(SCRIPT_DIR, "test_adlm_descriptions.py"),
        f"model.provider={model['provider']}",
        f"model.name={model_name}",
        f"model.temperature={model['temp']}",
        f"video.path={video['path']}",
        f"video.dialogue_srt={video['srt']}",
        "pipeline.num_shots=0",
        "output.dir=output",
    ]
    if "region" in model:
        cmd.append(f"model.region={model['region']}")

    env = os.environ.copy()
    if model["provider"] == "bedrock":
        env["AWS_PROFILE"] = "PowerUserAccess-933516006154"

    start = time.time()
    shots_done = 0
    events_total = 0

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=SCRIPT_DIR, env=env,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("--- Shot"):
                shots_done += 1
                update_run(key, shots_done=shots_done,
                           elapsed_s=round(time.time() - start, 1))
            elif line.strip().startswith("Events:"):
                m = re.match(r'\s*Events:\s*(\d+)', line)
                if m:
                    events_total += int(m.group(1))
                    update_run(key, events=events_total)
            elif "ERROR:" in line:
                update_run(key, last_error=line.strip()[:80])

        proc.wait(timeout=3600)
        elapsed = time.time() - start

        status = "done" if proc.returncode == 0 else "error"
        update_run(key, status=status, elapsed_s=round(elapsed, 1),
                   shots_done=shots_done, events=events_total)

    except subprocess.TimeoutExpired:
        proc.kill()
        update_run(key, status="timeout", elapsed_s=3600)
    except Exception as e:
        update_run(key, status="error", last_error=str(e)[:80])

    with progress_lock:
        progress["completed"] += 1
        save_progress()

    print(f"[{progress['completed']}/{progress['total']}] {key}: "
          f"{progress['runs'][key]['status']} | {shots_done} shots | "
          f"{events_total} events | {round(time.time()-start)}s", flush=True)


def main():
    global progress

    print("Counting shots for each video...", flush=True)
    shot_counts = {}
    for v in VIDEOS:
        vid = get_video_id(v["path"])
        n = count_shots(v["path"])
        shot_counts[vid] = n
        print(f"  {vid}: {n} shots", flush=True)

    total_runs = len(VIDEOS) * len(MODELS)
    total_shots = sum(shot_counts.values()) * len(MODELS)

    progress = {
        "total": total_runs,
        "completed": 0,
        "total_shots_all": total_shots,
        "shot_counts": shot_counts,
        "started_at": time.time(),
        "runs": {},
    }

    for model in MODELS:
        for video in VIDEOS:
            vid = get_video_id(video["path"])
            key = f"{vid}/{model['name']}"
            progress["runs"][key] = {
                "status": "pending",
                "video": vid,
                "model": model["name"],
                "shots_done": 0,
                "events": 0,
                "total_shots": shot_counts.get(vid, 0),
                "elapsed_s": 0,
            }

    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    save_progress()

    print(f"\n{total_runs} runs IN PARALLEL, {total_shots} total shots", flush=True)
    print(f"Videos: {list(shot_counts.keys())}", flush=True)
    print(f"Models: {[m['name'] for m in MODELS]}\n", flush=True)

    threads = []
    for model in MODELS:
        for video in VIDEOS:
            t = threading.Thread(target=run_single, args=(video, model),
                                 name=f"{get_video_id(video['path'])}/{model['name']}")
            threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    progress["finished_at"] = time.time()
    total_elapsed = progress["finished_at"] - progress["started_at"]
    save_progress()

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
    print(f"{'='*60}", flush=True)
    for key, info in progress["runs"].items():
        print(f"  {key}: {info['status']} | {info.get('elapsed_s','?')}s | "
              f"{info.get('shots_done','?')}/{info.get('total_shots','?')} shots | "
              f"{info.get('events','?')} events", flush=True)


if __name__ == "__main__":
    main()
