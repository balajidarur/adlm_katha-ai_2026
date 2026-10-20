# CMD-AD pre-VLM prep (character bank), 2026-10-17
Made by ~/prep_code/prep_charbank.py on box A (4x L40S): 8,599 clips, 3,178,925 frames, ~85 min total.

prep_360p/{year}/{id}/
  meta.json   shots (Gemini shot cache) + every frame's file and time (3 FPS from shot start;
              context frames = every 3rd = 1 FPS), letterbox crop, decoder used
  frames/     360p, letterbox cropped (mostly 640x268 / 640x346; bar-free sources ~872x360)
  circled/    same frames with closed coloured ellipses on recognised cast faces
              (hard links where no circle). No timestamp: burn it at request time.
  faces.npz   every face: frame_idx, bbox, kps, det_score, 512-d ArcFace embedding (fp16)
  faces.json  top-10 cast (colours, portrait crop path) + per-frame matches (raw and smoothed)
prep_360p/_portraits/{actor}.npz|.jpg   portrait embeddings + downloaded headshots
prep_360p/_crops/{actor}.jpg            portrait face crops (224 px), go into the prompt
failed_clips.txt                        clips whose video file has no readable video stream (skipped)

Use: charbank_pipeline.py --prep-root /prep/prep_360p --video {year}/{id}
Logs: prep_full_part1.log + prep_full.log, progress_part1.jsonl + progress.jsonl, dmon_full.log
