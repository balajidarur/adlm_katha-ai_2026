# M6 on the full eval591 needs Qwen3.5-4B DDs for the 21 eval591 clips that are not in the 4B run's chunks (16 from prep_hf, 5 from
# prep_360p_t2), the same 21 the 9B run added (dd_inputs/full9b_work/dd21_9b.sh). Same settings as the full 4B run
# (run_4b_full.sh / serve_qwen35_4b.sh: desc_action_v2, default sampling, max-model-len 49152, 110 images). Run on a box from the
# 9B DD run (has /scratch/vllm, /scratch/code_f9/prep_code, /scratch/asr). Output -> s3 dd_inputs/full4b/out21/.
#   bash dd21_4b.sh     (expects ~/miss21.txt)
set -x
. /scratch/cr.env 2>/dev/null || . $HOME/.cr.env; export PATH=$HOME/.local/bin:/usr/local/bin:$PATH; R=s3://balu-transfer-933516006154-apne1; E=$R/cmd_ad_eval591; W=/scratch
for p in $(pgrep -f "vllm serve"); do kill $p; done; sleep 5
[ -f /scratch/Qwen3.5-4B/config.json ] || /scratch/vllm/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='/scratch/Qwen3.5-4B')"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup /scratch/vllm/bin/vllm serve /scratch/Qwen3.5-4B --served-model-name Qwen/Qwen3.5-4B --port 8300 --max-model-len 49152 --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 110, "video": 0}' --max-num-seqs 32 --max-num-batched-tokens 16384 \
  --enable-prefix-caching > /scratch/vllm4b.log 2>&1 < /dev/null &
until curl -s localhost:8300/v1/models | grep -q 4B; do sleep 10; done; echo SERVER_READY
mkdir -p $W/e21 && cd $W/e21 && { [ -d prep_hf ] || aws s3 cp --only-show-errors $E/prep_hf.tar - | tar xf -; } && { [ -d prep_360p ] || aws s3 cp --only-show-errors $E/prep_360p_t2.tar - | tar xf -; }
aws s3 sync --only-show-errors $E/asr_hf/SRT/ $W/asr/SRT/
P=$W/prep21; mkdir -p $P/_crops $P/_portraits
for src in $W/e21/prep_hf $W/e21/prep_360p; do
  for d in $src/*/*/; do y=$(basename $(dirname $d)); v=$(basename $d); [ "${y#_}" = "$y" ] || continue; mkdir -p $P/$y; ln -sfn ${d%/} $P/$y/$v; done
  cp -rn $src/_crops/. $P/_crops/ 2>/dev/null; cp -rn $src/_portraits/. $P/_portraits/ 2>/dev/null
done
cp ~/miss21.txt $W/clips21.txt; n=0; while read c; do [ -f $P/$c/meta.json ] && n=$((n+1)); [ -f $W/asr/SRT/$c.srt ] || echo "NO_SRT $c"; done < $W/clips21.txt; echo "PREP_FOUND $n/21"
O=$W/out21_4b; mkdir -p $O; cd /scratch/code_f9/prep_code
WHISPERX_SRT=$W/asr/SRT PYTHONUNBUFFERED=1 /scratch/vllm/bin/python -u run_dense_all.py --clips $W/clips21.txt --prep-root $P --out-root $O \
  --ports 8300 --per-server 21 --prompt-variant desc_action_v2 --model Qwen/Qwen3.5-4B > $O/driver.log 2>&1
echo DD_EXIT $?; echo "DONE $(find $O -name adlm_descriptions.json | wc -l)/21"
. /scratch/cr.env 2>/dev/null || . $HOME/.cr.env; aws s3 sync --only-show-errors $O $R/dd_inputs/full4b/out21 --exclude "*" --include "*/adlm_descriptions.json" --include "*/adlm_timeline.json"
aws s3 cp --only-show-errors $O/driver.log $R/dd_inputs/full4b/out21/driver.log; echo DD21_UPLOADED
for p in $(pgrep -f "vllm serve"); do kill $p; done
