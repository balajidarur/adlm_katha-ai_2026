# s1a: run the full-run driver (same args as full_worker.sh + the run's q3_merge_cover t0.7/top-p 0.8) on the 21 clips
set -x
W=/opt/dlami/nvme/e3; cd $W/code/prep_code; . $W/cr.env
head -16 $W/new21.txt > $W/a16.txt; tail -5 $W/new21.txt > $W/b5.txt
md5sum run_dense_all.py charbank_pipeline.py prompts_adlm_charbank.py
export WHISPERX_SRT=$W/asr_all/SRT PYTHONUNBUFFERED=1
mkdir -p $W/q3_out
for pair in "a16.txt prep_hf" "b5.txt prep_360p"; do set -- $pair
  $W/vllm/bin/python -u run_dense_all.py --clips $W/$1 --prep-root $W/$2 --out-root $W/q3_out --ports 8200 --per-server 16 \
    --prompt-variant q3_merge_cover --model Qwen/Qwen3-VL-8B-Instruct --temperature 0.7 --top-p 0.8 >> $W/q3_out/driver.log 2>&1
  echo "DRIVER $1 exit=$?" | tee -a $W/progress.txt
done
n=$(ls $W/q3_out/*/Qwen3-VL-8B-Instruct/adlm_descriptions.json 2>/dev/null | wc -l); echo "DD21 done: $n descriptions" | tee -a $W/progress.txt
. $W/cr.env; aws s3 sync --only-show-errors $W/q3_out/ s3://balu-transfer-933516006154-apne1/cmd_ad_eval591/q3vl8b_dense_e3/ && echo "DD21_UPLOADED $(date -u +%T)" | tee -a $W/progress.txt
