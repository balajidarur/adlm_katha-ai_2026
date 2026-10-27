# box C: probe (top-5 heaviest, memory) -> overfit (random 5, 30 epochs) -> TF + free inference on those 5
cd ~/adlm/LLaMA-Factory && export PATH=$HOME/adlm/venv/bin:$PATH HF_HOME=$HOME/adlm/hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORCE_TORCHRUN=1
python /scratch2/rfcode/c_checks.py
for y in reframed_probe_top5 reframed_overfit5; do
  echo "$(date +%T) START $y"
  ( while sleep 20; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; done ) > /scratch/runs/$y.mem &
  M=$!; llamafactory-cli train examples/budget_ad/$y.yaml > /scratch/runs/$y.log 2>&1; rc=$?; kill $M
  echo "$(date +%T) END $y exit=$rc peak_MiB=$(sort -n /scratch/runs/$y.mem | tail -1)"
  [ $rc -eq 0 ] || { echo CHECKS_FAILED; exit 1; }
done
for m in tf free; do
  echo "$(date +%T) START infer_$m"
  llamafactory-cli train examples/budget_ad/reframed_overfit5_$m.yaml > /scratch/runs/overfit5_infer_$m.log 2>&1
  echo "$(date +%T) END infer_$m exit=$?"
done
echo CHECKS_DONE
