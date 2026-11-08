# E2 probe driver (4 GPUs): bash e2_probe_all.sh <commit> [n_gpus]
# Longest-first (probe_top10, sequential sampler) 2 steps per config; stops escalating at the first config that fits:
#   DDP bs1/ga8 -> ZeRO-2 bs1/ga8 -> ZeRO-3 bs1/ga8 (pure_bf16 off: incompatible with ZeRO-3).
# If a bs1 config fits with >= 12 GiB headroom, bs2 is tried on the same stage. The winner then runs 3 steps on
# probe_rand96 (typical length mix) for s/step. All results: $W/probe_runs/*/result.json and PROBE_SUMMARY lines.
WANT=$1; NG=${2:-4}; GA1=$((32 / NG)); GA2=$((16 / NG))
S=$(dirname $(readlink -f $0))/e2_9b_train.sh
W=/scratch; [ -d /opt/dlami/nvme ] && W=/opt/dlami/nvme
L=$W/probe_all.log
res(){ python3 -c "
import json,glob,os; f=max(glob.glob('$W/probe_runs/*/result.json'),key=os.path.getmtime); r=json.load(open(f))
ok = (not r['oom']) and r['rc']==0 and len(r['steps'])>0
print(('FIT' if ok else 'FAIL'), max(r['peak_mib'].values()) if r['peak_mib'] else 0, r['tag'])"; }
run(){ # stage bs ga dataset steps [extra]
  EXTRA_ARGS="$6" bash $S probe $WANT $NG $2 $3 $1 $4 $5 >> $L 2>&1
  R=$(res); echo "PROBE_SUMMARY $(date +%T) stage=$1 bs=$2 ga=$3 ds=$4 steps=$5 extra='$6' -> $R" | tee -a $W/probe_summary.txt; }
CH=""
for st in none z2 z3; do
  X=""; [ $st = z3 ] && X="pure_bf16=false"
  run $st 1 $GA1 probe_top10 2 "$X"
  set -- $R
  if [ "$1" = FIT ]; then CH="$st 1 $GA1 $X"
    if [ $(( 46000 - $2 )) -ge 12288 ]; then run $st 2 $GA2 probe_top10 2 "$X"; set -- $R; [ "$1" = FIT ] && CH="$st 2 $GA2 $X"; fi
    break
  fi
done
[ -z "$CH" ] && { echo "PROBE_SUMMARY nothing fits" | tee -a $W/probe_summary.txt; exit 1; }
set -- $CH
run $1 $2 $3 probe_rand96 3 "${4:-}"
echo "PROBE_CHOSEN $CH" | tee -a $W/probe_summary.txt
echo PROBE_ALL_DONE >> $W/probe_summary.txt
