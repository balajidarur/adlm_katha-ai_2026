# local driver for the scene-shot redo: merge shot caches -> prep on A -> distribute -> relaunch DDs on all boxes
D=/workspace/dbalu/reframed; C=$D/code; K=~/.ssh/balu-aws-test.pem; A=ubuntu@13.115.233.87
log(){ echo "$(date +%T) $*"; }

# 1. wait until local + A shot caches together cover every excerpt
while :; do
  scp -q -i $K $A:/scratch2/shots_cache_scenes_A.json $D/shots_cache_scenes_A.json 2>/dev/null
  n=$(python3 -c "
import json,os
a=json.load(open('$D/shots_cache_scenes.json')) if os.path.exists('$D/shots_cache_scenes.json') else {}
b=json.load(open('$D/shots_cache_scenes_A.json')) if os.path.exists('$D/shots_cache_scenes_A.json') else {}
print(len(set(a)|set(b)))" 2>/dev/null)
  log "shots union ${n:-0}/2023"
  [ "${n:-0}" -ge 2023 ] && break
  sleep 30
done
bash $C/kill_detect.sh
ssh -i $K $A 'pkill -f 'detect_[A].py'' < /dev/null
python3 - <<EOF
import json
a=json.load(open('$D/shots_cache_scenes.json')); b=json.load(open('$D/shots_cache_scenes_A.json'))
m={**b,**a}; json.dump(m,open('$D/shots_cache_scenes_merged.json','w'))
L=[x['end']-x['start'] for v in m.values() for x in v]
print(f"MERGED {len(m)} excerpts, {len(L)} shots, mean {sum(L)/len(L):.2f}s, median {sorted(L)[len(L)//2]:.2f}s, at 10s cap {sum(abs(x-10)<0.01 for x in L)/len(L):.0%}")
EOF

# 2. per-box clip lists, balanced by shot count per GPU
python3 - <<EOF
import json
m=json.load(open('$D/shots_cache_scenes_merged.json'))
import glob,os
clips={os.path.basename(p)[:-4]: p.split('/')[-2]+'/'+os.path.basename(p)[:-4] for p in glob.glob('/storage/datasets/Reframed/*/excerpt/video/*/*.mp4')}
boxes=[l.split() for l in open('$C/boxes.txt') if not l.startswith('#')]
g={b[0]:int(b[1]) for b in boxes}; load={b:0 for b in g}; lists={b:[] for b in g}
for k in sorted(m,key=lambda k:-len(m[k])):
    b=min(load,key=lambda x:(load[x]+len(m[k]))/g[x]); load[b]+=len(m[k]); lists[b].append(clips[k])
for b,l in lists.items():
    open(f'$C/lists/{b}.txt','w').write('\n'.join(l)+'\n'); print(b,len(l),'clips',load[b]//g[b],'shots/GPU')
EOF
ssh -i $K $A 'mkdir -p /tmp/lists' < /dev/null
scp -q -i $K $C/lists/*.txt $A:/tmp/lists/ && scp -q -i $K $C/dist_from_A.sh $A:/tmp/
scp -q -i $K $D/shots_cache_scenes_merged.json $A:/scratch2/shots_merged.json && ssh -i $K $A 'touch /scratch2/shots_merged.ready' < /dev/null
log "prep started on A"

# 3. wait for prep, then distribute
until ssh -i $K $A 'grep -q PREP2_DONE /scratch/rf_prep_scenes.log' < /dev/null 2>/dev/null; do
  log "prep $(ssh -i $K $A 'ls -d /scratch/rf_prep_scenes/*/*/meta.json 2>/dev/null | wc -l' < /dev/null)/2023"; sleep 30; done
ssh -i $K $A 'tail -1 /scratch/rf_prep_scenes.log' < /dev/null
ssh -i $K $A 'bash /tmp/dist_from_A.sh' < /dev/null

# 4. restart vLLM on A, launch DDs everywhere (each box waits for its servers)
ssh -i $K $A 'for g in 0 1 2 3; do nohup bash /scratch2/serveQ.sh $g > /scratch2/vllm_$g.log 2>&1 < /dev/null & done; echo "SERVERS_READY 4 (restart)" >> /tmp/box_setup.log' < /dev/null
bash $C/launch_all.sh
log ALL_LAUNCHED
