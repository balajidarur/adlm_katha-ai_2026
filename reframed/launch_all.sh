# copy each box its clip list + runner and start it (it waits for its own servers)
D=/workspace/dbalu/reframed/code
for n in $(grep -v '^#' $D/boxes.txt | awk '{print $1}'); do
  ( g=$(grep "^$n " $D/boxes.txt | awk '{print $2}')
    bash $D/onbox.sh $n "cat > /tmp/list.txt <<'L'
$(cat $D/lists/$n.txt)
L
cat > /tmp/run_box.sh <<'R'
$(cat $D/run_box.sh)
R
nohup bash /tmp/run_box.sh $g > /tmp/run_box.log 2>&1 < /dev/null &
echo $n launched \$(wc -l < /tmp/list.txt) clips" ) &
done; wait
