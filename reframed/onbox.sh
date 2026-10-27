# usage: onbox.sh NAME 'command'  — run a command on a fleet box (boxes.txt), via a jump host if needed
K=~/.ssh/balu-aws-test.pem; read n g ip j <<< "$(grep "^$1 " /workspace/dbalu/reframed/code/boxes.txt)"
O=(-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o LogLevel=ERROR)
case $j in A) O+=(-o "ProxyCommand=ssh -i $K -o StrictHostKeyChecking=no -W %h:%p ubuntu@13.115.233.87");;
           S) O+=(-o "ProxyCommand=ssh -i $K -o StrictHostKeyChecking=no -W %h:%p ubuntu@54.117.65.59");; esac
ssh "${O[@]}" -i $K ubuntu@$ip "bash -s" <<< "$2"
