# keep trying to start Tokyo L40S boxes (capacity errors) within the 128 G-vCPU quota; one line per start
export AWS_PROFILE=PowerUserAccess-933516006154 AWS_REGION=ap-northeast-1
# id:vcpu, in priority order (4xL40S first)
LIST="i-0f7aa5fa06eb456f0:48 i-0399ca646839ad221:48 i-0e97ee326c3e0481f:48 i-0d60f97e81b6cff39:8 i-047d79b492e17ffd8:8 i-08426abd2371fa429:8 i-0fb18319dc9e4e311:8 i-0ed096fc6b4741f82:8 i-0e87257dc5f698926:8"
for round in $(seq 1 60); do
  used=0; for x in $LIST i-06321e7ca91b0cf12:8; do id=${x%:*}; st=$(aws ec2 describe-instances --instance-ids $id --query 'Reservations[0].Instances[0].State.Name' --output text)
    case $st in pending|running) used=$((used+${x#*:}));; esac; done
  for x in $LIST; do id=${x%:*}; v=${x#*:}
    st=$(aws ec2 describe-instances --instance-ids $id --query 'Reservations[0].Instances[0].State.Name' --output text)
    [ "$st" = stopped ] || continue; [ $((used+v)) -le 128 ] || continue
    aws ec2 start-instances --instance-ids $id > /dev/null 2>&1 && { used=$((used+v)); echo "$(date +%T) STARTED $id ($v vCPU, used $used)"; }
  done
  [ $used -ge 120 ] && { echo "QUOTA_FULL $used"; exit 0; }
  sleep 60
done
echo RETRY_END
