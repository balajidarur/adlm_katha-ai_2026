# retry starting the Osaka L40S box every 2 min for 2 h; prints OSAKA_STARTED when it gets capacity
export AWS_PROFILE=PowerUserAccess-933516006154
for i in $(seq 1 60); do
  aws ec2 start-instances --region ap-northeast-3 --instance-ids i-0cb95bbcd773f4bdc > /dev/null 2>&1 && { echo "$(date +%T) OSAKA_STARTED"; exit 0; }
  sleep 120
done
echo "$(date +%T) OSAKA_GAVE_UP"
