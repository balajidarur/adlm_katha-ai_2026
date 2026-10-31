# box A: copy the 1 fps 448x252 frame pool to S3 once (for boxes without /data), then write the _DONE marker
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH
W=/scratch; until [ -f $W/creds.env ]; do sleep 10; done; . $W/creds.env
B=s3://balu-transfer-933516006154-apne1/adlm/frames
aws s3 ls $B/_DONE >/dev/null 2>&1 && exit 0
aws s3 sync --only-show-errors /data/cmd_ad/frames/1fps_448x252/ $B/1fps_448x252/ && . $W/creds.env && \
  echo "$(date -u +%FT%TZ) $(find /data/cmd_ad/frames/1fps_448x252 -type f | wc -l) files" | aws s3 cp - $B/_DONE
