#/bin/bash

# ffmpeg -i speaking_fish.avi -c copy -map 0 -segment_time 5 -f segment segments/out%03d.mp4
ffmpeg -i speaking_fish.avi \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac \
  -f segment -segment_time 5 \
  -reset_timestamps 1 \
  segments/out%03d.mp4
