#/bin/bash

# ffmpeg
ffmpeg -i speaking_fish.avi -r 1 frames/frame_%05d.jpg
