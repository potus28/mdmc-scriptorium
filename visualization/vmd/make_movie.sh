#!/bin/bash

cmd='/Applications/VMD_1.9.4a57-x86_64-Rev12.app/Contents/vmd/tachyon_MACOSXX86_64'


cd Movie
rm input.txt movie.mp4
for ii in $(seq 0 5 5000); do
    $cmd -aasamples 36 frame_${ii}.dat -format PPM -res 2048 2048 -o frame_${ii}.ppm
    echo "file 'frame_${ii}.ppm'" >> input.txt
    echo "duration 0.033" >> input.txt

done

ffmpeg -f concat -i input.txt -f mp4 -q:v 0 -vcodec mpeg4 -b:v 5000k movie.mp4
cd ..
