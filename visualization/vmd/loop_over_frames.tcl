
# Set the number of frames
set num_frames [molinfo top get numframes]

# Loop through frames and render Tachyon scenes
for {set frame 0} {$frame < $num_frames} {incr frame} {
    # Set the current frame
    animate goto $frame

    # Generate Tachyon scene file
    render Tachyon frame_$frame.dat
}
