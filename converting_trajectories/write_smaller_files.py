from ase.io import read, write

traj = read("train.xyz", index = ":")
nframes = len(traj)


index = 0

frame_count = 0
for atoms in traj:

    write(f"train.{index}.xyz", atoms, append = True)
    if frame_count % 300 == 0:
        index += 1

    frame_count += 1

