import numpy as np


temp_threshold = 573.0
std_threshold = 43.0

var = np.load("uncertainty.npy")
time_ps, etot_eV, epot_eV, ekin_eV, temp_K = np.loadtxt("../langevin.log", skiprows=1, unpack = True)

nframes = len(var)
std = np.sqrt(var)

frame = 0
print("Identifying frames for Future AIMD")


with open("High_Temp_Frames.txt", "w") as fht:
    with open("High_Unc_Frames.txt", "w") as fhu:

        for i in range(nframes):

            t = temp_K[i]
            s = std[i]

            if t >= temp_threshold:
                fht.write(f"Frame {i} exceeds temperature threshold with a temp of {t} K\n")

            if s >= std_threshold:
                fhu.write(f"Frame {i} exceeds uncertainty threshold with an uncertainty of {s}\n")

            frame += 1



