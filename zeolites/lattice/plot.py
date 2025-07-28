
import numpy as np
import matplotlib.pyplot as plt

errors = ["a_error", "b_error", "c_error","alpha_error", "beta_error", "gamma_error"]



for e in errors:
    plt.plot(np.load(f"{e}.npy"), label = e)


plt.show()

