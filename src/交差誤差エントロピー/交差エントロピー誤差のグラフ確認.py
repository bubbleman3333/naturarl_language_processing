import matplotlib.pyplot as plt
import numpy as np


def calc(yy, t):
    return -1 * (t * np.log(yy) + (1 - t) * np.log(1 - yy))


x = np.arange(0.01, 1, 0.01)
plt.plot(x, calc(x, 0))
plt.show()
