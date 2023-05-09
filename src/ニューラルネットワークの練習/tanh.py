import numpy as np
import matplotlib.pyplot as plt


def tan_h(xx):
    return (1 - np.exp(-2 * xx)) / (1 + np.exp(-2 * x))


fig, axes = plt.subplots()

x = np.arange(-3, 3, 0.01)
axes.plot(x, tan_h(x))
plt.hlines(0, min(x), max(x), "black")
plt.vlines(0, -1, 1, "black")
plt.show()
