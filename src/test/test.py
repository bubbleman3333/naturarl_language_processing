import numpy as np

x = np.arange(2, 6)

idx = np.array([2, 3, 2, 5])
dw = np.zeros(6)

x = np.add.at(dw, idx, x)


