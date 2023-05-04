import numpy as np

x = np.zeros(100)

t = [x, np.ones(100)]



t[0][...] = np.random.random(13)
print(t)
