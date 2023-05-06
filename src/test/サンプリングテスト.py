import numpy as np

x = np.arange(4)

p = np.array([0.1, 0.3, 0.5, 0.1])

s = np.random.choice(x, p=p)
print(s)
