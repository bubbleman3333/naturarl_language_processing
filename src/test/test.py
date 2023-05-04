import numpy as np

d, n = 8, 7
x = np.random.randn(1, d)

p = np.repeat(x, n, axis=0)
print(p)
