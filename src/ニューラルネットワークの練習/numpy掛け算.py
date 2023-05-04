import numpy as np
import time

start = time.time()
t = np.float32

x = np.random.random((10, 10000)).astype(t)

p = np.random.random((10000, 10000)).astype(t)

t = np.dot(x, p)
print(t.dtype)
print(time.time() - start)
