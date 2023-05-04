import cupy
import time

start = time.time()

x = cupy.random.random((10, 10000))

p = cupy.random.random((10000, 10000))

x = cupy.asarray(x)
p = cupy.asarray(p)
t = cupy.dot(x, p)
print(time.time() - start)
print(cupy.get_array_module(t))