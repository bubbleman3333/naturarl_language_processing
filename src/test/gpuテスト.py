import cupy as cp
import numpy as np
import timeit


# NumPy（CPU）での処理時間を計測
def numpy_multiply():
    x = np.random.rand(1000, 1000)
    y = np.random.rand(1000, 1000)
    z = np.dot(x, y)


numpy_time = timeit.timeit(numpy_multiply, number=100)


# CuPy（GPU）での処理時間を計測
def cupy_multiply():
    x = cp.random.rand(10000, 1000)
    y = cp.random.rand(1000, 10000)
    z = cp.dot(x, y)


cupy_time = timeit.timeit(cupy_multiply, number=100)

print("NumPy Time:", numpy_time)
print("CuPy Time:", cupy_time)
