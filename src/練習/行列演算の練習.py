import numpy as np

x = np.random.randn(10, 2)
w = np.random.randn(2, 4)
b = np.random.randn(4)

answer = np.dot(x, w) + b
print(answer.shape)
