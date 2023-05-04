import numpy as np


def sigmoid(xx):
    return 1 / np.exp(-xx)


x = np.random.randn(10, 2)
w = np.random.randn(2, 4)
b = np.random.randn(4)

answer = np.dot(x, w) + b
answer = sigmoid(answer)

w2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

answer = answer.dot(w2) + b2
print(answer)
