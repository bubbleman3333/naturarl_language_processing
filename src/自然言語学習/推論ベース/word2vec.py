from src.ニューラルネットワークの練習.クラス一覧 import MatMul

import numpy as np

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
w = np.random.randn(7, 3)
layer = MatMul(w)
h = layer.forward(c)
print(h)