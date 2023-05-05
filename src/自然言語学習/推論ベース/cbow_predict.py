import numpy as np
from src.ニューラルネットワークの練習.クラス一覧 import MatMul

c1, c2 = np.zeros((1, 7)), np.zeros((1, 7))

c1[0, 0] = 1
c2[0, 2] = 1

w_in = np.random.randn(7, 3)
w_out = np.random.randn(3, 7)

in_layer1 = MatMul(w_in)
in_layer2 = MatMul(w_in)
out_layer = MatMul(w_out)

h1 = in_layer1.forward(c1)
h2 = in_layer2.forward(c2)

h = 0.5 * (h1 + h2)

s = out_layer.forward(h)

print(s)
