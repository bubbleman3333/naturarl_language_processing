import numpy as np
from src.ニューラルネットワークの練習.クラス一覧 import TwoLayerNet

two_layer_net = TwoLayerNet(3, 10, 2)

result = two_layer_net.predict(np.random.randn(3))
print(result)
