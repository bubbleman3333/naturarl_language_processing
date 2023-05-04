import numpy as np
from src.ニューラルネットワークの練習.クラス一覧 import TwoLayerNet, SDG
from src.dataset import spiral
import matplotlib.pyplot as plt

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1

x, t = spiral.load_data()
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # ランダムにシャッフル
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]
