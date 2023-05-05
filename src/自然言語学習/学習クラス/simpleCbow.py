import numpy as np

from src.ニューラルネットワークの練習.クラス一覧 import MatMul, SoftmaxWithLoss


class SimpleCBow:
    def __init__(self, vocab_size, hidden_size):
        v, h = vocab_size, hidden_size
        w_in = 0.01 * np.random.randn(v, h).astype("f")
        w_out = 0.01 * np.random.randn(h, v).astype("f")

        self.in_layer0 = MatMul(w_in)
        self.in_layer1 = MatMul(w_in)
        self.out_layer = MatMul(w_out)
        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vectors = w_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer0.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, d_out=1):
        ds = self.loss_layer.backward(d_out)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return
