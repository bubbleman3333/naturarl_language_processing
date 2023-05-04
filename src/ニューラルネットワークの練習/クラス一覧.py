import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class Affine:
    def __init__(self, w, b):
        self.params = [w, b]

    def forward(self, x):
        w, b = self.params
        return np.dot(x, w) + b

    def backward(self):
        return


class Sigmoid:
    def __init__(self):
        self.params = []

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))


class TwoLayerNet:
    def __init__(self, i_size, h_size, o_size):
        w1 = np.random.randn(i_size, h_size)
        b1 = np.random.randn(h_size)
        w2 = np.random.randn(h_size, o_size)
        b2 = np.random.randn(o_size)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]

        self.params = [layer.params for layer in self.layers]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
