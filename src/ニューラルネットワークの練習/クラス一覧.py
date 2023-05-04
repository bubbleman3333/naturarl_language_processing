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
        self.grads = [np.zeros_like(w), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        w, b = self.params

        out = np.dot(x, w) + b
        self.x = x
        return out

    def backward(self, d_out):
        w, b = self.params
        dx = np.dot(d_out, w.T)
        dw = np.dot(self.x.T, d_out)
        db = np.sum(d_out, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        dx = d_out * (1 - self.out) * self.out
        return dx


class TwoLayerNet:
    def __init__(self, i_size, h_size, o_size):
        w1 = 0.01 * np.random.randn(i_size, h_size)
        b1 = np.zeros(h_size)
        w2 = 0.01 * np.random.randn(h_size, o_size)
        b2 = np.zeros(o_size)

        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w2, b2)
        ]

        self.loss_layer = SoftmaxWithLoss()

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, d_out=1):
        d_out = self.loss_layer.backward(d_out)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out


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

    def backward(self, d_out=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= d_out
        dx = dx / batch_size

        return dx


class SDG:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
