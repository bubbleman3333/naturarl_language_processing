import numpy as np
import collections


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


class MatMul:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.x = None

    def forward(self, x):
        w, = self.params
        out = np.dot(x, w)
        self.x = x
        return out

    def backward(self, d_out):
        w, = self.params
        dx = np.dot(d_out, w.T)
        dw = np.dot(self.x.T, d_out)
        self.grads[0][...] = dw
        return dx


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


class Embedding:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.idx = None

    def forward(self, idx):
        w, = self.params
        self.idx = idx
        out = w[idx]
        return out

    def backward(self, d_out):
        dw, = self.grads
        dw[...] = 0

        np.add.at(dw, self.idx, d_out)


class EmbeddingDot:
    def __init__(self, w):
        self.embed = Embedding(w)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_w = self.embed.forward(idx)
        # 列毎に総和を求めている
        out = np.sum(target_w * h, axis=1)

        self.cache = (h, target_w)
        return out

    def backward(self, d_out):
        h, target_w = self.cache
        d_out = d_out.reshape(d_out.shape[0], 1)

        d_target_w = d_out * h
        self.embed.backward(d_target_w)
        dh = d_out * target_w
        return dh


class SigmoidWithLoss:

    # 初期化メソッドの定義
    def __init__(self):
        self.params = []  # パラメータ
        self.grads = []  # 勾配
        self.loss = None  # 損失
        self.y = None  # 正規化後の値
        self.t = None  # 教師ラベル

    # 順伝播メソッドの定義
    def forward(self, x, t):
        self.t = t  # 教師ラベル

        # 確率に変換
        self.y = 1 / (1 + np.exp(-x))  # シグモイド関数:式(1.5)

        # 交差エントロピー誤差を計算
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)  # 1.3.1項

        return self.loss

    # 逆伝播メソッドの定義
    def backward(self, dout=1):
        # バッチサイズを取得
        batch_size = self.t.shape[0]

        # 勾配を計算
        dx = (self.y - self.t) * dout / batch_size  # :式(A.4)

        return dx


# 負例のサンプラーの実装
class UnigramSampler:

    # 初期化メソッドの定義
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size  # サンプリングする単語数
        self.vocab_size = None  # 語彙数
        self.word_p = None  # 単語ごとのサンプリング確率

        # 出現回数をカウント
        counts = collections.Counter()  # 受け皿を初期化
        for word_id in corpus:
            counts[word_id] += 1

        # 語彙数を保存
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        # 出現回数からサンプリング確率を計算
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]  # カウントを移す
        self.word_p = np.power(self.word_p, power)  # 値を調整:式(4.4)の分子
        self.word_p /= np.sum(self.word_p)  # 確率に変換(割合を計算):式(4.4)

    # サンプリングメソッドの定義
    def get_negative_sample(self, target):
        # バッチサイズ(ターゲット数)を取得
        batch_size = target.shape[0]

        # サンプリング
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)  # 受け皿を初期化
        for i in range(batch_size):
            # 単語ごとのサンプリング確率を取得
            p = self.word_p.copy()

            # ターゲット自体の単語の確率を0にする
            target_idx = target[i]  # ターゲットのインデックスを取得
            p[target_idx] = 0  # ターゲットの単語の確率を0にする
            p /= p.sum()  # 再正規化

            # サンプリング
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, w, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]

        self.embed_dot_layers = [EmbeddingDot(w) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正解レイヤ
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)

        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss

    def backward(self, d_out=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            d_score = l0.backward(d_out)
            dh += l1.backward(d_score)
        return dh


class CBOw:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        v, h = vocab_size, hidden_size
        w_in = 0.01 * np.random.randn(v, h).astype("f")
        w_out = 0.01 * np.random.randn(v, h).astype("f")

        self.in_layers =
