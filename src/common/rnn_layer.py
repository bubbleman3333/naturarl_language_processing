from src.conf import config

if config.GPU:
    import cupy as np
else:
    import numpy as np


class RNN:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        wx, wh, b = self.params
        t = np.dot(h_prev, wh) + np.dot(x, wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        wx, wh, b = self.params
        x, h_prev, h_next = self.cache
        # tan_hの微分
        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dwh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, wh.T)
        dwx = np.dot(x.T, dt)
        dx = np.dot(dt, wx.T)

        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, wx, wh, b, stateful=False):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]

        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        wx, wh, b = self.params
        n, t, d = xs.shape
        d, h = wx.shape
        self.layers = []
        hs = np.empty((n, t, h), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((n, h), dtype="f")

        for t_idx in range(t):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs
