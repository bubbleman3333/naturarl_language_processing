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
