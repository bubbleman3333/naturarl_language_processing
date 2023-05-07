import cupy as np

import pickle
from src.ニューラルネットワークの練習.クラス一覧 import Adam, CBOw, to_gpu, to_cpu
from src.dataset import ptb
from src.自然言語学習.common.trainer import Trainer

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
