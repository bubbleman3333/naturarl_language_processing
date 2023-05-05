from src.ニューラルネットワークの練習.クラス一覧 import MatMul
from src.自然言語学習.common.util import create_contexts_target, convert_one_hot
from src.自然言語学習.まずは単語をIDに変換 import preprocess

import numpy as np

text = "You say goodbye and I say hello ."
word_to_id, id_to_word, corpus = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(target)
print(contexts)
