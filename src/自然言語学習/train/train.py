import cupy as np

import pickle
from src.ニューラルネットワークの練習.クラス一覧 import Adam, CBOw, to_gpu, to_cpu
from src.dataset import ptb
from src.自然言語学習.common.trainer import Trainer
from src.自然言語学習.common.util import create_contexts_target

gpu = True
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if gpu:
    contexts, target = to_gpu(contexts), to_gpu(target)

model = CBOw(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vectors = model.word_vectors

if gpu:
    word_vectors = to_cpu(word_vectors)
params = {"word_vectors": word_vectors, "word_to_id": word_to_id, "id_to_word": id_to_word}
pkl = "cbow_params.pkl"
with open(pkl, "wb") as f:
    pickle.dump(params, f, -1)
