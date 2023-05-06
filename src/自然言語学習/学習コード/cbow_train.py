from src.自然言語学習.common.trainer import Trainer
from src.ニューラルネットワークの練習.クラス一覧 import Adam
from src.自然言語学習.学習クラス.simpleCbow import SimpleCBow
from src.自然言語学習.まずは単語をIDに変換 import preprocess
from src.自然言語学習.common.util import create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 10
batch_size = 3
max_epoch = 1000

text = "You say goodbye and I say hello ."
word_to_id, id_to_word, corpus = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)

target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBow(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
word_vectors = model.word_vectors

for word_id, word in id_to_word.items():
    print(word, word_vectors[word_id])
