import numpy as np
from src.自然言語学習.まずは単語をIDに変換 import preprocess

text = "You say goodbye and I say hello ."
text_array = text.split(" ")
word_to_id, id_to_word, corpus = preprocess(text)

t = np.zeros((len(word_to_id), len(word_to_id)))


# 共起行列

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


t = create_co_matrix(corpus, len(word_to_id))
print(t)
