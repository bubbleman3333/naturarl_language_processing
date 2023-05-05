import matplotlib.pyplot as plt
import numpy as np

from src.自然言語学習.common.util import preprocess, create_co_matrix, cos_similarity, most_similar, p_pmi

text = "You say Goodbye and I say hello ."

word_to_id, id_to_word, corpus = preprocess(text)
vocab_size = len(word_to_id)

c_matrix = create_co_matrix(corpus, vocab_size, 2)

word = "say"

most_similar(word, word_to_id, id_to_word, c_matrix)

w = p_pmi(c_matrix)
print(w)
most_similar(word, word_to_id, id_to_word, w)

u, s, v = np.linalg.svd(w)
print(u.shape)
print(u)

for word, word_id in word_to_id.items():
    plt.annotate(word, (u[word_id, 0], u[word_id, 1]))
plt.scatter(u[:, 0], u[:, 1], alpha=0.5)
plt.show()
