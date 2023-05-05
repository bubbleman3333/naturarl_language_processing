from src.dataset import ptb
from src.自然言語学習.common.util import preprocess, create_co_matrix, cos_similarity, most_similar, p_pmi
from sklearn.utils.extmath import randomized_svd
import numpy as np

window_size = 2
word_vec_size = 100
corpus, word_to_id, id_to_word = ptb.load_data("train")

vocab_size = len(word_to_id)

c = create_co_matrix(corpus, vocab_size, window_size)

w = p_pmi(c, verbose=True)

u, s, v = randomized_svd(w, n_components=word_vec_size, n_iter=5, random_state=None)

word_vectors = u[:, :word_vec_size]

queries = ["you", "hi", "great", "hey", "sleepy"]
for query in queries:
    most_similar(query, word_to_id, id_to_word, word_vectors)
