import numpy as np


def preprocess(t):
    words = t.split(" ")

    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
    id_to_word = {v: k for k, v in word_to_id.items()}

    corpus = np.array([word_to_id[w] for w in words])
    return word_to_id, id_to_word, corpus
