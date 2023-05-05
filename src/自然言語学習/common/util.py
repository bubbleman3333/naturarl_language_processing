import numpy as np


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)


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


def preprocess(t):
    words = t.lower().split(" ")

    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
    id_to_word = {v: k for k, v in word_to_id.items()}

    corpus = np.array([word_to_id[w] for w in words])
    return word_to_id, id_to_word, corpus


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f"{query} is not found")
        return
    print(f"[query]{query}")
    query_vec = word_matrix[word_to_id[query]]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    for i in (-1 * similarity).argsort()[:top]:
        print(f"{id_to_word[i]}:{similarity[i]}")


def p_pmi(c, verbose=False, eps=1e-8):
    m = np.zeros_like(c, dtype=np.float32)
    n = np.sum(c)
    s = np.sum(c, axis=0)
    total = c.shape[0] * c.shape[1]
    cnt = 0

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            pmi = np.log2(c[i, j] * n / (s[i] * s[j]) + eps)
            m[i, j] = max(0, pmi)
            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print(f"{100 * cnt / total} done")
    return m


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:  # 1次元の場合 (target の場合)
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)  # ゼロ行列作成
        for idx, word_id in enumerate(corpus):  # targetからword_idへ順次代入
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:  # 2次元の場合 (contexts の場合)
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)  # ゼロ行列作成
        for idx_0, word_ids in enumerate(corpus):  # contextsからword_idsへ順次代入
            for idx_1, word_id in enumerate(word_ids):  # word_idsからword_idへ順次代入
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
