from src.自然言語学習.common.util import preprocess, create_co_matrix, cos_similarity, most_similar

text = "You say Goodbye and I say hello ."

word_to_id, id_to_word, corpus = preprocess(text)
vocab_size = len(word_to_id)

c_matrix = create_co_matrix(corpus, vocab_size, 2)

word = "you"

most_similar(word, word_to_id, id_to_word, c_matrix)
