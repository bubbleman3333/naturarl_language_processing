import numpy as np
from src.自然言語学習.まずは単語をIDに変換 import preprocess

word_to_id, id_to_word, corpus = preprocess("You say goodbye and I say hello .")

print(corpus)
print(id_to_word)
