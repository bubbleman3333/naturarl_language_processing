import pickle

# from src.dataset import ptb
from src.自然言語学習.common.util import most_similar

gpu = True
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10
# corpus, word_to_id, id_to_word = ptb.load_data("train")

with open("../../../eval/cbow_params2.pkl", "rb") as f:
    params = pickle.load(f)
    word_vectors = params["word_vectors"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]


print(type(word_vectors))

print(most_similar("pretty",word_to_id,id_to_word,word_vectors))