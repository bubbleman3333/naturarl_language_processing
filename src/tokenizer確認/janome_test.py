from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()

sentences = [
    "とても眠いよー！",
    "It's very exciting!!",
    "ちょっと難しそうな文章を与えてみます。"
]

for sentence in sentences:
    print("=============================================")
    print(sentence)

    for token in tokenizer.tokenize(sentence):
        print("    " + str(token))
