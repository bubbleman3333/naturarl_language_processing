import MeCab

tagger = MeCab.Tagger()
result = tagger.parse("こんにちは、今日僕はとても美味しいカレーを作って食べました。ここまでの話から、RNNの全貌が見えてきました。私たちが"
                      "これから実装すべきは、つまるところ、横方向に伸びたニューラルネットワークです。")
print(result)

result = tagger.parse("きみはしらないか？")
print(result)
result = tagger.parse("君走らないか？")
print(result)

for i in result.splitlines():
    print(i.split("	")[0])