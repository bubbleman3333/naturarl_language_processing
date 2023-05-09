## RNNとは
- Recurrent Neural Network
過去の情報を利用して現在及び将来の入力に対するネットワークの性能の向上させるディープラーニングの構造
- 同じレイヤ内で循環する特性を持つ

<br>

## tanh関数の式
## $f(x) = \frac{e^x - e^-x}{e^x + e^-x}  $ 

![tanhグラフ](https://lh3.googleusercontent.com/0MV3pxIbo1fZtB7S6AcyOmT89_KDKi0lM8JjJmvGJZbfk_scZFVIsWEr7yiFh1SlXMpadGjFQseft4jPnZfdEOC7xJiV_5T7upwAEJ8L)
### 値域は -1 ～ 1

<br>

## RNNの時刻ｔの式
$h_t = tanh(h_{t-1}W_h + X_tW_x+b)$

 w_hは１つ前の入力の演算結果用の重み
 W_xは現在時刻の入力用の重み
RNNの出力 $h_t$ は<strong>隠れ状態・隠れベクトル</strong>と呼ばれる。

## RNNの問題点
- 長い時刻を遡れば遡るほど勾配消失しやすい
- 計算に時間がかかる