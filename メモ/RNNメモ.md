## RNNとは
- Recurrent Neural Network
過去の情報を利用して現在及び将来の入力に対するネットワークの性能の向上させるディープラーニングの構造
- 同じレイヤ内で循環する特性を持つ
- RNNの誤差逆伝播法は<strong>BPTT</strong>(Backprobagation Through Time)という
- データをシーケンシャルに与えることが重要

<br>

## tanh関数の式
## $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}  $ 

 値域は -1 ～ 1

<br>

## RNNの時刻ｔの式
$h_t = tanh(h_{t-1}W_h + X_tW_x+b)$

 w_hは１つ前の入力の演算結果用の重み<br>
 W_xは現在時刻の入力用の重み<br>
RNNの出力 $h_t$ は<strong>隠れ状態・隠れベクトル</strong>と呼ばれる。

## RNNの問題点
- 長い時刻を遡れば遡るほど勾配消失しやすい
- 計算に時間がかかる
- 逆伝播をする為に中間データをメモリに保持しておかなければいけない

## 解決策
時間軸方向に長くなりすぎたネットワークを適当な場所で切り取る
これを<strong>Truncated BPTT</strong>という
- 順伝播時　→　そのまま流す
- 逆伝播時  →  ある程度の長さで断ち切る



## 次元数確認式
$N \times H・H \times H =  N \times D・D \times H$

N:バッチサイズ 
H:隠れ状態ベクトル
D:入力次元
