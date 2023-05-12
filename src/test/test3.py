import numpy as np

k = int(input())
base_strings = input().split(" ")
holes = np.array([int(input()) for k in range(int(input()))]) - 1

answer = []
base = np.array([-1] * k)

ll = np.arange(k)
while True:
    temp = ll[holes]
    answer += temp[temp != -1].tolist()
    ll[holes] = -1
    base_copy = base.copy()
    temp = ll[ll != -1]
    if temp.size == 0:
        break
    base_copy[:temp.size] = temp
    ll = base_copy

output = [""] * k
for idx, i in enumerate(answer):
    output[i] = base_strings[idx]
print("".join(output))
