import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np


n = 100
num_k = 3
x = np.random.random((2, n))
mapp = x.reshape((n, 2))
base = mapp[np.random.choice(n, num_k, replace=False)]
labels = np.random.randint(0, num_k, n)
count = 0
markers = ["*", "+", "o"]
colors = ["r", "g", "b"]
frames = []
fig, ax = plt.subplots()
while True:
    count += 1
    t = []
    for i in range(num_k):
        p = mapp[labels == i]
        t.append(ax.scatter(p[:, 0], p[:, 1], marker=markers[i], color=colors[i]))
    frames.append(t)

    for i in range(num_k):
        clusters = mapp[labels == i]
        base[i] = clusters.mean(axis=0)
    dist = ((mapp[:, np.newaxis, :] - base[np.newaxis, :, :]) ** 2).sum(axis=2)
    labels_prev = labels.copy()
    labels = np.argmin(dist, axis=1)
    if (labels_prev == labels).all():
        break

ani = ArtistAnimation(fig, frames, interval=300)
plt.show()