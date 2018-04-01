from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def kmeans(img, iters=50, k=16):
    np.random.seed(42)
    dim = img.shape[0]

    img = img.reshape(-1, 3)

    length = dim ** 2
    indices = np.random.randint(0, length, size=k)
    centroids = img[indices]
    groups = None

    for iter in range(iters):
        dists = []
        for c in centroids:
            dists.append(np.sum((img - c) ** 2, axis=1))
        groups = np.stack(dists).argmin(axis=0)

        loss = np.sum(np.stack(dists).min(axis=0))

        new_centroids = []
        for i in range(k):
            new_centroids.append(np.mean(img[groups == i], axis=0))

        centroids = np.stack(new_centroids)

    graph = np.zeros_like(img)
    for i in range(k):
        idxes = np.where(groups == i)
        graph[idxes] = centroids[i]
    plt.imshow(graph.reshape(dim, dim, 3))
    plt.show()

    return


#
# small = imread('mandrill-small.tiff')
# kmeans(small)

large = imread('mandrill-large.tiff')
kmeans(large)
