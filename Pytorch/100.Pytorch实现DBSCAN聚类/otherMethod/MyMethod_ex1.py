import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([
            [1, 2],
            [2, 2],
            [2, 3],
            [4, 3],
            [8, 7],
            [8, 8],
            [2, 9],
            [4, 6]])

clustering = DBSCAN(eps=2, min_samples=2).fit(X)
print(clustering.labels_)
print(clustering)

import sklearn
sklearn.metrics.pairwise_distances