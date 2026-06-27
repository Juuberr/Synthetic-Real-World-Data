import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.random.rand(100, 5)

nn = NearestNeighbors(n_neighbors=2)
nn.fit(X)

dists, idx = nn.kneighbors(X)

print("OK")