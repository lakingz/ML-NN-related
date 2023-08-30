import numpy as np

#
# #main  code
#
class KMean:
    def __init__(self, max_iteration, k=3):
        self.max_iteration = max_iteration
        self.k = k
        self.x = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def initialization(self):
        num_samples, num_features = self.x.shape
        R = np.zeros((num_samples, self.k))
        for i in range(num_samples):
            which_cluster = np.random.randint(0, self.k)
            R[i, which_cluster] = 1
        return R

    def get_centroid(self, R):
        num_samples, num_features = self.x.shape
        centroids = np.zeros((self.k, num_features))
        for k in range(self.k):
            n = np.sum(R, axis=0)
            centroids[k, :] = np.sum(self.x * R[:, k].reshape(-1, 1), axis=0, keepdims=True) / n[k]
        return centroids

    def get_cluster(self, centroids):
        num_samples, num_features = self.x.shape
        R = np.zeros((num_samples, self.k))
        for i in range(num_samples):
            which_cluster = 0
            min_distance = float('inf')
            for k in range(self.k):
                distance = k_mean.euclidean_distance(self.x[i, :], centroids[k, :])
                if distance < min_distance:
                    min_distance = distance
                    which_cluster = k
            R[i, which_cluster] = 1
        return R

    def onehot_to_list(self, one_hot):
        __, num_cluster = one_hot.shape
        cluster_indx = range(num_cluster)
        cluster_list = np.matmul(one_hot, cluster_indx)
        return cluster_list

    def fit(self, x):
        self.x = x
        R = self.initialization()
        for i in range(self.max_iteration):
            centroids = self.get_centroid(R)
            R_new = self.get_cluster(centroids)
            change = (R - R_new)
            if change.any() is not True:
                R = R_new
        cluster_list = self.onehot_to_list(R)
        return cluster_list, centroids

#
# generate sample points
#
from sklearn.datasets import make_blobs
from pandas import DataFrame
import matplotlib.pyplot as plt

X_train, _ = make_blobs(n_samples=500, centers=3, n_features=2, random_state=20)
df = DataFrame(dict(x=X_train[:,0], y=X_train[:,1]))
fig, ax = plt.subplots(figsize=(8,8))
df.plot(ax=ax, kind='scatter', x='x', y='y')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.show()

#
# fit
#
k_mean = KMean(max_iteration=5, k=3)
print(k_mean.__dict__)
cluster_list, centroids = k_mean.fit(X_train)
print(cluster_list)
print(centroids)

#
# plot result
#
cdict = {0: 'blue', 1: 'green', 2: 'red'}
fig, ax = plt.subplots()
for g in np.unique(cluster_list):
    ix = np.where(cluster_list == g)
    ax.scatter(X_train[ix, 0], X_train[ix, 1], c=cdict[g], label=g)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='^')
ax.legend()
plt.show()
