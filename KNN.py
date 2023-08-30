import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        unique_labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(label_counts)]
        return most_common_label


#
# Example usage
#
X_train = np.array([[1, 2], [2, 3], [4, 5], [6, 7]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[3, 4]])

knn = KNN(k=2)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
print(prediction)

#
# plot the result
#
import matplotlib.pyplot as plt
cdict = {1: 'blue', 0: 'green'}

fig, ax = plt.subplots()
for g in np.unique(y_train):
    ix = np.where(y_train == g)
    ax.scatter(X_train[ix, 0], X_train[ix, 1], c=cdict[g], label=g)
plt.plot(X_test[0,0], X_test[0,1], c=cdict[prediction[0]], marker='*')
ax.legend()
plt.show()
