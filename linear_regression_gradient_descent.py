import numpy as np


def gradient_decent(x, y, learning_rate, max_iteration):
    __, num_features = x.shape
    m = np.random.randn(1, num_features) * 0.01
    beta = np.zeros((1, 1))
    for i in range(max_iteration):
        y_hat = np.matmul(x, m.T) + beta
        loss = np.sum((y_hat - y) ** 2, axis=0, keepdims=True) / (2 * len(y))
        print(loss)
        grad_m = np.sum(x * (y_hat - y), axis=0, keepdims=True) / len(y)
        grad_beta = np.sum((y_hat - y), axis=0, keepdims=True) / len(y)
        m = m - grad_m * learning_rate
        beta = beta - grad_beta * learning_rate

    return m, beta, loss, y_hat


a = 2 * np.random.rand(100, 3)
b = 4 + a[:, 0] + 2 * a[:, 1] + 3 * a[:, 2] + np.random.randn(100, 1).T
b = b.T
a.shape
b.shape
m, beta, loss, b_hat = gradient_decent(a, b, learning_rate=0.1, max_iteration=1000)
m, beta  # m = [1,2,3], beta = 4

import matplotlib.pyplot as plt

plt.plot(b_hat-b)
plt.show()





