import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def logistic_regression(x, y, learning_rate, max_iteration):
    __, num_features = x.shape
    m = np.random.randn(1, num_features) * 0.01
    beta = np.zeros((1, 1))
    for i in range(max_iteration):
        h = sigmoid(np.matmul(x, m.T) + beta)
        loss = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=0, keepdims=True) / (-len(y))
        print(loss)
        grad_m = np.sum(x * (h - y), axis=0, keepdims=True) / len(y)
        grad_beta = np.sum((h - y), axis=0, keepdims=True) / len(y)
        m = m - grad_m * learning_rate
        beta = beta - grad_beta * learning_rate
    for i in range(len(x)):
        y_hat = 1 * (h > 0.5)

    return m, beta, loss, y_hat


a = 2 * np.random.normal(loc=0, scale=1, size=[100, 3])
b_h = sigmoid(4 + a[:, 0] + 2 * a[:, 1] + 3 * a[:, 2] + np.random.randn(100, 1).T)
b = 1 * (b_h > 0.5)
b = b.T
a.shape
b.shape
m, beta, loss, b_hat = logistic_regression(a, b, learning_rate=0.1, max_iteration=1000)
m, beta  # m = [1,2,3], beta = 4

import matplotlib.pyplot as plt
plt.hist(b_hat-b)
plt.show()


