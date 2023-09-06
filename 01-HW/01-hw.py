import numpy as np
import matplotlib.pyplot as plt

zeros = np.zeros(1000)

x = np.random.uniform(-2, 2, 1000)
y = np.random.uniform(-2, 2, 1000)

x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

X = np.hstack((x, y)).T

W1 = np.array([[ 1, -1], [-1, -1], [-1,  0]])
b1 = np.array([[1, 1, 0]]).T

W2 = np.array([[1, 1, 1]])
b2 = np.array([[-1.5]]).T

Z = np.heaviside(W2 @ (np.heaviside(W1 @ X + b1, zeros)) + b2, zeros)

for i in range(X.shape[1]):
    if Z[0][i] == 0:
        c = 'b'
    else:
        c = 'r'
    plt.scatter(X[0][i], X[1][i], color=c)

plt.savefig('plot.jpg')