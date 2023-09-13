import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Step (a)
w0 =  np.random.uniform(-1/4, 1/4, 1)[0]

# Step (b)
w1 =  np.random.uniform(-1, 1, 1)[0]

# Step (c)
w2 =  np.random.uniform(-1, 1, 1)[0]

# Step (d)
S = np.random.uniform(-1, 1, (100, 2))

# Step (e) & (f)
W_tb = np.array([w0, w1, w2])
X = np.insert(S, 0, np.ones(100), axis=1)
Y = X @ W_tb.T
T_tb = Y >= 0

# Step (g)
for i in range(S.shape[0]):
    if T_tb[i]:
        c = 'b'
    else:
        c = 'r'
    plt.scatter(S[i, 0], S[i, 1], color=c, s=8)
x2 = lambda x1: ((w0 + w1 * x1) / -w2)
x1 = np.arange(-1, 2)
plt.plot(x1, x2(x1), color='g', label='Boundary')
plt.ylim(-1, 1)
plt.legend(title='Classes', loc="upper left")
plt.grid()
plt.savefig('./02-651340543-NEGRO-plot-01.jpg')

# Step (h)
# Hyperparameter
n = 1
epoch_number = 100

# Parameters
W = np.random.uniform(-1, 1, 3)

# History Trace
history_miss = list()

# Online Version
for epoch in range(epoch_number):
    T = np.heaviside(X @ W.T, 0)
    history_miss.append(np.logical_and(T, np.logical_not(T_tb)).sum())
    if history_miss[-1] == 0:
        break
    for index, x in enumerate(X):
        y = np.heaviside(x @ W.T, 0)
        if T_tb[index] == 0 and y == 1:
            W = W - n * x
        elif T_tb[index] == 1 and y == 0:
            W = W + n * x

# Step (i)
plt.clf()
plt.plot(history_miss, label='History')
plt.title("Curve of missprediction during epoches")
plt.grid()
plt.savefig('./02-651340543-NEGRO-plot-02.jpg')
