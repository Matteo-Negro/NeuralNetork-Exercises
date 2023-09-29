import numpy as np
import copy
import matplotlib.pyplot as plt

from keras.datasets import mnist
# Ranodm seed for reproducibility
np.random.seed(1)

# Training Step
def NN(X: list, Y: list, eps: float, eta: float, print_rate=1, max_epoch=200) -> list:
    print(f'X.shape: {X.shape}')
    print(f'Y.shape: {Y.shape}')
    epoch = 0
    history = list()
    W = np.random.uniform(-1, 1, (Y.shape[0], X.shape[0]))
    print(f'W.shape: {W.shape}')
    while epoch < max_epoch:
        history.append(0)
        for i in range(X.shape[1]):
            v = W @ X[:, i:i+1]
            pred_value = np.argmax(v)

            if Y[pred_value, i] != 1:
                history[epoch] += 1

        if epoch % print_rate == 0:
            print(f'{epoch}: errors -> {history[epoch]}; % -> {history[epoch] / X.shape[1]}')
            
        epoch += 1

        if (history[epoch - 1] / X.shape[1]) <= eps:
            break

        for i in range(X.shape[1]):
            W = W + eta * ((Y[:, i:i+1] - np.heaviside(W @ X[:, i:i+1], 0))) @ X[:, i:i+1].T

    plt.plot(history, marker = 'o')
    plt.xlabel('Number of epochs')
    plt.ylabel('Number of misclassification')
    plt.grid()
    plt.savefig(f'./03-651340543-NEGRO-history-samples{X.shape[1]}-eta{eta}-tollerance{eps}.png', dpi=400, bbox_inches="tight")

    plt.clf()

    return W


def test(W: list, X: list, Y: list):
    errors = 0
    for i in range(X.shape[1]):
        v = np.heaviside(W @ X[:, i], 0)
        true_value = Y[:, i].tolist().index(1)
        if v[true_value] != 1:
            errors += 1
    print(f'Errors: {errors}; Percentage: {errors/Y.shape[1]:.2f}')


(train_X, train_y), (test_X, test_y) = mnist.load_data()

X_train = (np.array(train_X).reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])).T
T_train = np.zeros((10, X_train.shape[1]))
for i, n in enumerate(train_y):
    T_train[n][i] = 1

X_test = (np.array(test_X).reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])).T
T_test = np.zeros((10, X_test.shape[1]))
for i, n in enumerate(test_y):
    T_test[n][i] = 1

fig, axs = plt.subplots(2, 5, figsize = (20,6))
axs = axs.flatten()

for i in range(len(axs)):
    index = list(test_y).index(i)
    digit = test_X[index, :, :]
    axs[i].imshow(digit, cmap='gray')
    axs[i].axis('off')

plt.savefig(f'./03-651340543-NEGRO-dataset.png', dpi=400, bbox_inches="tight")
plt.clf()

# NNs trained with different dataset size, weight and eps
n = [50, 1000, 60000]
eps = [0, 0, 0.12]
for i in range(3):
    W = NN(copy.deepcopy(X_train[:, :n[i]]), copy.deepcopy(T_train[:, :n[i]]), eps[i], 1)
    test(W, X_test, T_test)

# NNs trained with different dataset size, weight, eta and eps
eps = [0.11, 0.12, 0.14]
eta = [0.001, 0.1, 10]
for i in range(3):
    W = NN(copy.deepcopy(X_train[:, :]), copy.deepcopy(T_train[:, :]), eps[i], eta[i], print_rate=10)
    test(W, X_test, T_test)