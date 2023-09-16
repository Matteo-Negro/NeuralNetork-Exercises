import copy
import numpy as np
import matplotlib.pyplot as plt

def PTA(W_1: list, S: list, T_tb: list) -> None:
    # Hyperparameter
    N = [1, 10, 0.1]
    epoch_number = 1000 # Just in case

    # History Trace
    history_miss = dict()

    for n in N:
        W = copy.deepcopy(W_1)
        # Online Version
        history_miss[n] = list()
        for _ in range(epoch_number):
            T = np.heaviside(X @ W.T, 0)
            history_miss[n].append(np.logical_xor(T, T_tb).sum())
            if history_miss[n][-1] == 0:
                break
            for index, x in enumerate(X):
                y = np.heaviside(x @ W.T, 0)
                if T_tb[index] == 0 and y == 1:
                    W = W - n * x
                elif T_tb[index] == 1 and y == 0:
                    W = W + n * x
        print(f'Weights for n = {n} and # of samples = {S.shape[0]} -> {W}')
        print(f'Initial number of misclassifications: {history_miss[n][0]}')
          
        # Plotting the history 
        plt.plot(history_miss[n])
        plt.title(f'Curve of misprediction during epoches (n = {n}, # of sample = {S.shape[0]})')
        plt.xlabel('Number of epochs')
        plt.ylabel('Number of misclassification')
        plt.grid()
        plt.savefig(f'./02-651340543-NEGRO-history-{S.shape[0]}samples-eta{n}.png', dpi=400, bbox_inches="tight")
        plt.clf()

        # Plotting the results with respect to the true value
        plt.scatter(S[T_tb, 0], S[T_tb, 1], color='r', s=8, label='Positive Class')
        plt.scatter(S[np.logical_not(T_tb), 0], S[np.logical_not(T_tb), 1], color='b', s=8, label='Negative Class')

        x1 = np.arange(-1, 2)
        x2 = lambda x1: ((w0 + w1 * x1) / -w2)
        plt.plot(x1, x2(x1), color='m', label='Boundary')

        y_hat = lambda x1: ((W[0] + W[1] * x1) / -W[2])
        plt.plot(x1, y_hat(x1), '--', color='c', label='PredBoundary')

        # Fancy plotting
        plt.plot([0,0], [-1,1], color='black', linestyle='-')
        plt.plot([-1,1], [0,0], color='black', linestyle='-')
        plt.legend(loc="upper left")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.savefig(f'./02-651340543-NEGRO-pred-{S.shape[0]}samples-eta{n}.png', dpi=400, bbox_inches="tight")
        plt.clf()
        

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(1)

    # Sampling the truth base weight
    w0 =  np.random.uniform(-1/4, 1/4, 1)[0]
    w1 =  np.random.uniform(-1, 1, 1)[0]
    w2 =  np.random.uniform(-1, 1, 1)[0]

    # Sampling the dataset
    S_100 = np.random.uniform(-1, 1, (100, 2))

    # Plotting the dataset
    plt.scatter(S_100[:, 0], S_100[:, 1], s=8)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.plot([0,0], [-1,1], color='black', linestyle='-')
    plt.plot([-1,1], [0,0], color='black', linestyle='-')
    plt.grid()
    plt.savefig('./02-651340543-NEGRO-dataset-100samples.png', dpi=400, bbox_inches="tight")
    plt.clf()

    # Obtaining the labels
    W_tb = np.array([w0, w1, w2])
    X = np.insert(S_100, 0, np.ones(100), axis=1)
    Y = X @ W_tb.T
    T_tb = Y >= 0

    # Plotting the classified dataset
    plt.scatter(S_100[T_tb, 0], S_100[T_tb, 1], color='r', s=8, label='Positive Class')
    plt.scatter(S_100[np.logical_not(T_tb), 0], S_100[np.logical_not(T_tb), 1], color='b', s=8, label='Negative Class')

    x2 = lambda x1: ((w0 + w1 * x1) / -w2)
    x1 = np.arange(-1, 2)
    plt.plot(x1, x2(x1), color='m', label='Boundary')

    # Fancy plot
    plt.plot([0,0], [-1,1], color='black', linestyle='-')
    plt.plot([-1,1], [0,0], color='black', linestyle='-')
    plt.legend(loc="upper left")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.savefig('./02-651340543-NEGRO-dataset-100samples-classified.png', dpi=400, bbox_inches="tight")
    plt.clf()

    # Parameters
    W_1 = np.random.uniform(-1, 1, 3)

    PTA(W_1, S_100, T_tb)

    S_1000 = np.random.uniform(-1, 1, (1000, 2))

    # Plotting the dataset
    plt.scatter(S_1000[:, 0], S_1000[:, 1], s=8)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.plot([0,0], [-1,1], color='black', linestyle='-')
    plt.plot([-1,1], [0,0], color='black', linestyle='-')
    plt.grid()
    plt.savefig('./02-651340543-NEGRO-dataset-1000samples.png', dpi=400, bbox_inches="tight")
    plt.clf()

    # Obtaining the labels
    W_tb = np.array([w0, w1, w2])
    X = np.insert(S_1000, 0, np.ones(1000), axis=1)
    Y = X @ W_tb.T
    T_tb = Y >= 0

    # Plotting the classified dataset
    plt.scatter(S_1000[T_tb, 0], S_1000[T_tb, 1], color='r', s=8, label='Positive Class')
    plt.scatter(S_1000[np.logical_not(T_tb), 0], S_1000[np.logical_not(T_tb), 1], color='b', s=8, label='Negative Class')

    x2 = lambda x1: ((w0 + w1 * x1) / -w2)
    x1 = np.arange(-1, 2)
    plt.plot(x1, x2(x1), color='m', label='Boundary')

    # Fancy plot
    plt.plot([0,0], [-1,1], color='black', linestyle='-')
    plt.plot([-1,1], [0,0], color='black', linestyle='-')
    plt.legend(loc="upper left")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.savefig('./02-651340543-NEGRO-dataset-1000samples-classified.png', dpi=400, bbox_inches="tight")
    plt.clf()

    PTA(W_1, S_1000, T_tb)
