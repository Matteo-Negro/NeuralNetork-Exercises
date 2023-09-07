import numpy as np
import matplotlib.pyplot as plt

# Drawing 1000 points uniformly at random from the square [−2, 2)^2
x = np.random.uniform(-2, 2, 1000)
y = np.random.uniform(-2, 2, 1000)

# Merging the data in a matrix (2,1000)
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)
X = np.hstack((x, y)).T

# Fixing the Weights and the Biases 
W1 = np.array([[ 1, -1], [-1, -1], [-1,  0]])
b1 = np.array([[1, 1, 0]]).T

W2 = np.array([[1, 1, -1]])
b2 = np.array([[-1.5]]).T

# Feeding each point to the Neural Network
Z = np.heaviside(W2 @ (np.heaviside(W1 @ X + b1, 0)) + b2, 0)

# Plotting the outputs (blue pts -> z == 0, red pts -> o.w.)
for i in range(X.shape[1]):
    if Z[0][i] == 0:
        c = 'b'
    else:
        c = 'r'
    plt.scatter(X[0][i], X[1][i], color=c, s=4)
plt.plot([ 0, 0], [-2, 2], color='black', linewidth=2)
plt.plot([-2, 2], [ 0, 0], color='black', linewidth=2)

# Estimating the boudaries that defines the positive regions (red)
plt.plot([ 0, 2], [ 1,-1], '--', color='green', linewidth=2)
plt.plot([ 0, 0], [1, -2], '--', color='green', linewidth=2)

# Saving the plot
plt.savefig('./01-651340543-NEGRO-plot.jpg')