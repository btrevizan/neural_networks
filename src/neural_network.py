from .model import Model
from math import exp
import numpy as np


class NeuralNetwork(Model):

    def __init__(self, w, r, alpha):
        """Initialize the network's parameters.

        a: list
            Activation values. List of np.array objects, one for each layer. With bias.

        :param w: list
            Weights. List of np.array objects, one for each layer. With bias.

        :param r: float
            Regularization parameter.

        :param alpha: float
            Learning rate.
        """
        self.__alpha = alpha
        self.__w = w                                                    # weights
        self.__r = r                                                    # lambda in regularization
        self.__a = [np.array([]) for _ in range(self.n_layers + 1)]     # activation values
        self.__z = [np.array([]) for _ in range(self.n_layers + 1)]     # values before g()
        self.__d = [np.array([]) for _ in range(self.n_layers + 1)]     # deltas
        self.__g = []                                                   # gradients

    @property
    def n_layers(self):
        return len(self.__w)

    def fit(self, x, y, batch_size) -> None:
        n = x.shape[0]
        b = np.ceil(n / batch_size)

        for i in range(int(b)):
            j = i * batch_size
            k = slice(j, j + batch_size)
            self.backward_propagation(x[k, :], y[k])

        if n - b * batch_size > 0:
            self.backward_propagation(x[b * batch_size:, :], y[b * batch_size:])

    def predict(self, x) -> list:
        predictions = [self.forward_propagation(x[i]) for i in range(len(x))]
        return predictions

    def forward_propagation(self, x):
        """Forward propagation.

        :param x: np.array
            Instance values.

        :return: list
            list of outputs
        """
        self.__a[0] = np.append([1], x)  # input with bias

        for l in range(1, self.n_layers):
            self.__z[l] = np.matmul(self.__w[l - 1], self.__a[l - 1])
            self.__a[l] = np.append([1], self.sigmoid(self.__z[l]))

        return self.__a[-1][1:]  # output without bias

    def cost(self, x, y):
        """Compute the cost function for each x[i].

        :param x: np.array
            Set of data.

        :param y: np.array
            Classes.

        :return: float
        """
        j = 0
        n = x.shape[0]

        for i in range(n):
            f = self.forward_propagation(x[i])
            cost = np.multiply(-y[i], np.log2(f)) - np.multiply(1 - y[i], np.log2(1 - f))
            j = j + np.sum(cost)

        j = j / n
        s = self.regularization(n)

        return j + s

    def regularization(self, n):
        """Compute the regularization term.

        :param n: int
            Number of training examples.

        :return: float
        """
        s = np.power(np.array(self.__w), 2)
        s = np.sum(s)
        s = (self.__r / (2 * n)) * s
        return s

    @staticmethod
    def sigmoid(z):
        """Compute a vector sigmoid.

        :param z: np.array
        :return: np.array
        """
        return np.array(map(lambda x: 1.0 / (1.0 + exp(-x)), z))

    def backward_propagation(self, x, y):
        """Backpropagation.

        :param x: np.array
            Instances.

        :param y: np.array
            Instances' classes.
        """
        n = x.shape[0]
        last_layer = self.n_layers - 1
        self.__g = [np.zeros((self.__w[i].shape[0], 1)) for i in range(self.n_layers + 1)]

        for i in range(n):
            # Deltas for output neurons
            pred = self.forward_propagation(x[i, :])
            self.__d[last_layer] = pred - y[i]

            # Deltas for hidden layers
            for k in range(last_layer - 1, 0, -1):
                deltas = np.matmul(self.__w[k].T, self.__d[k + 1])
                deltas = np.multiply(deltas, self.__a[k])
                deltas = np.multiply(deltas, 1 - self.__a[k])
                self.__d[k] = deltas[1:]

            # Accumulate gradients
            for k in range(last_layer - 1, -1, -1):
                self.__g[k] = self.__g[k] + np.matmul(self.__d[k], self.__a[k].T)

        # Final gradients
        for k in range(last_layer - 1, -1, -1):
            p = np.multiply(self.__w[k][:, 1:], self.__r)  # regularization
            self.__g[k] = np.multiply(1 / n, self.__g[k][1:] + p)

        # Update weights
        for k in range(last_layer - 1, -1, -1):
            self.__w[k] = self.__w[k] - np.multiply(self.__alpha, self.__g[k])
