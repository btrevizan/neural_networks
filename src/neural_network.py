from .model import Model
from math import exp
import numpy as np


class NeuralNetwork(Model):

    def __init__(self, w, r):
        """Initialize the network's parameters.

        a: list
            Activation values. List of np.array objects, one for each layer. With bias.

        :param w: list
            Weights. List of np.array objects, one for each layer. With bias.

        :param r: float
            Regularization parameter.
        """
        self.__a = [np.array([]) for _ in range(len(w) + 1)]
        self.__z = [np.array([]) for _ in range(len(w) + 1)]
        self.__w = w
        self.__r = r

    @property
    def n_layers(self):
        return len(self.__w)

    def fit(self, x, y) -> None:
        pass

    def predict(self, x) -> list:
        predictions = [self.forward(x[i]) for i in range(len(x))]
        return predictions

    def forward(self, x):
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
            f = self.forward(x[i])
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
