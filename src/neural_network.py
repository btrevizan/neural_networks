from .model import Model
from math import exp
import numpy as np


class NeuralNetwork(Model):

    def __init__(self, w, r, alpha, beta):
        """Initialize the network's parameters.

        a: list
            Activation values. List of np.array objects, one for each layer. With bias.

        :param w: list
            Weights. List of np.array objects, one for each layer. With bias.

        :param r: float
            Regularization parameter.

        :param alpha: float
            Learning rate.

        :param beta: float
            Moving average param.
        """
        self.alpha = alpha
        self.beta = beta
        self.epoch = 10

        self.w = w                                                    # weights
        self.r = r                                                    # lambda in regularization
        self.a = [np.mat([]) for _ in range(self.n_layers + 1)]       # activation values
        self.z = [np.mat([]) for _ in range(self.n_layers + 1)]       # values before g()
        self.d = [np.mat([]) for _ in range(self.n_layers)]           # deltas
        self.g = []                                                   # gradients
        self.m = []                                                   # moving average

    @property
    def n_layers(self):
        return len(self.w)

    @property
    def last_layer(self):
        return self.n_layers - 1

    def fit(self, x, y, batch_size) -> None:
        n = x.shape[0]
        b = np.ceil(n / batch_size)

        for _ in range(self.epoch):
            for i in range(int(b)):
                j = i * batch_size
                k = slice(j, j + batch_size)
                self.backward_propagation(x[k, :], y[k])

            if n - b * batch_size > 0:
                self.backward_propagation(x[b * batch_size:, :], y[b * batch_size:])

    def predict(self, x):
        predictions = []
        for i in range(x.shape[0]):
            prob_class = self.forward_propagation(x[i, :])
            pred_class = np.argmax(prob_class)
            predictions.append(pred_class)

        return np.array(predictions)

    def forward_propagation(self, x):
        """Forward propagation.

        :param x: np.array
            Instance values.

        :return: list
            list of outputs
        """
        self.a[0] = np.asmatrix(np.append([1], x))  # input with bias

        for l in range(1, len(self.a)):
            self.z[l] = np.matmul(self.w[l - 1], self.a[l - 1].T).T
            self.a[l] = np.asmatrix(np.append([1], self.sigmoid(self.z[l].T)))

        return self.a[-1][:, 1:]  # output without bias

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
            f = self.forward_propagation(x[i, :])
            cost = self.cost_x(y[i, :], f)
            j = j + cost

        j = j / n
        s = self.regularization(n)

        return j + s

    def cost_x(self, y, f):
        """Compute the cost function for a x[i].

        :param y:
            Expected output.

        :param f:
            Predicted output.

        :return: float
        """
        c = np.multiply(-y, np.log(f)) - np.multiply(1 - y, np.log(1 - f))
        return np.sum(c)

    def regularization(self, n):
        """Compute the regularization term.

        :param n: int
            Number of training examples.

        :return: float
        """
        s = [np.power(np.array(self.w[i][:, 1:]), 2) for i in range(self.n_layers)]
        s = np.sum([np.sum(s[i]) for i in range(self.n_layers)])
        s = (self.r / (2 * n)) * s
        return s

    @staticmethod
    def sigmoid(z):
        """Compute a vector sigmoid.

        :param z: np.array
        :return: np.array
        """
        return np.array(list(map(lambda x: 1.0 / (1.0 + exp(-x)), z)))

    def backward_propagation(self, x, y):
        """Backpropagation.

        :param x: np.array
            Instances.

        :param y: np.array
            Instances' classes.
        """
        n = x.shape[0]
        self.g = [np.zeros(self.w[i].shape) for i in range(self.n_layers)]
        self.m = [np.zeros(self.w[i].shape) for i in range(self.n_layers)]

        for i in range(n):
            # Deltas for output neurons
            pred = self.forward_propagation(x[i, :])
            self.d[self.last_layer] = pred - y[i, :]

            self.update_deltas(x[i, :])
            self.accumulate_gradients()

        self.final_gradients(n)

        # Update weights
        for k in range(self.last_layer, -1, -1):
            self.w[k] = self.w[k] - np.multiply(self.alpha, self.m[k])

    def update_deltas(self, x):
        """Update delta given a x[i].

        :param x: np.array
            Instance values.
        """
        # Deltas for hidden layers
        for k in range(self.last_layer, 0, -1):
            deltas = np.matmul(self.w[k].T, self.d[k].T)
            deltas = np.multiply(deltas.T, self.a[k])
            deltas = np.multiply(deltas, 1 - self.a[k])
            self.d[k - 1] = deltas[:, 1:]

    def accumulate_gradients(self):
        """Just accumulate gradients for mini-batch training."""
        for k in range(self.last_layer, -1, -1):
            self.g[k] = self.g[k] + np.matmul(self.d[k].T, self.a[k])

    def final_gradients(self, n):
        """Calculate gradients with regularization.

        :param n: int
            Number of instances (x.shape[0]).
        """
        for k in range(self.last_layer, -1, -1):
            p = np.multiply(self.w[k], self.r)  # regularization
            p[:, 0] = 0  # ignore bias

            self.g[k] = self.g[k] + p
            self.g[k] = np.multiply(self.g[k], 1 / n)
            self.m[k] = np.multiply(self.beta, self.m[k]) + self.g[k]  # momentum
