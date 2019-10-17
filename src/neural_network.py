from .model import Model
import numpy as np


class NeuralNetwork(Model):

    def __init__(self, w, r):
        """Initialize the network's parameters.

        z: list
            Activation values. List of np.array objects, one for each layer. Without bias.

        :param w: list
            Weights. List of np.array objects, one for each layer. With bias.

        :param r: float
            Regularization parameter.
        """
        self.__z = [np.ones((weights.shape[1] + 1, 1)) for weights in w]
        self.__w = w
        self.__r = r

    def fit(self, x, y) -> None:
        pass

    def predict(self, x) -> list:
        pass
