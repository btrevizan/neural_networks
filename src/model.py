class Model:

    def fit(self, x, y) -> None:
        """Fit the model.

        :param x: matrix (numpy.ndarray)
            Training data.

        :param y: list
            Labels for training data.
        """
        raise NotImplementedError

    def predict(self, x) -> list:
        """Given a list of instances, predict a class for each one.

        :param x: matrix (numpy.ndarray)
            Instances to be labeled.

        :return: list
            List of labels.
        """
        raise NotImplementedError
