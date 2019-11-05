from pandas import read_csv, concat
from .metrics import Scorer
from math import ceil
from os import path
import numpy as np
import json


datasets = ['breast-cancer', 'ionosphere', 'pima', 'wine']


def stratified_split(y, n_splits, random_state):
    """Create k folds for k-fold cross validation.

    :param y: list
        Instances' label.

    :param n_splits: int
        Number of splits (folds).

    :param random_state: instance of numpy.random.RandomState
        Seed for random generator.

    :return: iterator
        Each element (a tuple) has the instances' index
        for training set and for the test set, respectively.
    """
    sety = sorted(set(y))
    classes_indexes = []

    for c in sety:
        indexes = list(random_state.permutation(np.where(y == c)[0]))
        length = ceil(len(indexes) / n_splits)
        classes_indexes.append((indexes, length))

    folds = []
    for i in range(n_splits):
        folds.append([])
        for indexes, n in classes_indexes:
            folds[i] += indexes[i * n:(i * n) + n]

    for i in range(n_splits):
        test = folds[i]
        train = [index for indexes in folds[0:i] + folds[i+1:] for index in indexes]

        yield train, test


def cross_validate(model, x, y, k, random_state):
    """Run a k-fold cross validation.

    :param model: object
        Object with a fit(x, y) and predict([x]) function.

    :param x: matrix
        Instance's attributes.

    :param y: list
        Instance's classes.

    :param k: int
        Number of folds.

    :param random_state: instance of numpy.random.RandomState
        Seed for random generator.

    :return: list
        Test metric for each fold.
    """
    metrics = []
    n_labels = len(np.unique(y))

    for train_i, test_i in stratified_split(y, k, random_state):
        x_train = x[train_i, :]
        y_train = y[train_i]

        x_test = x[test_i, :]
        y_test = y[test_i]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        scorer = Scorer(y_test, y_pred, n_labels)
        metrics.append(scorer.f1_score())

    return metrics


def load_data(dataset):
    """Load a dataset.

    :param dataset: str
        Dataset name. Options: 'breast_cancer', 'ionosphere', 'pima', 'wine'

    :return: tuple
        (x, y)
    """
    default_path = 'tests/datasets/'

    if dataset not in datasets:
        raise ValueError('{} does not exist. The option are breast_cancer, ionosphere, pima, wine.'.format(dataset))

    data_path = path.join(default_path, '{}.csv'.format(dataset))
    data = read_csv(data_path, header=0)

    x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

    return x, y


def load_network(network_path):
    """Load the networks parameters.

    :param network_path: str
        Relative/absolute path to file.

    :return: tuple
        lambda, number of inputs, number of neurons for each layer (list), number of outputs
    """
    with open(network_path, 'r') as file:
        values = [float(line) for line in file]

    lbd = values[0]
    n_inputs = values[1]
    layers = values[2:-1]
    n_outputs = values[-1]

    return lbd, n_inputs, layers, n_outputs


def load_weights(weights_path):
    """Load the network's weights.

    :param weights_path: str
        Relative/absolute path to file.

    :return: list
        List of np.array objects, one for each layer.
    """
    with open(weights_path, 'r') as file:
        weights = [np.array(np.mat(line)) for line in file]

    return weights


def load_benchmark(dataset_path):
    """Load the network's weights.

    :param dataset_path: str
        Relative/absolute path to file.

    :return: tuple
        x, y
    """
    x, y = [], []
    with open(dataset_path, 'r') as file:
        for line in file:
            content = line.split(';')
            x += [[float(i) for i in content[0].split(',')]]
            y += [float(i) for i in content[1].split(',')]

    x = np.array(x)
    return x, y


def __unique_counts(values):
    _, counts = np.unique(values, return_counts=True)
    return counts, sum(counts)
