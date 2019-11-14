from src.neural_network import NeuralNetwork
from pandas import DataFrame
from src.utils import *
from os import path
import numpy as np
import argparse


def optimize(args):
    results_dt_path = results_path.format(args.optimize)

    x, y = load_data(args.optimize)

    classes = np.unique(y)
    n_classes = len(classes)

    rs = np.random.RandomState(22)

    batch_sizes = [5, 50, 100, 200, x.shape[0]]
    n_layers = [1, 2, 3, 5, 10, 25, 50, 100]
    n_neurons = [1, 2, 3, 5, 10, 15]
    r_param = [0.01, 0.1, 0.5, 0.9]
    alphas = [0.003, 0.01, 0.1, 0.2, 0.5]
    betas = [0, 0.25, 0.50, 0.75, 0.99]

    default_n_layers = 2
    default_n_neurons = 5

    default_weights = [weights(x.shape[1] + 1, default_n_neurons, rs)]
    default_weights += [weights(default_n_neurons + 1, default_n_neurons, rs) for _ in range(default_n_layers)]
    default_weights += [weights(n_classes, default_n_neurons, rs)]

    default_regularization = 0.5
    default_alpha = 0.1
    default_beta = 0.5
    default_bs = 100

    print("Optimizing number of layers...")
    layer_metrics = {}
    for n in n_layers:
        w = [weights(x.shape[1] + 1, default_n_neurons, rs)]
        w += [weights(default_n_neurons + 1, default_n_neurons, rs) for _ in range(n)]
        w += [weights(n_classes, default_n_neurons, rs)]

        model = NeuralNetwork(w, default_regularization, default_alpha, default_beta)
        layer_metrics[n] = cross_validate(model, x, y, k_fold, default_bs, rs)
        print("\tNumber of layers = {} with {} mean F1-Score.".format(n, np.mean(layer_metrics[n])))

    layer_df = DataFrame(layer_metrics)
    layer_df.to_csv(path.join(results_dt_path, 'cv_layers.csv'), header=True, index=False)

    print("Optimizing number of neurons on hidden layers...")
    neuron_metrics = {}
    for n in n_neurons:
        w = [weights(x.shape[1] + 1, n, rs)]
        w += [weights(n + 1, n, rs) for _ in range(default_n_layers)]
        w += [weights(n_classes, n, rs)]

        model = NeuralNetwork(w, default_regularization, default_alpha, default_beta)
        neuron_metrics[n] = cross_validate(model, x, y, k_fold, default_bs, rs)
        print("\tNumber of neurons = {} with {} mean F1-Score.".format(n, np.mean(neuron_metrics[n])))

    neuron_df = DataFrame(neuron_metrics)
    neuron_df.to_csv(path.join(results_dt_path, 'cv_neurons.csv'), header=True, index=False)

    print("Optimizing the regularization parameter...")
    r_metrics = {}
    for r in r_param:
        model = NeuralNetwork(default_weights, r, default_alpha, default_beta)
        r_metrics[r] = cross_validate(model, x, y, k_fold, default_bs, rs)
        print("\tR = {} with {} mean F1-Score.".format(r, np.mean(r_metrics[r])))

    r_df = DataFrame(r_metrics)
    r_df.to_csv(path.join(results_dt_path, 'cv_regularization.csv'), header=True, index=False)

    alpha_metrics = {}
    print("Optimizing the learning rate...")
    for a in alphas:
        model = NeuralNetwork(default_weights, default_regularization, a, default_beta)
        alpha_metrics[a] = cross_validate(model, x, y, k_fold, default_bs, rs)
        print("\tAlpha = {} with {} mean F1-Score.".format(a, np.mean(alpha_metrics[a])))

    alpha_df = DataFrame(alpha_metrics)
    alpha_df.to_csv(path.join(results_dt_path, 'cv_alpha.csv'), header=True, index=False)

    batch_metrics = {}
    print("Optimizing the batch size...")
    for bs in batch_sizes:
        model = NeuralNetwork(default_weights, default_regularization, default_alpha, default_beta)
        batch_metrics[bs] = cross_validate(model, x, y, k_fold, bs, rs)
        print("\tBatch size = {} with {} mean F1-Score.".format(bs, np.mean(batch_metrics[bs])))

    bs_df = DataFrame(batch_metrics)
    bs_df.to_csv(path.join(results_dt_path, 'cv_batchsize.csv'), header=True, index=False)

    beta_metrics = {}
    print("Optimizing the beta...")
    for b in betas:
        model = NeuralNetwork(default_weights, default_regularization, default_alpha, b)
        beta_metrics[b] = cross_validate(model, x, y, k_fold, default_bs, rs)
        print("\tBeta = {} with {} mean F1-Score.".format(b, np.mean(beta_metrics[b])))

    beta_df = DataFrame(beta_metrics)
    beta_df.to_csv(path.join(results_dt_path, 'cv_beta.csv'), header=True, index=False)

    best_layer = layer_df.mean(axis=1).idxmax(axis=1)
    best_neuron = neuron_df.mean(axis=1).idxmax(axis=1)
    best_r = r_df.mean(axis=1).idxmax(axis=1)
    best_alpha = alpha_df.mean(axis=1).idxmax(axis=1)
    best_bs = bs_df.mean(axis=1).idxmax(axis=1)
    best_beta = beta_df.mean(axis=1).idxmax(axis=1)

    print("Best layer number: {}".format(best_layer))
    print("Best neuron number: {}".format(best_neuron))
    print("Best regularization: {}".format(best_r))
    print("Best alpha: {}".format(best_alpha))
    print("Best batch size: {}".format(best_bs))
    print("Best beta: {}".format(best_beta))

    with open(path.join(results_dt_path, 'best.csv'), 'w') as best:
        best.write('best_layer,best_neuron,best_r,best_alpha,batch_size,beta\n')
        best.write('{},{},{},{}\n'.format(best_layer, best_neuron, best_r, best_alpha, best_bs, best_beta))


def evaluate(args):
    results_dt_path = results_path.format(args.evaluate)

    x, y = load_data(args.evaluate)
    rs = np.random.RandomState(22)

    with open(path.join(results_dt_path, 'best.csv'), 'w') as best:
        lines = best.readlines()[1]

    bests = [float(b) for b in lines[:-1].split(',')]

    w = weights(int(bests[0]), rs)
    r = bests[1]
    a = bests[2]
    bs = int(bests[3])
    b = bests[5]

    model = NeuralNetwork(w, r, a, b)

    # Number of folds to get 30% test set
    k = np.ceil(x.shape[0] / (x.shape[0] * 0.3))

    print("Evaluating costs for {}...".format(args.evaluate))
    costs = evaluate_cost(model, x, y, k, bs, rs)
    print("Done.")

    costs_df = DataFrame(costs)
    mean_costs = costs_df.mean(axis=1)

    batch_sizes = mean_costs.columns.value
    mean_costs = mean_costs.values

    with open(path.join(results_dt_path, 'costs.csv'), 'w') as file:
        file.write("batch_size,mean_cost\n")
        for bs, c in zip(batch_sizes, mean_costs):
            file.write("{},{}\n".format(bs, c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--optimize',
                        dest='optimize',
                        required=False,
                        default=None,
                        help="Optimize the parameters for a given dataset.",
                        choices=['weights', 'regularization', 'alpha'])

    parser.add_argument('-e', '--evaluate',
                        dest='evaluate',
                        required=False,
                        default=None,
                        help="Evaluate a parameterized model for a given dataset.",
                        choices=datasets)

    parsed_args = parser.parse_args()

    if parsed_args.optimize:
        optimize(parsed_args)
    elif parsed_args.evaluate:
        evaluate(parsed_args)
    else:
        parser.print_help()
