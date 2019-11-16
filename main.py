from src.neural_network import NeuralNetwork
from pandas import DataFrame
from src.utils import *
from time import time
from os import path
import numpy as np
import argparse


def optimize(args):
    results_dt_path, x, y, rs = init(args)
    defaults = get_defaults(x, y.shape[1], rs)

    if args.optimize == 'batchsize':
        optimize_batchsizes(results_dt_path, x, y, rs, defaults)
    elif args.optimize == 'nlayers':
        optimize_nlayers(results_dt_path, x, y, rs, defaults)
    elif args.optimize == 'nneurons':
        optimize_nneurons(results_dt_path, x, y, rs, defaults)
    elif args.optimize == 'regularization':
        optimize_regularization(results_dt_path, x, y, rs, defaults)
    elif args.optimize == 'alpha':
        optimize_alpha(results_dt_path, x, y, rs, defaults)
    elif args.optimize == 'beta':
        optimize_beta(results_dt_path, x, y, rs, defaults)


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


def init(args):
    results_dt_path = results_path.format(args.dataset)
    x, y = load_data(args.dataset)

    rs = np.random.RandomState(22)

    return results_dt_path, x, y, rs


def get_defaults(x, n_classes, rs):
    defaults = {}

    defaults['default_n_layers'] = 2
    defaults['default_n_neurons'] = 5

    defaults['default_weights'] = [weights(defaults['default_n_neurons'], x.shape[1] + 1, rs)]
    defaults['default_weights'] += [weights(defaults['default_n_neurons'], defaults['default_n_neurons'] + 1, rs)
                                    for _ in range(defaults['default_n_layers'] - 1)]

    defaults['default_weights'] += [weights(n_classes, defaults['default_n_neurons'] + 1, rs)]

    defaults['default_regularization'] = 0.5
    defaults['default_alpha'] = 0.1
    defaults['default_beta'] = 0.5
    defaults['default_bs'] = 100

    return defaults


def optimize_batchsizes(results_dt_path, x, y, rs, defaults):
    batch_sizes = [5, 10, 25, 50, 100, 200, 300, x.shape[0]]
    batch_metrics = {}

    result_file = path.join(results_dt_path, 'cv_batchsize.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing the batch size...")
    start = int(time())

    for bs in batch_sizes:
        start_i = int(time())
        batch_metrics[bs] = run(x, y,
                                defaults['default_weights'],
                                defaults['default_regularization'],
                                defaults['default_alpha'],
                                defaults['default_beta'],
                                bs, rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in batch_metrics[bs]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(bs, elapsed, values))

        print("\tParam value = {} with {} mean F1-Score.".format(bs, np.mean(batch_metrics[bs])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())

    bs_df = DataFrame(batch_metrics)
    best_bs = bs_df.mean(axis=0).idxmax(axis=1)

    print("Best batch size: {}".format(best_bs))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('batch_size,{}\n'.format(best_bs))


def optimize_nlayers(results_dt_path, x, y, rs, defaults):
    n_layers = [1, 2, 3, 5, 10, 25, 50, 100, 150, 200, 250, 500, 1000]
    n_classes = y.shape[1]
    layer_metrics = {}

    result_file = path.join(results_dt_path, 'cv_nlayers.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing number of layers...")
    start = int(time())

    for n in n_layers:
        w = [weights(defaults['default_n_neurons'], x.shape[1] + 1, rs)]
        w += [weights(defaults['default_n_neurons'], defaults['default_n_neurons'] + 1, rs) for _ in range(n - 1)]
        w += [weights(n_classes, defaults['default_n_neurons'] + 1, rs)]

        start_i = int(time())
        layer_metrics[n] = run(x, y, w,
                               defaults['default_regularization'],
                               defaults['default_alpha'],
                               defaults['default_beta'],
                               defaults['default_bs'], rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in layer_metrics[n]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(n, elapsed, values))

        print("\tNumber of layers = {} with {} mean F1-Score.".format(n, np.mean(layer_metrics[n])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())

    layer_df = DataFrame(layer_metrics)
    best_layer = layer_df.mean(axis=0).idxmax(axis=1)

    print("Best layer number: {}".format(best_layer))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('best_layer,{}\n'.format(best_layer))


def optimize_nneurons(results_dt_path, x, y, rs, defaults):
    n_neurons = [1, 2, 3, 5, 10, 15, 25, 50, 100, 200, 400, 800, 1600]
    n_classes = y.shape[1]
    neuron_metrics = {}

    result_file = path.join(results_dt_path, 'cv_nneurons.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing number of neurons on hidden layers...")
    start = int(time())

    for n in n_neurons:
        w = [weights(n, x.shape[1] + 1, rs)]
        w += [weights(n, n + 1, rs) for _ in range(defaults['default_n_layers'])]
        w += [weights(n_classes, n + 1, rs)]

        start_i = int(time())
        neuron_metrics[n] = run(x, y, w,
                                defaults['default_regularization'],
                                defaults['default_alpha'],
                                defaults['default_beta'],
                                defaults['default_bs'], rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in neuron_metrics[n]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(n, elapsed, values))

        print("\tNumber of neurons = {} with {} mean F1-Score.".format(n, np.mean(neuron_metrics[n])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())

    neuron_df = DataFrame(neuron_metrics)
    best_neuron = neuron_df.mean(axis=0).idxmax(axis=1)

    print("Best neuron number: {}".format(best_neuron))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('best_neuron,{}\n'.format(best_neuron))


def optimize_regularization(results_dt_path, x, y, rs, defaults):
    r_param = [0.01, 0.1, 0.5, 0.9, 0.99]
    r_metrics = {}

    result_file = path.join(results_dt_path, 'cv_regularization.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing the regularization parameter...")
    start = int(time())

    for r in r_param:
        start_i = int(time())
        r_metrics[r] = run(x, y,
                           defaults['default_weights'], r,
                           defaults['default_alpha'],
                           defaults['default_beta'],
                           defaults['default_bs'], rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in r_metrics[r]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(r, elapsed, values))

        print("\tR = {} with {} mean F1-Score.".format(r, np.mean(r_metrics[r])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())

    r_df = DataFrame(r_metrics)
    best_r = r_df.mean(axis=0).idxmax(axis=1)

    print("Best regularization: {}".format(best_r))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('best_r,{}\n'.format(best_r))


def optimize_alpha(results_dt_path, x, y, rs, defaults):
    alphas = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99]
    alpha_metrics = {}

    result_file = path.join(results_dt_path, 'cv_alpha.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing the learning rate...")
    start = int(time())

    for a in alphas:
        start_i = int(time())
        alpha_metrics[a] = run(x, y,
                               defaults['default_weights'],
                               defaults['default_regularization'], a,
                               defaults['default_beta'],
                               defaults['default_bs'], rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in alpha_metrics[a]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(a, elapsed, values))

        print("\tAlpha = {} with {} mean F1-Score.".format(a, np.mean(alpha_metrics[a])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())

    alpha_df = DataFrame(alpha_metrics)
    best_alpha = alpha_df.mean(axis=0).idxmax(axis=1)

    print("Best alpha: {}".format(best_alpha))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('best_alpha,{}\n'.format(best_alpha))


def optimize_beta(results_dt_path, x, y, rs, defaults):
    betas = [0, 0.25, 0.50, 0.75, 0.99]
    beta_metrics = {}

    result_file = path.join(results_dt_path, 'cv_beta.csv')
    with open(result_file, 'w') as file:
        file.write("param_value,seconds_elapsed,it1,it2,it3,it4,it5,it6,it7,it8,it9,it10\n")

    print("Optimizing the beta...")
    start = int(time())

    for b in betas:
        start_i = int(time())
        beta_metrics[b] = run(x, y,
                              defaults['default_weights'],
                              defaults['default_regularization'],
                              defaults['default_alpha'], b,
                              defaults['default_bs'], rs)

        stop_i = int(time())

        elapsed = stop_i - start_i
        values = ','.join([str(v) for v in beta_metrics[b]])
        with open(result_file, 'a') as file:
            file.write("{},{},{}\n".format(b, elapsed, values))

        print("\tBeta = {} with {} mean F1-Score.".format(b, np.mean(beta_metrics[b])))
        print("\tTime elapsed: {} minutes".format((stop_i - start_i) // 60))

    stop = int(time())
    beta_df = DataFrame(beta_metrics)
    best_beta = beta_df.mean(axis=0).idxmax(axis=1)

    print("Best beta: {}".format(best_beta))
    print("Time elapsed: {} minutes".format((stop - start) // 60))

    with open(path.join(results_dt_path, 'best.csv'), 'a') as best:
        best.write('best_beta,{}\n'.format(best_beta))


def run(x, y, w, r, a, b, bs, rs):
    model = NeuralNetwork(w, r, a, b)
    return cross_validate(model, x, y, k_fold, bs, rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--optimize',
                        dest='optimize',
                        required=False,
                        default=None,
                        help="Optimize the parameters for a given dataset.",
                        choices=['batchsize', 'nlayers', 'nneurons', 'regularization', 'alpha', 'beta'])

    parser.add_argument('-e', '--evaluate',
                        dest='evaluate',
                        required=False,
                        default=None,
                        help="Evaluate a parameterized model for a given dataset.",
                        choices=datasets)

    parser.add_argument('-d', '--dataset',
                        dest='dataset',
                        required=False,
                        default=None,
                        help="Dataset for parameter optimization.",
                        choices=datasets)

    parsed_args = parser.parse_args()

    if parsed_args.optimize:
        optimize(parsed_args)
    elif parsed_args.evaluate:
        evaluate(parsed_args)
    else:
        parser.print_help()
