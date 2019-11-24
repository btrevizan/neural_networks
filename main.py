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
    n_classes = y.shape[1]

    rs = np.random.RandomState(25)

    defaults = get_defaults(x, n_classes, rs)

    # with open(path.join(results_dt_path, 'best.csv'), 'r') as best:
    #     content = best.read()
    #
    # bests = {}
    # for line in content.split('\n'):
    #     b = line.split(',')
    #     if len(b) > 1:
    #         bests[b[0]] = float(b[1])
    #
    # bests['best_neuron'] = int(bests['best_neuron'])
    # bests['best_layer'] = int(bests['best_layer'])
    #
    # r = bests['best_r']
    # a = bests['best_alpha']
    # bs = int(bests['batch_size'] * x.shape[0])

    r = defaults['default_regularization']
    a = defaults['default_alpha']
    bs = defaults['default_bs']

    # Number of folds to get 20% test set
    k = int(np.ceil(x.shape[0] / (x.shape[0] * 0.2)))

    # Beta
    costs_df = None
    for b in [0, 0.25, 0.5, 0.75, 0.99]:
        w = [weights(defaults['default_n_neurons'], x.shape[1] + 1, rs)]
        w += [weights(defaults['default_n_neurons'], defaults['default_n_neurons'] + 1, rs) for _ in range(defaults['default_n_layers'] - 1)]
        w += [weights(n_classes, defaults['default_n_neurons'] + 1, rs)]

        model = NeuralNetwork(w, r, a, b)

        print("Evaluating costs for {} with beta {}...".format(args.evaluate, b))
        costs = evaluate_cost(model, x, y, k, bs, rs)
        print("Done.")

        if costs_df is None:
            costs_df = DataFrame({'n_instance': costs['n_instance'], str(b): costs['cost']})
        else:
            costs_df = concat([costs_df, DataFrame({str(b): costs['cost']})], axis=1)

    costs_df.to_csv(path.join(results_dt_path, 'costs_beta.csv'), index=False)
    b = defaults['default_beta']

    # Batch size
    costs_df = None
    for bs in [1/16, 1/8, 1/4, 1/2, 1]:
        w = [weights(defaults['default_n_neurons'], x.shape[1] + 1, rs)]
        w += [weights(defaults['default_n_neurons'], defaults['default_n_neurons'] + 1, rs) for _ in range(defaults['default_n_layers'] - 1)]
        w += [weights(n_classes, defaults['default_n_neurons'] + 1, rs)]

        model = NeuralNetwork(w, r, a, b)

        print("Evaluating costs for {} with batch size {}...".format(args.evaluate, bs))
        costs = evaluate_cost(model, x, y, k, bs, rs)
        print("Done.")

        if costs_df is None:
            costs_df = DataFrame({'n_instance': costs['n_instance'], str(bs): costs['cost']})
        else:
            costs_df = concat([costs_df, DataFrame({str(bs): costs['cost']})], axis=1)

    costs_df.to_csv(path.join(results_dt_path, 'costs_batchsize.csv'), index=False)
    bs = defaults['default_bs']

    # Alpha
    costs_df = None
    for a in [0.001, 0.01, 0.1, 0.2, 0.5, 0.9, 0.99]:
        w = [weights(defaults['default_n_neurons'], x.shape[1] + 1, rs)]
        w += [weights(defaults['default_n_neurons'], defaults['default_n_neurons'] + 1, rs) for _ in range(defaults['default_n_layers'] - 1)]
        w += [weights(n_classes, defaults['default_n_neurons'] + 1, rs)]

        model = NeuralNetwork(w, r, a, b)

        print("Evaluating costs for {} with alpha {}...".format(args.evaluate, a))
        costs = evaluate_cost(model, x, y, k, bs, rs)
        print("Done.")

        if costs_df is None:
            costs_df = DataFrame({'n_instance': costs['n_instance'], str(a): costs['cost']})
        else:
            costs_df = concat([costs_df, DataFrame({str(a): costs['cost']})], axis=1)

    costs_df.to_csv(path.join(results_dt_path, 'costs_alpha.csv'), index=False)


def init(args):
    results_dt_path = results_path.format(args.dataset)
    x, y = load_data(args.dataset)

    rs = np.random.RandomState(25)

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
    defaults['default_bs'] = 1 / 4

    return defaults


def optimize_batchsizes(results_dt_path, x, y, rs, defaults):
    batch_sizes = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
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


def optimize_nlayers(results_dt_path, x, y, rs, defaults):
    n_layers = range(1, 21)
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


def optimize_nneurons(results_dt_path, x, y, rs, defaults):
    n_neurons = list(range(1, 100, 5)) + [100, 200]
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


def optimize_regularization(results_dt_path, x, y, rs, defaults):
    r_param = [0.01, 0.1, 1, 10, 100, 1000, 10000]
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
