from src.utils import load_benchmark, load_network, load_weights
from src.neural_network import NeuralNetwork
from copy import deepcopy
import numpy as np
import argparse

precision = 5


def main(args):
	np.set_printoptions(precision=precision)

	network_path = args.files[0]
	initial_weights_path = args.files[1]
	dataset_path = args.files[2]

	r, n_inputs, n_neurons, n_outputs = load_network(network_path)
	initial_weights = load_weights(initial_weights_path)
	x, y = load_benchmark(dataset_path)
	epsilon = 0.0000010000
	n = x.shape[0]

	model = NeuralNetwork(deepcopy(initial_weights), r, 0.99, 0)

	print("Parâmetro de regularização lambda={}\n".format(round(r, 3)))
	print("Inicializando rede com a seguinte estrutura de neurônios por camadas: {}\n".format([n_inputs] + n_neurons + [n_outputs]))

	for i in range(len(initial_weights)):
		print("Theta{} inicial (pesos de cada neurônio, incluindo bias, armazenados nas linhas):\n{}".format(i + 1, str_matrix(initial_weights[i], '\t')))

	print("Conjunto de treinamento")
	for i in range(x.shape[0]):
		print("\tExemplo {}".format(i + 1))
		print("\t\tx: {}".format(x[i, :]))
		print("\t\ty: {}".format(y[i, :]))

	print("\n--------------------------------------------")
	print("Calculando erro/custo J da rede")

	for i in range(x.shape[0]):
		print("\tProcessando exemplo de treinamento {}".format(i + 1))
		print("\tPropagando entrada {}".format(x[i, :]))

		f = model.forward_propagation(x[i, :])
		cost = model.cost_x(y[i, :], f)

		print("\t\ta1: {}\n".format(model.a[0]))

		for l in range(1, model.n_layers + 1):
			print("\t\tz{}: {}".format(l + 1, model.z[l]))
			print("\t\ta{}: {}\n".format(l + 1, model.a[l]))

		print("\t\tf(x[{}]): {}".format(i + 1, f))

		print("\tSaida predita para o exemplo {}: {}".format(i + 1, f))
		print("\tSaida esperada para o exemplo {}: {}".format(i + 1, y[i, :]))
		print("\tJ do exemplo {}: {}\n".format(i + 1, cost))

	print("J total do dataset (com regularizacao): {}\n".format(model.cost(x, y)))

	print("\n--------------------------------------------")
	print("Rodando backpropagation")

	for i in range(n):
		print("\tCalculando gradientes com base no exemplo {}".format(i + 1))

		model.g = [np.zeros(model.w[i].shape) for i in range(model.n_layers)]
		model.m = [np.zeros(model.w[i].shape) for i in range(model.n_layers)]

		pred = model.forward_propagation(x[i, :])
		model.d[model.last_layer] = pred - y[i, :]
		model.update_deltas(x[i, :])

		for d in range(model.last_layer, -1, -1):
			print("\t\tdelta{}: {}".format(d + 2, model.d[d]))

		model.accumulate_gradients()

		for t in range(model.last_layer, -1, -1):
			print("\t\tGradientes de Theta{} com base no exemplo {}:\n{}".format(t + 1, i + 1, str_matrix(model.g[t], '\t\t\t')))

	print("\tDataset completo processado. Calculando gradientes regularizados")

	model.final_gradients(n)

	for t in range(model.n_layers):
		print("\t\tGradientes finais para Theta{} (com regularizacao):\n{}".format(t + 1, str_matrix(model.g[t], '\t\t\t')))

	print("\n--------------------------------------------")
	print("Rodando verificacao numerica de gradientes (epsilon={})".format(epsilon))

	backprop_gradients = deepcopy(model.g)
	model.g = [np.zeros(model.w[i].shape) for i in range(model.n_layers)]

	for t in range(model.n_layers):

		for i in range(model.g[t].shape[0]):
			for j in range(model.g[t].shape[1]):
				w = model.w[t][i, j]

				model.w[t][i, j] = w + epsilon
				c1 = model.cost(x, y)

				model.w[t][i, j] = w - epsilon
				c2 = model.cost(x, y)

				model.g[t][i, j] += (c1 - c2) / (2 * epsilon)
				model.w[t][i, j] = w

		print("\tGradiente numerico de Theta{}:\n{}".format(t + 1, str_matrix(model.g[t], '\t\t')))

	print("\n--------------------------------------------")
	print("Verificando corretude dos gradientes com base nos gradientes numericos:")
	for t in range(model.n_layers):
		errors = np.sum(np.abs(model.g[t] - backprop_gradients[t]))
		print("\tErro entre gradiente via backprop e gradiente numerico para Theta{}: {}".format(t + 1, errors))


def str_matrix(m, prefix=''):
	res = ""
	for i in range(m.shape[0]):
		res += prefix

		for j in range(m.shape[1]):
			res += "{0:.5f} ".format(round(m[i, j], precision))

		res += "\n"

	return res


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('files', metavar='filepath', nargs=3, help="1 - Network configuration file path. 2 - Initial weights configuration file path. 3 - Dataset file path.")

	parsed_args = parser.parse_args()

	main(parsed_args)


