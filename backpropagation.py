import numpy as np
import sys

network, initial_weights, dataset = sys.argv[1:]

with open(network, 'r') as netfile:
	lines = netfile.readlines()
	regularization = float(lines[0])
	number_of_neurons = [int(i) for i in lines[1:]]  # number of neurons on each layer.

all_layers_weights = []

with open(initial_weights, 'r') as weightsfile:
	lines = weightsfile.readlines()
	for layer in lines:
		# for each layer:
		layer_neurons = []
		for neuron in layer.split(';'):
			# for each neuron:
			weights = [float(i) for i in neuron.strip().split(',')]
			layer_neurons += weights
		layer_neurons_nparray = np.asarray(layer_neurons, dtype=np.float32)
		all_layers_weights.append(layer_neurons_nparray)

with open(dataset, 'r') as datasetfile:
	lines = datasetfile.readlines()
	x = []
	y = []
	for instance in lines:
		# for each instance:
		attrib_values, outputs = instance.split(';')
		attrib_values = [float(i) for i in attrib_values.strip().split(',')]
		outputs = [float(i) for i in outputs.strip().split(',')]
		x += [attrib_values]
		y += [outputs]
	x = np.asarray(x, dtype=np.float32)
	y = np.asarray(y, dtype=np.float32)

