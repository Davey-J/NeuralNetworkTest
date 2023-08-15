from ActivationFunctions import *
import numpy as np
rng = np.random.default_rng()


# implements a basic layer structure consisting of a set of values and an empty function to be returned when iterating
# through the layers in a network
class Layer:
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros(size)

    def calc_layer(self, previous_layer_data):
        raise NotImplementedError

    def __repr__(self):
        return str([self.activations])


# Implements a data layer, which is just an empty data array that on update stores the values given to it
class DataLayer(Layer):
    def __init__(self, size):
        super().__init__(size)

    def calc_layer(self, previous_layer_data):
        self.activations = np.array(previous_layer_data)
        return self.activations


# Implements a child of the data layer used to store the output, which returns the derivative of the loss function when
# the cost function is called
class OutputLayer(DataLayer):
    def __init__(self, size):
        super().__init__(size)
        self.result = 0
        self.expected = 0
        self.error = 0

    def output_error(self, results, expected):
        self.result = results
        self.expected = expected

    def calc_layer_costs(self):
        return (self.result - self.expected) * 2


# Implements a basic fully connected layer that performs feed forward calculations and backpropagation
class ComputeLayer(Layer):

    def __init__(self, init_weights, init_biases, node_count, activation_func):
        super().__init__(node_count)
        self.activation_function = activation_func
        self.weighted_inputs = np.zeros(self.size)
        self.weights = init_weights
        self.weight_errors = np.zeros(self.weights.shape)
        self.prev_size = self.weights.shape[1]
        self.biases = init_biases
        self.node_errors = np.ones(self.size)
        self.activation_derivatives = np.zeros(self.size)
        self.forward_costs = np.zeros(self.prev_size)
        self.batch_error_size = (1, self.weights.shape[0], self.weights.shape[1])
        self.batch_errors = np.zeros(self.batch_error_size)
        self.batch_index = 0

    def gen_batch_errors(self, batch_size):
        self.batch_error_size = (batch_size, self.weights.shape[0], self.weights.shape[1])
        self.batch_errors = np.zeros(self.batch_error_size)

    def calc_layer(self, previous_layer_data):
        self.weighted_inputs = np.dot(self.weights, previous_layer_data) + self.biases
        self.activations = self.layer_activation_func(self.weighted_inputs)
        return self.activations

    def layer_activation_func(self, x):
        if self.activation_function == 0:
            return x
        elif self.activation_function == 1:
            return relu(x)
        elif self.activation_function == 2:
            return sigmoid(x)
        elif self.activation_function == 3:
            return leaky_relu(x, 0.05)
        elif self.activation_function == 4:
            return softmax(x)

    def d_layer_activation_func(self, x):
        if self.activation_function == 0:
            return 1
        elif self.activation_function == 1:
            return d_relu(x)
        elif self.activation_function == 2:
            return d_sigmoid(x)
        elif self.activation_function == 3:
            return d_leaky_relu(x, 0.05)
        elif self.activation_function == 4:
            return d_softmax(x)

    def calculate_error(self, forward_layer, last_layer):
        self.activation_derivatives = self.d_layer_activation_func(self.weighted_inputs)
        self.node_errors = forward_layer.calc_layer_costs() * self.activation_derivatives
        self.weight_errors = self.weights * self.node_errors[:, None]
        last_layer = last_layer.activations
        self.batch_errors[self.batch_index] = last_layer * self.node_errors[:, None]
        self.batch_index += 1

    def calc_layer_costs(self):
        for x in range(self.prev_size):
            self.forward_costs[x] = np.sum(self.weight_errors[:, x])
        return self.forward_costs

    def update_layer(self, train_rate):
        error_activations = np.average(self.batch_errors, axis=0)
        self.weights -= error_activations * train_rate
        self.batch_index = 0


# Implements a hidden layer, which is identical to the basic compute layer
class HiddenLayer(ComputeLayer):
    def __init__(self, init_weights, init_biases, node_count, activation_func):
        super().__init__(init_weights, init_biases, node_count, activation_func)


# Implements a final layer, overriding the loss function in order to get useful outputs
class FinalLayer(ComputeLayer):
    def __init__(self, init_weights, init_biases, node_count, activation_func):
        super().__init__(init_weights, init_biases, node_count, 2)


# Implements a neural network consisting of a data only input layer, a set of hidden compute layers,
# a final compute layer and then a data only output layer
# Implements calculating the output, training the network in batches and testing the output against a dataset
class NeuralNetwork:
    # Layer array is an array of each layer and the size ([2,4,2] is a NN with 3 layers, at 2, 4 and 2 neurons each)
    def __init__(self, layer_size_array, weights_and_biases, activation_func):
        self.layers = []
        self.layer_count = len(layer_size_array)
        self.layer_sizes = layer_size_array
        self.layers.append(DataLayer(layer_size_array[0]))
        for layer in range(1, len(layer_size_array)):
            if layer < len(layer_size_array) - 1:
                self.layers.append(FinalLayer(weights_and_biases[layer-1][0], weights_and_biases[layer-1][1], layer_size_array[layer], activation_func))
            elif layer > 0:
                self.layers.append(HiddenLayer(weights_and_biases[layer-1][0], weights_and_biases[layer-1][1], layer_size_array[layer], activation_func))
        self.layers.append(OutputLayer(layer_size_array[len(layer_size_array) - 1]))
        self.learning_rate = 0.1

    def calc_nn(self, data):
        previous_layer_data = data
        for layer in self.layers:
            previous_layer_data = layer.calc_layer(previous_layer_data)
        return previous_layer_data

    def train_nn(self, train_data, test_data, batch_size, epoch_size, learning_rate):
        training_error = np.zeros(epoch_size)
        test_error = np.zeros(epoch_size)
        self.learning_rate = learning_rate

        for layer in self.layers[1:len(self.layers)-1]:
            layer.gen_batch_errors(batch_size)

        for epoch in range(epoch_size):
            batches = self.batch_data(train_data, batch_size)
            for batch in batches:
                self.run_batch(batch)
                self.update_network()
            training_error[epoch] = self.test_nn(train_data)
            test_error[epoch] = self.test_nn(test_data)
            print("Epoch {0} complete".format(epoch + 1))
        return training_error, test_error

    def batch_data(self, labelled_data, batch_size):
        data_size = len(labelled_data)
        rng.shuffle(labelled_data)
        batches = []
        for x in range(0, data_size, batch_size):
            if x + batch_size < data_size:
                batches.append(labelled_data[x:x + batch_size])
            else:
                batches.append(labelled_data[x:])
        return batches

    def run_batch(self, batch):
        batch_run = np.fromiter(map(self.run_calc, batch), dtype=object)

    def run_calc(self, data):
        results = self.calc_nn(data[0])
        expected = data[1]
        self.layers[self.layer_count].output_error(results, expected)
        for layer in range(1,len(self.layers) - 1)[::-1]:
            self.layers[layer].calculate_error(self.layers[layer + 1],self.layers[layer-1])
        return results, expected

    def apply_output_errors(self, error_list):
        for x in range(len(error_list)):
            self.layers[self.layer_count].output_error(error_list[x][0], error_list[x][1])

    def calculate_errors(self):
        for layer in range(1,len(self.layers) - 1)[::-1]:
            self.layers[layer].calculate_error(self.layers[layer + 1],self.layers[layer-1])

    def update_network(self):
        for layer in range(1,self.layer_count):
            self.layers[layer].update_layer(self.learning_rate)

    def test_run(self, labelled_data):
        correct_output = labelled_data[1]
        result = self.calc_nn(labelled_data[0])
        cost = np.sum(np.power(result - correct_output, 2))
        return cost

    def test_nn(self, labelled_data):
        results = []
        costs = np.zeros(len(labelled_data))
        costs = np.fromiter(map(self.test_run, labelled_data), dtype=np.float64)
        return np.average(costs)


def generate_weights_and_biases(layer_sizes):
    gen_size_array = layer_sizes + [layer_sizes[len(layer_sizes)-1]]
    weights_biases = np.zeros(len(layer_sizes), dtype=object)
    for layer in range(len(layer_sizes)):
        weights_biases[layer] = [rng.normal(0, 1, (gen_size_array[layer+1], gen_size_array[layer])),
                                 rng.normal(0, 1, (gen_size_array[layer+1]))]
    return weights_biases