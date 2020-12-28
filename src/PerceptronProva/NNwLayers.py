import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class NeuralNetwork:

    def __init__(self, dim, activation):
        self.layers = []
        np.random.seed(0)
        for i in range(len(dim)-1):
            layer = Layer(dim[i], dim[i+1], activation, False)
            self.layers.append(layer)
        self.l1 = Layer(4, 5, 'tanh', False)
        self.l2 = Layer(5, 2, 'tanh', False)


    def feed_forward(self, X):
        out = X
        for layer in self.layers:
            layer.foward(out)
            out = layer.output
        self.output = out

    def back_prop(self):
        pass

    def fit(self):
        pass

class Layer:
    """
        N° input, N° neurons and Activation function 'relu, sigm, tanh'
    """
    def __init__(self, n_inputs, n_neurons, activation, islast_layer):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)  # rand is Gaussian distribution
        self.biases = np.zeros((1, n_neurons))
        self.activation = ActFunctions(activation)
        self.islast_layer = islast_layer

    def foward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.function(self.output)

    def back_prop(self):
        pass


class ActFunctions:
    """Activation functions"""

    def __init__(self, name):
        assert name in ['sigm', 'relu', 'tanh']
        self.name = name

    def function(self, x):
        """"""
        if self.name == 'sigm':
            return 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return np.maximum(x, 0)
        elif self.name == 'tanh':
            return np.tanh(x)

    def derivative(self, x):
        """Derivatives of activation functions"""
        if self.name == 'sigm':
            return x * (1 - x)
        if self.name == 'relu':
            return np.greater(x, 0)
        if self.name == 'tanh':
            return 1 - x ** 2

nn = NeuralNetwork([4, 5, 2], 'tanh')
nn.feed_forward(X)
print(nn.output)
