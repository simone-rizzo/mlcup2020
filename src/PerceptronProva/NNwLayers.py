import numpy as np

class NeuralNetwork:

    def __init__(self, dim, activation, lr, regression):
        self.layers = []
        self.regression = regression
        self.activation = ActFunctions(activation)
        self.lr = lr
        np.random.seed(0)
        for i in range(len(dim) - 1):
            layer = Layer(dim[i], dim[i + 1], activation, regression)
            self.layers.append(layer)

    def feed_forward(self, x):
        out = x
        self.outputs = [np.copy(x)]
        for layer in self.layers:
            out = layer.foward(out)
            self.outputs.append(out)
        self.output = out

    def back_prop(self, y):
        dw = []  # dC/dW
        db = []  # dC/dB
        deltas = [None] * len(self.layers)
        deltas[-1] = (y-self.outputs[-1]) * self.activation.derivative(self.layers[-1].wx_b)
        for i in reversed(range(len(deltas)-1)):
            #W Delta
            w_delta = deltas[i+1].dot(self.layers[i+1].weights.T)
            #W delta * f(net k)
            #con multilayer da problemi
            deltas[i] = w_delta * self.activation.derivative(self.layers[i+1].wx_b)
        batch_size = y.shape[0]
        db = []
        for d in deltas:
            eyes = np.ones((batch_size, 1))
            val = eyes.T.dot(d)/float(batch_size)
            db.append(val)
        dw = []
        for i, d in enumerate(deltas):
            numeratore = self.outputs[i].T.dot(d)
            val = numeratore / float(batch_size)
            dw.append(val)
        #dw = [self.layers[i].output.T.dot(d) / float(batch_size) for i, d in enumerate(deltas)]
        for w, dweight, dbias in zip(self.layers, dw, db):
            w.weights = w.weights + self.lr * dweight
            w.biases = w.biases + self.lr * dbias

    def fit(self, x, y, batch_size=10, ephocs=100):
        for e in range(ephocs):
            i = 0
            while(i < len(y)):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                i = i+batch_size
                self.feed_forward(x_batch)
                self.back_prop(y_batch)
                print("loss = {}".format(np.linalg.norm(self.output - y_batch)))

class Layer:
    """
        N° input, N° neurons and Activation function 'relu, sigm, tanh'
    """

    def __init__(self, n_inputs, n_neurons, activation, is_regression):
        self.weights = 0.10 * np.random.rand(n_inputs, n_neurons)  # rand is Gaussian distribution
        self.biases = np.zeros((1, n_neurons))
        self.activation = ActFunctions(activation)
        self.is_regression = is_regression

    def foward(self, inputs):
        self.wx_b = np.dot(inputs, self.weights) + self.biases
        if not self.is_regression:
            return self.activation.function(self.wx_b)
        return self.wx_b


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


X = 2 * np.pi * np.random.rand(1000).reshape(-1, 1)
y = np.sin(X)
nn = NeuralNetwork([1, 10, 1], 'relu', 0.01, True)
nn.fit(X, y)
# nn.feed_forward(X)
print(nn.output)
