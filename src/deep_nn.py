import numpy as np
from sklearn.metrics import accuracy_score


class DeepNeuralNetwork():
    """"""
    def __init__(self, layer_sizes, ETA, ALPHA=0, LAMBDA=0, epochs=500, act_hidden="relu", act_out="sigm", loss="MSE", regression=False):
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.epochs = epochs
        self.regression = regression
        self.loss = loss

        self.layers = []
        for i in range(len(layer_sizes)-2):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], act_hidden)
            self.layers.append(layer)
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], act_out))

        # assert self.weight_init in ['xav', 'he', 'type1']
        # assert self.loss in ['MSE', 'MEE']

    def feedforward(self, x):
        """"""
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def backpropagate(self, diff):
        """"""
        # calculate error delta layers
        for layer in reversed(self.layers):
            diff = layer.backpropagate(diff)

        # update weights layers
        for layer in self.layers:
            layer.update_weights(self.ETA)

    def fit(self, train_data, train_label, valid_data, valid_label):
        """"""
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []

        for iteration in range(self.epochs):
            print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")

            # train feedforward and backpropagation
            train_out = self.feedforward(train_data)
            diff = train_label - train_out
            self.backpropagate(diff)
            self.train_losses.append(self.get_loss(train_label, train_out))

            # validation feedforward
            valid_out = self.feedforward(valid_data)
            self.valid_losses.append(self.get_loss(valid_label, valid_out))

            # train loss and accuracy 
            if not self.regression:
                self.train_accuracies.append(self.get_accuracy(train_label, train_out))
            if not self.regression:
                self.valid_accuracies.append(self.get_accuracy(valid_label, valid_out))

    def get_accuracy(self, y_true, y_out):
        """"""
        if not self.regression:
            y_out = np.around(y_out)
        accuracy = np.sum([1 if pred == true else 0 for pred, true in zip(y_out, y_true)])/len(y_true)
        return accuracy

    def get_loss(self, y_true, y_out):
        """"""
        if self.loss == "MSE":
            return np.mean(np.square(y_true - y_out))
        elif self.loss == "MEE":
            return np.mean(np.sqrt(np.sum(np.square(y_true - y_out), axis=1)))


class Layer:
    """"""
    def __init__(self, dim_in, dim_out, activation):
        self.activation = ActFunctions(activation)
        self.init_weights(dim_in, dim_out)

    def init_weights(self, dim_in, dim_out):
        self.w = np.random.randn(dim_in, dim_out)/2
        self.b = np.random.randn(1, dim_out)/2

    def feedforward(self, x):
        self.x = x
        self.h = np.dot(self.x, self.w) + self.b
        return self.activation.function(self.h)

    def backpropagate(self, diff):
        self.delta = diff * self.activation.derivative(self.h)
        return np.dot(self.delta, np.transpose(self.w))

    def update_weights(self, eta):
        self.w += eta * np.dot(np.transpose(self.x), self.delta)
        # self.b += eta * self.delta


class ActFunctions:
    """"""
    def __init__(self, name):
        assert name in ['sigm', 'relu', 'iden']
        self.name = name

    def function(self, x):            
        """"""
        if self.name == 'sigm':
            return 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return np.maximum(x, 0)
        elif self.name == 'iden':
            return x

    def derivative(self, x):            
        """"""
        if self.name == 'sigm':
            return self.function(x) * (1 - self.function(x))
        elif self.name == 'relu':
            return np.greater(x, 0)
        elif self.name == 'iden':
            return 1