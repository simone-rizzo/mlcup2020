import numpy as np


class DeepNeuralNetwork():
    """"""

    def __init__(self, layer_sizes, ETA, ALPHA=0, LAMBDA=0, epochs=500, act_hidden="relu", act_out="sigm", loss="MSE", weight_init="xav", regression=False):
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.epochs = epochs
        self.loss = loss
        self.regression = regression

        self.layers = []
        for i in range(len(layer_sizes)-2):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], act_hidden, weight_init)
            self.layers.append(layer)
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1], act_out, weight_init))

        assert self.loss in ['MSE', 'MEE']
        assert act_out in ['iden', 'sigm']
        assert act_hidden in ['relu', 'sigm']
        assert weight_init in ['xav', 'he']

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
            layer.update_weights(self.ETA, self.LAMBDA, self.ALPHA)

    def fit(self, train_data, train_label, valid_data=None, valid_label=None):
        """"""
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []

        for iteration in range(self.epochs):
            # print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")

            # train feedforward and backpropagation
            train_out = self.feedforward(train_data)
            diff = train_label - train_out
            self.backpropagate(diff)
            self.train_losses.append(self.get_loss(train_label, train_out))

            # validation feedforward
            valid_out = None
            if valid_data is not None:
                valid_out = self.feedforward(valid_data)
                self.valid_losses.append(self.get_loss(valid_label, valid_out))

            # train loss and accuracy 
            if not self.regression:
                self.train_accuracies.append(self.get_accuracy(train_label, train_out))
            if not self.regression and valid_out is not None:
                self.valid_accuracies.append(self.get_accuracy(valid_label, valid_out))

    def get_accuracy(self, y_true, y_out):
        """"""
        if self.regression:
            return
        y_out = np.around(y_out)
        accuracy = np.sum([1 if out == true else 0 for out, true in zip(y_out, y_true)])/len(y_true)
        return accuracy

    def get_loss(self, y_true, y_out):
        """"""
        if self.loss == "MSE":
            return np.mean(np.square(y_true - y_out))
        elif self.loss == "MEE":
            return np.mean(np.sqrt(np.sum(np.square(y_true - y_out), axis=1)))


class Layer:
    """"""

    def __init__(self, dim_in, dim_out, activation, weight_init):
        self.activation = ActFunctions(activation)
        self.init_weights(dim_in, dim_out, weight_init)

    def init_weights(self, dim_in, dim_out, weight_init):
        """"""
        if weight_init == 'xav':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(1/dim_out)
            self.b = np.random.randn(1, dim_out)*np.sqrt(1/dim_out)
        elif weight_init == 'he':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(2/dim_out)
            self.b = np.random.randn(1, dim_out)*np.sqrt(2/dim_out)
        # if weight_init == 'xav':
        #     self.w = np.random.randn(dim_in, dim_out)*np.sqrt(6/dim_in+dim_out)
        #     self.b = np.zeros([1, dim_out])
        # elif weight_init == 'he':
        #     self.w = np.random.randn(dim_in, dim_out)*np.sqrt(6/dim_in+dim_out)/2
        #     self.b = np.zeros([1, dim_out])
        self.old_delta_w = 0

    def feedforward(self, x):
        """"""
        self.x = x
        self.h = np.dot(self.x, self.w) + self.b
        return self.activation.function(self.h)

    def backpropagate(self, diff):
        """"""
        self.delta = diff * self.activation.derivative(self.h)
        return np.dot(self.delta, np.transpose(self.w))

    def update_weights(self, eta, lamb, alpha):
        """"""
        delta_w = eta * np.dot(np.transpose(self.x), self.delta)
        delta_b = eta * np.ones((1, self.delta.shape[0])).dot(self.delta)

        # lambda regularization + momentum
        delta_w += -2 * lamb * self.w + alpha * self.old_delta_w
        delta_b += -2 * lamb * self.b 
        self.old_delta_w = delta_w

        # update weights
        self.w += delta_w * (1/self.delta.shape[0]) 
        self.b += delta_b * (1/self.delta.shape[0]) 


class ActFunctions:
    """Class that contains activation functions and derivatives"""

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