import numpy as np


"""
This script contains the neural network implementation using 3 classes: 

    DeepNeuralNetwork, neural network implementation from scratch, functionalities:
        feedforward()   - feedforward the input data through the network
        backpropagate() - backpropagation algorithm through the network
        fit()           - train and validate the network using data
        get_loss()      - calculate the loss
        get_accuracy()  - calculate the accuracy (classification problem)

    Layer, layer componenet for the network, functionalities:
        init_weights()      - initialize the weights of the layer
        feedforward()       - feedforward the input data through the layer
        backpropagate()     - compute the error term of the layer using the backpropagation algorithm
        update_weights()    - update the weights using the backpropagation algorithm

    ActFunctions, contain activation functions and its corresponding derivatives
"""


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork is a implementation of the Neural Network used for Deep Learning
    """

    def __init__(self, layer_sizes, ETA, ALPHA=0, LAMBDA=0, epochs=500, act_hidden="relu", act_out="sigm", loss="MSE", weight_init="default", regression=False):
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
        assert act_hidden in ['relu', 'leak', 'sigm', 'tanh']
        assert weight_init in ['default', 'xav', 'he']

    def feedforward(self, x):
        """Compute the feedforward by passing the input throught every layer"""
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def backpropagate(self, diff):
        """Compute the backpropagation algorithm"""
        # calculate error delta layers
        for layer in reversed(self.layers):
            diff = layer.backpropagate(diff)

        # update weights layers
        for layer in self.layers:
            layer.update_weights(self.ETA, self.LAMBDA, self.ALPHA)

    def fit(self, train_data, train_label, valid_data=None, valid_label=None):
        """Train the network and validate (optionally)"""
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

    def get_loss(self, y_true, y_out):
        """Compute the loss score"""
        if self.loss == "MSE":
            return np.mean(np.square(y_true - y_out))
        elif self.loss == "MEE":
            return np.mean(np.sqrt(np.sum(np.square(y_true - y_out), axis=1)))

    def get_accuracy(self, y_true, y_out):
        """Compute the accuracy score (only classification problem)"""
        if self.regression:
            return
        y_out = np.around(y_out)
        accuracy = np.sum([1 if out == true else 0 for out, true in zip(y_out, y_true)])/len(y_true)
        return accuracy


class Layer:
    """
    Fully connected layer used in DeepNeuralNetwork.
    """

    def __init__(self, dim_in, dim_out, activation, weight_init):
        self.activation = ActFunctions(activation)
        self.init_weights(dim_in, dim_out, weight_init)

    def init_weights(self, dim_in, dim_out, weight_init):
        """Initialize the weights of the layer"""
        if weight_init == 'default':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(1/dim_out)
            self.b = np.zeros([1, dim_out])
        if weight_init == 'xav':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(6/dim_in+dim_out)
            self.b = np.zeros([1, dim_out])
        elif weight_init == 'he':
            self.w = np.random.randn(dim_in, dim_out)*(np.sqrt(6/dim_in+dim_out)/2)
            self.b = np.zeros([1, dim_out])
        self.old_delta_w = 0
        if weight_init == 'xav':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(1/dim_out)
            self.b = np.random.randn(1, dim_out)*np.sqrt(1/dim_out)
        elif weight_init == 'he':
            self.w = np.random.randn(dim_in, dim_out)*np.sqrt(2/dim_out)
            self.b = np.random.randn(1, dim_out)*np.sqrt(2/dim_out)
        elif weight_init == 'default':
            self.w = np.random.randn(dim_in, dim_out) / 2
            self.b = np.random.randn(1, dim_out) / 2
        elif weight_init == 'type1':
            self.w = np.random.randn(dim_in, dim_out) * np.sqrt(2 / dim_in+dim_out)
            self.b = np.random.randn(1, dim_out) * np.sqrt(2 / 1+dim_out)

    def feedforward(self, x):
        """Compute the feedforward of the layer"""
        self.x = x
        self.h = np.dot(self.x, self.w) + self.b
        return self.activation.function(self.h)

    def backpropagate(self, diff):
        """Compute the error term in the backpropagation algorithm"""
        self.delta = diff * self.activation.derivative(self.h)
        return np.dot(self.delta, np.transpose(self.w))

    def update_weights(self, eta, lamb, alpha):
        """Update each weight of the layer"""
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
    """
    This class contains activation functions and its corresponding derivatives
    """

    def __init__(self, name):
        self.name = name

    def function(self, x):
        """Activation functions"""
        if self.name == 'sigm':
            return 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return np.maximum(x, 0)
        elif self.name == 'leak':
            return np.where(x > 0, x, x * 0.1)
        elif self.name == 'iden':
            return x
        elif self.name == 'tanh':
            return np.tanh(x)

    def derivative(self, x):
        """Derivatives of activation functions"""
        if self.name == 'sigm':
            return self.function(x) * (1 - self.function(x))
        elif self.name == 'relu':
            return np.greater(x, 0)
        elif self.name == 'leak':
            return np.where(x > 0, 1, 0.1)
        elif self.name == 'iden':
            return 1
        elif self.name == 'tanh':
            return 1 - np.power(np.tanh(x), 2)