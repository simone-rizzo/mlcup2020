import numpy as np
import matplotlib.pyplot as plt

from src.loadData import loadMonk


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)

def loss(yTrue, yPred):
    return np.mean(np.square(yTrue - yPred))

class myPerceptron():

    def __init__(self, inputs_len, hidden_len, outputs_len, epochs, ETA, LAMBDA, ALPHA):
        self.weights_in = 2 * np.random.random((inputs_len, hidden_len))-1
        self.weights_out = 2 * np.random.random((hidden_len, outputs_len)) - 1
        self.EPOCS = epochs
        self.ETA = ETA
        self.LAMBDA = LAMBDA
        self.APLPHA = ALPHA

    def feed_forward(self, x):
        out_H = sigmoid(np.dot(x, self.weights_in))
        out_T = sigmoid(np.dot(out_H, self.weights_out))
        return (out_H, out_T)

    def back_prop(self, target, outputs, out_h, winold, woutold):
        difference = target-outputs
        deriv = sigmoid_der(outputs)
        deltaOutput_ = difference * deriv
        deriv = sigmoid_der(out_h)
        deltaHidden_ = deltaOutput_.dot(self.weights_out.T)*deriv
        deltaWho_ = out_h.T.dot(deltaOutput_) * self.ETA
        deltaWih_ = outputs.T.dot(deltaHidden_) * self.ETA
        self.weights_out = self.weights_out + deltaWho_ -(self.LAMBDA * self.weights_out)+self.APLPHA*woutold
        self.weights_in = self.weights_in + deltaWih_ -(self.LAMBDA * self.weights_in)+self.APLPHA*winold
        return self.weights_in, self.weights_out

    def train(self, inputs, targets):
        self.losses = []
        oldh = 0
        oldout = 0
        for i in range(self.EPOCS):
            (out_H, preditions) = self.feed_forward(inputs)
            oldh, oldout = self.back_prop(targets, preditions, out_H, oldh, oldout)
            self.losses.append(loss(targets, preditions))
        return preditions


training_inputs = np.array([[1, 0, 0, 1],
                           [1, 1, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T
training_inputs, training_outputs = loadMonk(1, 'train', encodeLabel=False)
training_inputs = np.c_[ np.ones(training_inputs.shape[0]), training_inputs] # aggiungo 1 alla x
np.random.seed(1) #setto il seed
ETA = 0.1
EPOCS = 1000
LAMBDA = 0.00001
ALPHA = 0.005
model = myPerceptron(17, 4, 1, EPOCS, ETA, LAMBDA, ALPHA)
out = model.train(training_inputs, training_outputs)
print(out)
plt.plot(np.array(model.losses))
plt.show()
"""ETA = 0.1
EPOCS = 1000
LAMBDA = 0.00001
ALPHA = 0.005
model = myPerceptron(4, 3, 1, EPOCS, ETA, LAMBDA, ALPHA)
out = model.train(training_inputs, training_outputs)
print(out)
plt.plot(np.array(model.losses))
plt.show()"""