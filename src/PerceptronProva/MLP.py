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

    def back_prop(self, target, outputs, out_h, winold, woutold, inputs):
        difference = target-outputs
        deriv = sigmoid_der(outputs)
        deltaOutput_ = difference * deriv
        deriv = sigmoid_der(out_h)
        deltaHidden_ = deltaOutput_.dot(self.weights_out.T)*deriv
        deltaWho_ = out_h.T.dot(deltaOutput_) * self.ETA
        deltaWih_ = inputs.T.dot(deltaHidden_) * self.ETA
        self.weights_out = self.ETA * self.weights_out + deltaWho_ -(self.LAMBDA * self.weights_out)+self.APLPHA*woutold
        self.weights_in = self.ETA * self.weights_in + deltaWih_ -(self.LAMBDA * self.weights_in)+self.APLPHA*winold
        return self.weights_in, self.weights_out

    def train(self, inputs, targets):
        self.losses = []
        oldh = 0
        oldout = 0
        for i in range(self.EPOCS):
            (out_H, preditions) = self.feed_forward(inputs)
            oldh, oldout = self.back_prop(targets, preditions, out_H, oldh, oldout, inputs)
            self.losses.append(loss(targets, preditions))
        return preditions

    def predict(self, dataMatrix, labels):
        _, result = self.feed_forward(dataMatrix)
        n_correct=0
        tot = dataMatrix.shape[0]
        for i in range(dataMatrix.shape[0]):
            arrotondo = 1 if result[i] >= 0.5 else 0
            if labels[i] == arrotondo:
                n_correct += 1
        return n_correct/tot

training_inputs = np.array([[1, 0, 0, 1],
                           [1, 1, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T
training_inputs, training_outputs = loadMonk(1, 'train', encodeLabel=False)
training_inputs = np.c_[np.ones(training_inputs.shape[0]), training_inputs] # aggiungo 1 alla x
np.random.seed(1) #setto il seed
ETA = 0.1
EPOCS = 400 #400 per il resto
LAMBDA = 0.001
ALPHA = 0.9
model = myPerceptron(18, 4, 1, EPOCS, ETA, LAMBDA, ALPHA)
out = model.train(training_inputs, training_outputs)
testData, testLabels = loadMonk(1, 'test', encodeLabel=False)
testData = np.c_[np.ones(testData.shape[0]), testData]
accuracy = model.predict(testData, testLabels)
print("test: accuracy")
print(accuracy)
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