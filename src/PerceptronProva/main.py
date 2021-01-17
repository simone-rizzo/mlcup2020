import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return x * (1 - x)

class myPerceptron():

    def __init__(self, inputs_len, epochs, ETA):
        self.weights = 2*np.random.random((inputs_len, 1))-1
        self.EPOCS = epochs
        self.ETA = ETA

    def feed_forward(self, x):
        out = sigmoid(np.dot(x, self.weights))
        return out

    def back_prop(self, target, outputs, inputs):
        err = target-outputs
        di = err * sigmoid_der(outputs)
        Deltaw = self.ETA * np.dot(inputs.T, di)
        self.weights += Deltaw

    def train(self, inputs, targets):
        for i in range(self.EPOCS):
            preditions = self.feed_forward(inputs)
            self.back_prop(targets, preditions, inputs)
        return preditions


training_inputs = np.array([[1, 0, 0, 1],
                           [1, 1, 1, 1],
                           [1, 1, 0, 1],
                           [1, 0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1) #setto il seed
ETA = 0.01
EPOCS = 20000
model = myPerceptron(4, EPOCS, ETA)
out = model.train(training_inputs, training_outputs)
print(out)