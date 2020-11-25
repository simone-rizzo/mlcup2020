import math
import pdb
import numpy as np
from sklearn.metrics import accuracy_score


class NeuralNetwork():
    """"""

    def __init__(self, hidden_units, activation, ETA, ALPHA, LAMBDA, weightInitialization='xav', epochs=500, earlyStopping=False, tolerance=1e-3, patience=None, loss="MEE", regression=False):
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.loss_fun = loss
        self.hidden_units = hidden_units
        self.activation = ActFunctions(activation)
        self.weightInitialization = weightInitialization
        self.regression = regression
        self.labelThreshold = 0.5
        self.epochs = epochs
        self.earlyStopping = earlyStopping
        self.validationAccuracies = "The model is NOT trained with validation"
        self.validationLosses = "The model is NOT trained with validation"

        if self.earlyStopping:
            assert patience is not None, "Please provide the 'patience' as number of deterioration epochs."
            self.tolerance = tolerance
            self.patience = patience
        else:
            self.patience = None
            self.tolerance = None

        assert self.weightInitialization in ['xav', 'he', 'type1']
        assert self.loss_fun in ['MSE', 'MEE']

    def create_model(self):
        """"""
        np.random.seed(0)
        if self.weightInitialization == 'xav':
            return{
                'Wih': np.random.randn(self.inputUnits, self.hidden_units)*np.sqrt(1/self.hidden_units),
                'bi': np.random.randn(1, self.hidden_units),
                'Who': np.random.randn(self.hidden_units, self.outputUnits)*np.sqrt(1/self.hidden_units),
                'bh': np.random.randn(1, self.outputUnits)
            }
        elif self.weightInitialization == 'he':
            return{
                'Wih': np.random.randn(self.inputUnits, self.hidden_units)*np.sqrt(2/self.hidden_units),
                'Who': np.random.randn(self.hidden_units, self.outputUnits)*np.sqrt(2/self.hidden_units),
                'bi': np.random.randn(1, self.hidden_units),
                'bh': np.random.randn(1, self.outputUnits)
            }
        elif self.weightInitialization == 'type1':
            return{
                'Wih': np.random.randn(self.inputUnits, self.hidden_units)*np.sqrt(2/(self.hidden_units+self.inputUnits)),
                'Who': np.random.randn(self.hidden_units, self.outputUnits)*np.sqrt(2/(self.hidden_units+self.outputUnits)),
                'bi': np.random.randn(1, self.hidden_units),
                'bh': np.random.randn(1, self.outputUnits)
            }

    def get_loss(self, yTrue, yPred):
        """"""
        if self.loss_fun == "MSE":
            return np.mean(np.square(yTrue - yPred))
        elif self.loss_fun == "MEE":
            return np.mean(np.sqrt(np.sum(np.square(yTrue - yPred), axis=1)))

    def get_accuracy(self, y_true, y_pred):
        """"""
        if not self.regression:
            y_pred = np.around(y_pred)
        return accuracy_score(y_pred, y_true)

    def feedforward(self, dataMatrix):
        """"""
        ih_ = np.dot(dataMatrix, self.model['Wih']) + self.model['bi']  # W*X+b
        hh_ = self.activation.function(ih_)  # 𝝈
        ho_ = np.dot(hh_, self.model['Who']) + self.model['bh']  # W*X1+bh
        if self.regression:
            return hh_, ho_
        oo_ = self.activation.function(ho_)
        return hh_, oo_

    def backpropagation(self, dataMatrix, labelMatrix, hh_, oo_, prevDeltaWho_, prevDeltaWih_):
        """"""
        difference = labelMatrix - oo_

        # o_k * (1 - o_k)(t_k - o_k)
        deriv = self.activation.derivative(oo_)
        deltaOutput_ = difference * deriv

        deriv = self.activation.derivative(hh_)
        deltaHidden_ = deltaOutput_.dot(self.model['Who'].T) * deriv

        # if self.regression:
        #     from sklearn.preprocessing import normalize
        #     # output doesn't passes through nonlinear function for regression
        #     deltaOutput_ = normalize(difference, axis=1, norm='l1')

        # learningrate factor - regularization facor + momentum factor
        deltaWho_ = hh_.T.dot(deltaOutput_) * self.ETA
        otherUpdatesWho = self.model['Who'] * (-self.LAMBDA) + self.ALPHA * prevDeltaWho_
        deltaWih_ = dataMatrix.T.dot(deltaHidden_) * self.ETA
        otherUpdatesWih = self.model['Wih'] * (-self.LAMBDA) + self.ALPHA * prevDeltaWih_

        if self.regression:
            deltaWho_ = deltaWho_/dataMatrix.shape[0]
            deltaWih_ = deltaWih_ / dataMatrix.shape[0]

        self.model['Who'] += deltaWho_ + otherUpdatesWho
        self.model['Wih'] += deltaWih_ + otherUpdatesWih
        self.model['bh'] += np.sum(deltaOutput_,
                                   axis=0, keepdims=True) * self.ETA
        self.model['bi'] += np.sum(deltaHidden_,
                                   axis=0, keepdims=True) * self.ETA
        if self.regression:
            self.model['bh'] /= dataMatrix.shape[0]
            self.model['bi'] /= dataMatrix.shape[0]

        return deltaWho_, deltaWih_

    def predict(self, dataMatrix, labels=None, acc_=False, fromVal=False):
        """"""
        _, result = self.feedforward(dataMatrix)
        if acc_:
            accuracies = []
            for i in range(dataMatrix.shape[0]):
                assert labels is not None, "true values (as labels) must be provided for to calculate accuracy"
                accuracies.append(self.get_accuracy(
                    labels[i], result[i]))
            return result, np.sum(accuracies)/dataMatrix.shape[0]
        if fromVal or self.regression:
            return result
        else:
            return np.around(result)

    def fit(self, features, labels, validationFeatures=None, validationLabels=None, realTimePlotting=False, earlyStoppingLog=True, comingFromGridSearch=False):
        """"""
        self.inputUnits = features.shape[1]
        self.outputUnits = labels.shape[1]
        self.model = self.create_model()
        self.accuracies = []
        self.losses = []
        if validationFeatures is not None:
            self.validationAccuracies = []
            self.validationLosses = []
        deltaWho = 0
        deltaWih = 0
        patience = self.patience
        for iteration in range(self.epochs):
            print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")
            hh, oo = self.feedforward(features)
            prevDeltaWih = deltaWih
            prevDeltaWho = deltaWho
            deltaWho, deltaWih = self.backpropagation(
                features, labels, hh, oo, prevDeltaWho, prevDeltaWih)

            epochLoss = self.get_loss(labels, oo)
            self.losses.append(epochLoss)

            if not self.regression:
                epochAccuracy = self.get_accuracy(labels, oo)
                self.accuracies.append(epochAccuracy)

            if validationFeatures is not None:
                validationResults = self.predict(
                    validationFeatures, acc_=False, fromVal=True)
                self.validationLosses.append(self.get_loss(
                    validationLabels, validationResults))
                if not self.regression:
                    self.validationAccuracies.append(self.get_accuracy(
                        validationLabels, validationResults))

            # if self.earlyStopping:
            #     if iteration > 0:
            #         if comingFromGridSearch:
            #             self.newEpochNotification = False
            #         lossDecrement = (
            #             self.losses[iteration-1]-self.losses[iteration])/self.losses[iteration-1]
            #         if lossDecrement < self.tolerance:
            #             patience -= 1
            #             if patience == 0:
            #                 if earlyStoppingLog:  # researcher mode ;D
            #                     print("The algorithm has run out of patience. \nFinishing due to early stopping on epoch {}. \n PS. Try decreasing 'tolerance' or increasing 'patience'".format(
            #                         iteration))
            #                 self.newEpochNotification = True
            #                 self.bestEpoch = iteration
            #                 break
            #         else:
            #             patience = self.patience


class ActFunctions:
    """"""
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
        """"""
        if self.name == 'sigm':
            return x * (1 - x)
        if self.name == 'relu':
            return np.greater(x, 0)
        if self.name == 'tanh':
            return 1 - x ** 2
