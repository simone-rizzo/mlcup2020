import numpy as np
from sklearn.metrics import accuracy_score


class NeuralNetwork():
    """"""

    def __init__(self, hidden_units, activation, ETA, ALPHA, LAMBDA, weight_init='xav', epochs=500, early_stopping=False, tolerance=1e-3, patience=None, loss="MEE", regression=False):
        self.ETA = ETA
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.loss = loss
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.weight_init = weight_init
        self.regression = regression
        self.early_stopping = early_stopping
        self.activation = ActFunctions(activation)

        # if self.early_stopping:
        #     assert patience is not None, "Please provide the 'patience' as number of deterioration epochs."
        #     self.tolerance = tolerance
        #     self.patience = patience
        # else:
        #     self.patience = None
        #     self.tolerance = None

        assert self.weight_init in ['xav', 'he', 'type1']
        assert self.loss in ['MSE', 'MEE']

    def create_model(self, train_data, train_label):
        """"""
        # np.random.seed(0)
        self.input_units = train_data.shape[1]
        self.output_units = train_label.shape[1]
        
        if self.weight_init == 'xav':
            return{
                'W_ih': np.random.randn(self.input_units, self.hidden_units)*np.sqrt(1/self.hidden_units),
                'b_h': np.random.randn(1, self.hidden_units),
                'W_ho': np.random.randn(self.hidden_units, self.output_units)*np.sqrt(1/self.hidden_units),
                'b_o': np.random.randn(1, self.output_units)
            }
        elif self.weight_init == 'he':
            return{
                'W_ih': np.random.randn(self.input_units, self.hidden_units)*np.sqrt(2/self.hidden_units),
                'W_ho': np.random.randn(self.hidden_units, self.output_units)*np.sqrt(2/self.hidden_units),
                'b_h': np.random.randn(1, self.hidden_units),
                'b_o': np.random.randn(1, self.output_units)
            }
        elif self.weight_init == 'type1':
            return{
                'W_ih': np.random.randn(self.input_units, self.hidden_units)*np.sqrt(2/(self.hidden_units+self.input_units)),
                'W_ho': np.random.randn(self.hidden_units, self.output_units)*np.sqrt(2/(self.hidden_units+self.output_units)),
                'b_h': np.random.randn(1, self.hidden_units),
                'b_o': np.random.randn(1, self.output_units)
            }

    def get_loss(self, y_true, y_pred):
        """"""
        if self.loss == "MSE":
            return np.mean(np.square(y_true - y_pred))
        elif self.loss == "MEE":
            return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=1)))

    def get_accuracy(self, y_true, y_pred):
        """"""
        if not self.regression:
            y_pred = np.around(y_pred)
        accuracy = np.sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)])/len(y_true)
        return accuracy

    def feedforward(self, data):
        """"""
        ih_ = np.dot(data, self.model['W_ih']) + self.model['b_h']
        hh_ = self.activation.function(ih_)
        ho_ = np.dot(hh_, self.model['W_ho']) + self.model['b_o']
        if self.regression:
            return hh_, ho_
        oo_ = self.activation.function(ho_)
        return hh_, oo_

    def backpropagation(self, data, label, hh_, oo_, old_delta_out, old_delta_hid):
        """"""
        difference = label - oo_

        # o_k * (1 - o_k)(t_k - o_k)
        deriv = self.activation.derivative(oo_)
        delta_out = difference * deriv

        deriv = self.activation.derivative(hh_)
        delta_hid = delta_out.dot(self.model['W_ho'].T) * deriv

        # if self.regression:
        #     from sklearn.preprocessing import normalize
        #     # output doesn't passes through nonlinear function for regression
        #     delta_out = normalize(difference, axis=1, norm='l1')

        # learningrate factor - regularization facor + momentum factor
        delta_out_ = hh_.T.dot(delta_out) * self.ETA
        otherUpdatesW_ho = self.model['W_ho'] * (-self.LAMBDA) + self.ALPHA * old_delta_out
        delta_hid_ = data.T.dot(delta_hid) * self.ETA
        otherUpdatesW_ih = self.model['W_ih'] * (-self.LAMBDA) + self.ALPHA * old_delta_hid

        if self.regression:
            delta_out_ = delta_out_/data.shape[0]
            delta_hid_ = delta_hid_ / data.shape[0]

        self.model['W_ho'] += delta_out_ + otherUpdatesW_ho
        self.model['W_ih'] += delta_hid_ + otherUpdatesW_ih
        self.model['b_o'] += np.sum(delta_out,
                                   axis=0, keepdims=True) * self.ETA
        self.model['b_h'] += np.sum(delta_hid,
                                   axis=0, keepdims=True) * self.ETA
        if self.regression:
            self.model['b_o'] /= data.shape[0]
            self.model['b_h'] /= data.shape[0]

        return delta_out_, delta_hid_

    def predict(self, data, labels=None, acc_=False):
        """"""
        _, result = self.feedforward(data)
        if acc_:
            accuracies = []
            for i in range(data.shape[0]):
                assert labels is not None, "Labels are not provided. Can't calculate accuracy."
                accuracies.append(self.get_accuracy(
                    labels[i], result[i]))
            return result, np.sum(accuracies)/data.shape[0]

        if self.regression:
            return result
        else:
            return np.around(result)

    def fit(self, train_data, train_label, valid_data=None, valid_label=None, realTimePlotting=False, early_stoppingLog=True, comingFromGridSearch=False):
        """"""
        self.model = self.create_model(train_data, train_label)
        self.train_accuracies = []
        self.train_losses = []
        self.valid_accuracies = []
        self.valid_losses = []

        delta_out = 0
        delta_hid = 0

        for iteration in range(self.epochs):
            print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")
            hh, oo = self.feedforward(train_data)
            old_delta_out = delta_out
            old_delta_hid = delta_hid
            delta_out, delta_hid = self.backpropagation(
                train_data, train_label, hh, oo, old_delta_out, old_delta_hid)

            epochLoss = self.get_loss(train_label, oo)
            self.train_losses.append(epochLoss)

            if not self.regression:
                self.train_accuracies.append(self.get_accuracy(train_label, oo))

            if valid_data is not None:
                valid_result = self.predict(valid_data, acc_=False, fromVal=True)
                self.valid_losses.append(self.get_loss(valid_label, valid_result))
                if not self.regression:
                    self.valid_accuracies.append(self.get_accuracy(
                        valid_label, valid_result))

            # patience = self.patience
            # if self.early_stopping:
            #     if iteration > 0:
            #         if comingFromGridSearch:
            #             self.newEpochNotification = False
            #         lossDecrement = (
            #             self.losses[iteration-1]-self.losses[iteration])/self.losses[iteration-1]
            #         if lossDecrement < self.tolerance:
            #             patience -= 1
            #             if patience == 0:
            #                 if early_stoppingLog:  # researcher mode ;D
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