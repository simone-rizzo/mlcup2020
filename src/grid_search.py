import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from src.network import DeepNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def model_selection(params, train_data, train_labels, topn=5, repeat=10):
    """"""
    list_params = list(product(*list(params.values())))
    best_params = []
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.3)

    # grid search
    for i, param in enumerate(list_params):
        print("{}/{}".format(i, len(list_params)), end="\r")

        param_set = {}
        for j in range(len(params)):
            param_set[list(params.keys())[j]] = param[j]

        # selection using the mean repetead hold out
        mean_repeated_valid_losses = 0
        for j in range(repeat):
            train_data, train_labels = shuffle(train_data, train_labels)
            valid_data, valid_labels = shuffle(valid_data, valid_labels)
            model = DeepNeuralNetwork(**param_set)
            model.fit(train_data, train_labels, valid_data, valid_labels)
            mean_repeated_valid_losses += np.mean(model.valid_losses)
        
        # append in the list
        model_info = {
            'params': param_set,
            'valid_loss': mean_repeated_valid_losses/repeat,
        }
        best_params.append(model_info)

    # sort and return topn in the list
    best_params = sorted(best_params, key=lambda k: k['valid_loss'])
    return best_params[:topn]


def model_assessment(best_param, train_data, train_labels, test_data, test_labels):
    """"""
    # fit model with train data
    model = DeepNeuralNetwork(**best_param)
    model.fit(train_data, train_labels)

    # compute output from model using test data
    test_out = model.feedforward(test_data)
    accur = model.get_accuracy(test_labels, test_out)

    # plot
    plt.plot(model.train_losses)
    plt.title(accur)
    plt.show()