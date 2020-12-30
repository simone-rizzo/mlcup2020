import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from network import DeepNeuralNetwork


def model_selection(params, train_data, train_labels, valid_data, valid_labels, topn=5):
    """"""
    list_params = list(product(*list(params.values())))
    best_params = []
    for i, param in enumerate(list_params):
        print("{}/{}".format(i, len(list_params)), end="\r")

        param_set = {}
        for j in range(len(params)):
            param_set[list(params.keys())[j]] = param[j]

        model = DeepNeuralNetwork(**param_set)
        model.fit(train_data, train_labels, valid_data, valid_labels)
        
        model_info = {
            'params': param_set,
            'mean valid loss': np.mean(model.valid_losses),
        }
        best_params.append(model_info)

    best_params = sorted(best_params, key=lambda k: k['mean valid loss'])
    return best_params[:topn]


def model_assessment(best_param, train_data, train_labels, valid_data, valid_labels, test_data, test_labels):
    """"""
    # combine train data and valid data
    train_data = np.concatenate([train_data, valid_data])
    train_labels = np.concatenate([train_labels, valid_labels])

    # fit model with train data
    model = DeepNeuralNetwork(**best_param)
    model.fit(train_data, train_labels)

    # compute output from model using test data
    test_out = model.feedforward(test_data)
    accur = model.get_accuracy(test_labels, test_out)

    plt.plot(model.train_losses)
    plt.title(accur)
    plt.show()