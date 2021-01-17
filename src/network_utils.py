import numpy as np
from itertools import product
from network import DeepNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math


"""
This script contains 3 utility methods used for the network:

    model_selection()   - given a list of hyperparameters, it choose the best one by searching throught gridsearch and 
                            validating using k-fold cross validation

    model_assessment()  - once the best hyperparameter is chosen through model_selection(), this function assesses the 
                            trains the final model on the train + valid set and assesses the performance using the final test set

    plot_models()       - create a grid-plot of n-models by plotting their training and validation losses
"""


def model_selection(params, train_data, train_labels, topn=9, kfold=4):
    """Select the best hyperparameters using gridsearch + k-fold cross validation"""
    list_params = list(product(*list(params.values())))
    best_params = []

    # grid search
    for i, param in enumerate(list_params):
        print("gridsearch {}/{}".format(i+1, len(list_params)), end="\r")
        print(str(i+1)+"/"+str(len(list_params)))
        param_set = {}
        for j in range(len(params)):
            param_set[list(params.keys())[j]] = param[j]

        # k-fold cross validation
        kfold_valid_losses = 0
        ksize = math.floor(len(train_data)/kfold)
        for j in range(kfold):
            model = DeepNeuralNetwork(**param_set)
            train_data_part = np.concatenate((train_data[:j*ksize], train_data[:(j+1)*ksize]), axis=0)
            train_labels_part = np.concatenate((train_labels[:j*ksize], train_labels[:(j+1)*ksize]), axis=0)
            valid_data_part = train_data[j*ksize:(j+1)*ksize]
            valid_labels_part = train_labels[j*ksize:(j+1)*ksize]
            model.fit(train_data_part, train_labels_part, valid_data_part, valid_labels_part)
            kfold_valid_losses += model.valid_losses[-1]
       
        # append in the list
        model_info = {
            'params': param_set,
            'valid_loss': kfold_valid_losses/kfold,
        }
        best_params.append(model_info)

    # sort and return topn in the list
    best_params = sorted(best_params, key=lambda k: k['valid_loss'])
    return best_params[:topn]


def model_assessment(best_param, train_data, train_labels, test_data, test_labels):
    """Evaluate the final model on the test data"""
    # fit model with train data + valid data
    model = DeepNeuralNetwork(**best_param)
    model.fit(train_data, train_labels)

    # compute output from model using test data
    test_out = model.feedforward(test_data)

    if not model.regression:
        accur = model.get_accuracy(test_labels, test_out)
        print(f'Best model accuracy: {accur}')
    else:
        loss = model.get_loss(test_labels, test_out)
        print(f'Best model test loss: {loss}')

    return model


def plot_models(params, train_data, train_labels):
    """Create a grid-plot of n-models by plotting their training and validation losses"""
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.3)

    _, axs = plt.subplots(3, 3)
    r = c = 0
    for i in range(len(params)):
        param = params[i]['params']
        model = DeepNeuralNetwork(**param)
        model.fit(train_data, train_labels, valid_data, valid_labels)
        axs[r, c].plot(np.array(model.train_losses), 'b-', label='Train Loss')
        axs[r, c].plot(np.array(model.valid_losses), 'r--', label='Valid Loss')
        axs[r, c].set_title(f"MODEL: { param['layer_sizes'] } ETA: { param['ETA'] } LAMBDA: { param['LAMBDA'] } ALPHA: { param['ALPHA'] } ACT: { param['act_hidden'] }", fontsize=8)
        axs[r, c].set(xlabel='Ephocs', ylabel='Loss')
        axs[r, c].legend()

        if c < 2:
            c += 1
        else:
            r += 1
            c = 0

    plt.show()