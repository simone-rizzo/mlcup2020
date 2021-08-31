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
    regression = False

    # grid search
    for i, param in enumerate(list_params):
        print("gridsearch {}/{}".format(i+1, len(list_params)), end="\r")

        param_set = {}
        for j in range(len(params)):
            param_set[list(params.keys())[j]] = param[j]

        # k-fold cross validation
        kfold_valid_losses = 0
        kfold_valid_accuracies = 0
        ksize = math.floor(len(train_data)/kfold)
        for j in range(kfold):
            model = DeepNeuralNetwork(**param_set)
            regression = model.regression
            train_data_part = np.concatenate((train_data[:j*ksize], train_data[(j+1)*ksize:]), axis=0)
            train_labels_part = np.concatenate((train_labels[:j*ksize], train_labels[(j+1)*ksize:]), axis=0)
            valid_data_part = train_data[j*ksize:(j+1)*ksize]
            valid_labels_part = train_labels[j*ksize:(j+1)*ksize]
            model.fit(train_data_part, train_labels_part, valid_data_part, valid_labels_part)
            kfold_valid_losses += model.valid_losses[-1]
            if not regression:
                kfold_valid_accuracies += model.valid_accuracies[-1]
       
        # append in the list
        model_info = {
            'params': param_set,
            'valid_loss': kfold_valid_losses/kfold,
            'valid_accuracies': kfold_valid_accuracies/kfold
        }
        best_params.append(model_info)

    # sort and return topn in the list
    if regression:
        best_params = sorted(best_params, key=lambda k: k['valid_loss'])
    else:
        best_params = sorted(best_params, key=lambda k: k['valid_accuracies'], reverse=True)
    return best_params[:topn]


def model_assessment(best_param, train_data, train_labels, test_data, test_labels, repeat=10):
    """Evaluate the final model on the test data"""
    loss = 0
    accur = 0
    model = None
    for i in range(repeat):
        # fit model with train data + valid data
        model = DeepNeuralNetwork(**best_param['params'])
        model.fit(train_data, train_labels)

        # compute output from model using test data
        test_out = model.feedforward(test_data)

        # compute loss and accur (accuracy will be computed only if classification problem)
        loss += model.get_loss(test_labels, test_out)
        accur += model.get_accuracy(test_labels, test_out)

    print(f'Best model average test loss over { repeat } repetitions: { loss/repeat }')
    if not model.regression:
        print(f'Best model average accuracy over { repeat } repetitions: { accur/repeat }')

    return model


def ensemble_assessment(best_params, train_data, train_labels, test_data, test_labels):
    """Evaluate the n-final model on the test data"""
    # fit the first model with train data + valid data
    model = DeepNeuralNetwork(**best_params[0]['params'])
    model.fit(train_data, train_labels)
    mean_test_out = model.feedforward(test_data) 

    # fit the other models with train data + valid data
    for i in range(1, len(best_params)):
        model = DeepNeuralNetwork(**best_params[i]['params'])
        model.fit(train_data, train_labels)
        test_out = model.feedforward(test_data)
        mean_test_out += test_out

    # compute loss and accur (accuracy will be computed only if it is a classification problem)
    # the mean_test_out is the mean of all the best_params models results
    # the loss is computed by using the mean_test_out
    n_models = len(best_params)
    mean_test_out /= n_models
    loss = model.get_loss(test_labels, mean_test_out)
    accur = model.get_accuracy(test_labels, mean_test_out)

    print(f'Ensemble average test loss over { n_models } models: { loss }')
    if not model.regression:
        print(f'Ensemble average accuracy over { n_models } models: { accur }')


def plot_models(params, train_data, train_labels, name, save=False):
    """Create a grid-plot of n-models by plotting their training and validation losses"""
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.2)

    _, axs = plt.subplots(3, 3)
    r = c = 0
    for i in range(len(params)):
        param = params[i]['params']
        model = DeepNeuralNetwork(**param)
        model.fit(train_data, train_labels, valid_data, valid_labels)
        axs[r, c].plot(np.array(model.train_losses), 'b-', label='Train Loss')
        axs[r, c].plot(np.array(model.valid_losses), 'r--', label='Valid Loss')
        axs[r, c].set_title(f"MODEL: { param['layer_sizes'] } ETA: { param['ETA'] } LAMBDA: { param['LAMBDA'] } ALPHA: { param['ALPHA'] } ACT: { param['act_hidden'] } EPOCHS: { param['epochs'] }", fontsize=8)
        axs[r, c].set(xlabel='Ephocs', ylabel='Loss')
        axs[r, c].legend()

        if c < 2:
            c += 1
        else:
            r += 1
            c = 0

    plt.gcf().set_size_inches((30, 20), forward=False)
    if save:
        plt.savefig(f"./plots/{ name }", bbox_inches='tight')
    plt.show()