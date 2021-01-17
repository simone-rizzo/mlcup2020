import numpy as np
from itertools import product
from network import DeepNeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def model_selection(params, train_data, train_labels, topn=5, repeat=5, kfold=5):
    """Select the best hyperparameters using gridsearch + k-fold cross validation"""
    list_params = list(product(*list(params.values())))
    best_params = []
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.3)

    # grid search
    for i, param in enumerate(list_params):
        print("gridsearch {}/{}".format(i+1, len(list_params)), end="\r")

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
            mean_repeated_valid_losses += np.min(model.valid_losses)
        
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
        axs[r, c].set_title(f"MODEL: { param['layer_sizes'] } ETA: { param['ETA'] } LAMBDA: { param['LAMBDA'] } ALPHA: { param['ALPHA'] }", fontsize=8)
        axs[r, c].set(xlabel='Ephocs', ylabel='Loss')
        axs[r, c].legend()

        if c < 2:
            c += 1
        else:
            r += 1
            c = 0

    plt.show()