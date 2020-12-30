import numpy as np
import matplotlib.pyplot as plt
from deep_nn import DeepNeuralNetwork
from sklearn.model_selection import train_test_split
from load_data import load_monk

params = {
    'layer_sizes': [17, 4, 1],
    'ETA': 0.1,
    'LAMBDA': 0.001,
}

def show_single_model(param, train_data, train_labels, valid_data, valid_labels):
    model = DeepNeuralNetwork(**param)
    model.fit(train_data, train_labels, valid_data, valid_labels)
    plt.plot(model.train_losses)
    plt.plot(model.valid_losses)
    plt.show()

monk = 2
train_data, train_labels = load_monk(monk, 'train', encodeLabel=False)
test_data, test_labels = load_monk(monk, 'test', encodeLabel=False)
train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.3)

show_single_model(**params, train_data, train_labels, valid_data, valid_labels)