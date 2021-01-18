from src.grid_search import model_selection, model_assessment
from src.load_data import load_monk, load_cup
from src.network import DeepNeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

"""params = {
    'layer_sizes': [10, 100, 50, 2],
    'act_hidden': 'relu',
    'act_out': 'iden',
    'ETA': 0.001,
    'LAMBDA': 0.01,
    'ALPHA': 0.6,
    'WEIGHT_INI': 'he',
    'regression': True,
    'epochs': 500
}"""
params = {
    'layer_sizes': [10, 100, 50, 2],
    'act_hidden': 'tanh',
    'act_out': 'iden',
    'ETA': 0.00450,
    'LAMBDA': 0.00001,
    'ALPHA': 0.6,
    'WEIGHT_INI': 'he',
    'regression': True,
    'epochs': 10000,
    'loss': 'MEE'
}
np.random.seed(0)
filename = "../data/cup/ML-CUP20-TR.csv"
tr_data, tr_label, test_data, test_label = load_cup(filename)
trd, vldata, trlb, vllbl = train_test_split(tr_data, tr_label, test_size=0.20)
nn = DeepNeuralNetwork(**params)
nn.fit(trd, trlb, vldata, vllbl)
plt.plot(nn.train_losses)
plt.plot(nn.valid_losses)
plt.show()
print(nn.train_losses[-1])
print(nn.valid_losses[-1])
# model_assessment(params, tr_data, tr_label, tr_data, tr_label)