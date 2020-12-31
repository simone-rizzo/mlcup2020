from src.grid_search import model_selection, model_assessment
from src.load_data import load_monk, load_cup
from src.network import DeepNeuralNetwork
import matplotlib.pyplot as plt

params = {
    'layer_sizes': [10, 10, 10, 2],
    'act_hidden': 'relu',
    'act_out': 'iden',
    'ETA': 0.5,
    'LAMBDA': 0.1,
    'ALPHA': 0.9,
    'WEIGHT_INI': 'xav',
    'regression': True
}

filename = "../data/cup/ML-CUP20-TR.csv"
tr_data, tr_label, test_data, test_label = load_cup(filename)
nn = DeepNeuralNetwork(**params)
nn.fit(tr_data, tr_label)
plt.plot(nn.train_losses)
plt.show()
# model_assessment(params, tr_data, tr_label, tr_data, tr_label)