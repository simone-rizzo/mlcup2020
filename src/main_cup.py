from src.network_utils import model_selection, model_assessment, plot_models
from src.load_data import load_cup
import numpy as np

params_grid = {
    'layer_sizes': [[10, 100, 50, 2]],
    'ETA': [0.00450],
    'LAMBDA': [0.00001],
    'ALPHA': [0.7],
    'act_out': ['iden'],
    'act_hidden': ['tanh'],
    'weight_init': ['xav'],
    'regression': [True],
    'epochs': [10000],
    'loss': ['MEE']
}
"""params_grid = {
    'layer_sizes': [[10, 100, 50, 2]],
    'ETA': list(np.linspace(0.0005, 0.005, 10)),
    'LAMBDA': list(np.linspace(0.00001, 0.0005, 10)),
    'ALPHA': list(np.linspace(0.1, 0.9, 9)),
    'act_out': ['iden'],
    'act_hidden': ['tanh'],
    'weight_init': ['default'],
    'regression': [True],
    'epochs': [500],
    'loss': ['MEE']
}"""
filename = "../data/cup/ML-CUP20-TR.csv"
train_data, train_labels, test_data, test_labels = load_cup(filename)
np.random.seed(0)
best_params = model_selection(params_grid, train_data, train_labels, topn=9)
# best_model = model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
plot_models(best_params, train_data, train_labels)
print(f'Best model parameters { best_params[0] }')
print(best_params)