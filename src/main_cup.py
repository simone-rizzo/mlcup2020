from network_utils import model_selection, model_assessment, plot_models
from load_data import load_cup
import numpy as np

# params_grid = {
#     'layer_sizes': [[10, 100, 50, 2]],
#     'ETA': list(np.linspace(0.001, 0.01, 3)),
#     'LAMBDA': list(np.linspace(0.00001, 0.0001, 3)),
#     'ALPHA': list(np.linspace(0.8, 0.9, 2)),
#     'act_out': ['iden'],
#     'act_hidden': ['tanh', 'sigm', 'relu'],
#     'weight_init': ['default', 'xav', 'he'],
#     'regression': [True],
#     'epochs': [500],
#     'loss': ['MEE']
# }

params_grid = {
    'layer_sizes': [[10, 100, 50, 2]],
    # 'ETA': list(np.linspace(0.0005, 0.001, 3)),
    # 'LAMBDA': list(np.linspace(0.00001, 0.0001, 3)),
    # 'ALPHA': list(np.linspace(0.5, 0.9, 5)),
    'ETA': [0.001],
    'LAMBDA': [0],
    'ALPHA': [0.9],
    'act_out': ['iden'],
    'act_hidden': ['tanh'],
    'weight_init': ['default'],
    'regression': [True],
    'epochs': [10000],
    'loss': ['MEE']
}

filename = "./data/cup/ML-CUP20-TR.csv"
train_data, train_labels, test_data, test_labels = load_cup(filename)

best_params = model_selection(params_grid, train_data, train_labels, topn=9)
# best_model = model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
plot_models(best_params, train_data, train_labels)
print(f'Best model parameters { best_params[0] }')