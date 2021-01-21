from network_utils import ensemble_assessment, model_selection, model_assessment, plot_models
from load_data import load_cup
import numpy as np

# params_grid = {
#     'layer_sizes': [[10, 100, 50, 2]],
#     'ETA': list(np.linspace(0.0005, 0.005, 10)),
#     'LAMBDA': list(np.linspace(0.00001, 0.0005, 5)),
#     'ALPHA': list(np.linspace(0.1, 0.9, 9)),
#     'act_out': ['iden'],
#     'act_hidden': ['leak'],
#     'weight_init': ['default', 'xav', 'he'],
#     'regression': [True],
#     'epochs': [500],
#     'loss': ['MEE']
# }

params_grid = {
    'layer_sizes': [[10, 70, 30, 2]],
    'ETA': [0.003, 0.004, 0.005, 0.006, 0.007],
    'LAMBDA': [0.00001, 0.0001, 0.001, 0.01],
    'ALPHA': [0.5, 0.7, 0.9],
    'act_out': ['iden'],
    'act_hidden': ['tanh'],
    'weight_init': ['default'],
    'regression': [True],
    'epochs': [750, 1000, 1250],
    'loss': ['MEE']
}

filename = "./data/cup/ML-CUP20-TR.csv"
train_data, train_labels, test_data, test_labels = load_cup(filename)

best_params = model_selection(params_grid, train_data, train_labels, topn=9)
# best_model = model_assessment(best_params[0], train_data, train_labels, test_data, test_labels)
# best_model = ensemble_assessment(best_params, train_data, train_labels, test_data, test_labels)
print(f'Best model parameters { best_params[0] }')
print(f'Best model parameters { best_params }')

# uncomment if want to save the result
# change number to save a new configuration
config = 3.2000
f = open(f"cup-{ config }-configuration.txt", "w")
f.write('\n'.join([str(param) for param in best_params]))
f.write('\n')
f.write(str(np.mean([float(param['valid_loss']) for param in best_params])))

# plot and save model
plot_models(best_params, train_data, train_labels, f"cup-{ config }.pdf", save=True)