from network_utils import model_selection, model_assessment, plot_models
from load_data import load_monk
import numpy as np

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': list(np.linspace(0.1, 0.5, 5)),
    # 'ETA': [0.1],
    'LAMBDA': [0.01],
    'ALPHA': [0.9],
    'weight_init': ['default'],
    'act_hidden': ['relu', 'sigm', 'tanh', 'leak']
}

# params_grid = {
#     'layer_sizes': [[17, 2, 1], [17, 3, 1], [17, 4, 1]],
#     'ETA': list(np.linspace(0.1, 0.5, 5)),
#     'LAMBDA': list(np.linspace(0, 0.1, 11)),
#     'ALPHA': list(np.linspace(0, 0.9, 10)),
#     'weight_init': ['default'],
#     'act_hidden': ['relu', 'leak', 'sigm', 'tanh'],
#     'epochs': [400],
#     'loss': ['MSE']
# }

monk = 2
train_data, train_labels = load_monk(monk, 'train')
test_data, test_labels = load_monk(monk, 'test')

best_params = model_selection(params_grid, train_data, train_labels, topn=9)
best_model = model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
plot_models(best_params, train_data, train_labels)
print(f'Best model parameters { best_params[0] }')
print(f'Best model parameters { best_params }')

# f = open("monk-configuration.txt", "a")
# f.write(''.join(str(best_params)))