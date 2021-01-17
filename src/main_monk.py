from network_utils import model_selection, model_assessment, plot_models
from load_data import load_monk
import numpy as np

# params_grid = {
#     'layer_sizes': [[17, 4, 1]],
#     'ETA': list(np.linspace(0.1, 0.5, 5)),
#     'LAMBDA': list(np.linspace(0.01, 0.1, 5)),
#     'ALPHA': list(np.linspace(0.6, 0.9, 3)),
#     'weight_init': ['default', 'xav', 'he']
# }

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': list(np.linspace(0.1, 0.5, 1)),
    'LAMBDA': list(np.linspace(0.01, 0.1, 1)),
    'ALPHA': list(np.linspace(0.6, 0.9, 1)),
    'weight_init': ['default', 'xav', 'he']
}

monk = 3
train_data, train_labels = load_monk(monk, 'train')
test_data, test_labels = load_monk(monk, 'test')

best_params = model_selection(params_grid, train_data, train_labels, topn=9)
best_model = model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
plot_models(best_params, train_data, train_labels)
print(f'Best model parameters { best_params[0] }')