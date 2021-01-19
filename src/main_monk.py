from network_utils import ensemble_assessment, model_selection, model_assessment, plot_models
from load_data import load_monk
import numpy as np

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': list(np.linspace(0.1, 0.5, 2)),
    'LAMBDA': [0.01],
    'ALPHA': [0.9],
    'weight_init': ['default'],
    'act_hidden': ['relu', 'sigm', 'tanh']
}

monk = 2
train_data, train_labels = load_monk(monk, 'train')
test_data, test_labels = load_monk(monk, 'test')

best_params = model_selection(params_grid, train_data, train_labels, topn=9)
# best_model = model_assessment(best_params[0], train_data, train_labels, test_data, test_labels)
# best_model = ensemble_assessment(best_params, train_data, train_labels, test_data, test_labels)
plot_models(best_params, train_data, train_labels)
print(f'Best model parameters { best_params[0] }')
print(f'Best model parameters { best_params }')

# uncomment if want to save the result
# f = open(f"monk-{ monk }-configuration.txt", "w")
# f.write('\n'.join([str(param) for param in best_params]))