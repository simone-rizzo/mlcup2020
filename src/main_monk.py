from grid_search import model_selection, model_assessment
from load_data import load_monk

params_grid = {
    'layer_sizes': [[17, 4, 1]],
    'ETA': [0.4, 0.5, 0.6],
    'LAMBDA': [0.01, 0.05, 0.1],
    'ALPHA': [0.9]
}

monk = 1
train_data, train_labels = load_monk(monk, 'train', encodeLabel=False)
test_data, test_labels = load_monk(monk, 'test', encodeLabel=False)

best_params = model_selection(params_grid, train_data, train_labels)
model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
print(best_params[0])