from grid_search import model_selection, model_assessment
from load_data import load_cup


params_grid = {
    'layer_sizes': [[10, 10, 2]],
    'ETA': [0.01, 0.05, 0.1],
    'LAMBDA': [0.1, 0.3, 0.6],
    'ALPHA': [0.9],
    'act_out': ['iden'],
    'weight_init': ['he'],
    'regression': [True],
    'epochs': [500]
}

filename = "./data/cup/ML-CUP20-TR.csv"
train_data, train_labels, test_data, test_labels = load_cup(filename)

best_params = model_selection(params_grid, train_data, train_labels, repeat=5)
model_assessment(best_params[0]['params'], train_data, train_labels, test_data, test_labels)
print(best_params[0])