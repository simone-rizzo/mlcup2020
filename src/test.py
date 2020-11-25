from sklearn.metrics import accuracy_score
import numpy as np
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

print(accuracy_score(y_true, y_pred))
print(np.sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)])/len(y_true))