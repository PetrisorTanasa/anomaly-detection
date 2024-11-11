from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('/Users/ptanasa/Desktop/Anomaly Detection/Lab 3/shuttle.mat')
X = data['X']
y = data['y'].ravel()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def evaluate_models(X_train, X_test, y_train, y_test):
    results = {}
    models = {
        'Isolation Forest': IForest(contamination=0.02),
        'LODA': LODA(contamination=0.02),
        'Deep Isolation Forest': DIF(contamination=0.02, hidden_neurons=[16, 8])
    }

    for name, model in models.items():
        model.fit(X_train)

        scores = model.decision_function(X_test)
        predictions = model.predict(X_test)

        ba = balanced_accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, scores)

        print(f"{name} - Balanced Accuracy: {ba:.4f}, ROC AUC: {roc_auc:.4f}")

        results[name] = (ba, roc_auc)
    print("-------------------------------------")

    return results

n_splits = 10
metrics = {'Isolation Forest': [], 'LODA': [], 'Deep Isolation Forest': []}

for i in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = evaluate_models(X_train, X_test, y_train, y_test)
    for model_name, (ba, roc_auc) in results.items():
        metrics[model_name].append((ba, roc_auc))