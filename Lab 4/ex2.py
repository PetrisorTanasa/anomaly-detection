from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = loadmat('/Users/ptanasa/Desktop/Anomaly Detection/cardio.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)

import sklearn.svm as svm
import sklearn.model_selection as modelSelection
from sklearn.preprocessing import StandardScaler
import numpy as np
import sklearn.pipeline


standardScaler = StandardScaler()
param_grid = {
    'svm__nu': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'svm__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'svm__gamma': ['scale', 0.1, 0.25,0.5,0.75, 1.0, 'auto'], 
    'svm__max_iter': [10,20,30,40,50,100,200,-1],
    'svm__cache_size' : [200, 400, 800, 1600]
}


oc_svm = svm.OneClassSVM(cache_size=200, gamma=0.75, kernel='sigmoid', max_iter=50, nu=0.01)

pipe = sklearn.pipeline.Pipeline(
    steps=[
        ('scaler', standardScaler),
        ('svm', oc_svm)
    ]
)


# grid_search = modelSelection.GridSearchCV( pipe, param_grid, n_jobs=8, scoring='balanced_accuracy', verbose=1)
# grid_search.fit(X, np.where(y == 1, -1, 1))

# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)
# Best parameters: {'svm__cache_size': 200, 'svm__gamma': 0.1, 'svm__kernel': 'sigmoid', 'svm__max_iter': 50, 'svm__nu': 0.3}
# Best score: 0.9006342327502811
# Best parameters: {'svm__cache_size': 200, 'svm__gamma': 'scale', 'svm__kernel': 'poly', 'svm__max_iter': 10, 'svm__nu': 0.2}
# Best score: 0.9589167633599729
# Best parameters: {'svm__cache_size': 200, 'svm__gamma': 0.75, 'svm__kernel': 'sigmoid', 'svm__max_iter': 50, 'svm__nu': 0.01}
# Best score: 0.9006551167411823

pipe.fit(X_train, y_train)

y_test_pred = pipe.predict(X_test)
y_train_pred = pipe.predict(X_train)

y_test_pred = np.where(y_test_pred == -1, 1, 0)
y_train_pred = np.where(y_train_pred == -1, 1, 0)

from sklearn.metrics import balanced_accuracy_score

BA = balanced_accuracy_score(y_test, y_test_pred)
print("accuracy " + np.where(y_test == y_test_pred, 1, 0).mean().astype(str))
print("Test " + str(BA))
BA = balanced_accuracy_score(y_train, y_train_pred)
print("accuracy " + np.where(y_train == y_train_pred, 1, 0).mean().astype(str))
print("Train " + str(BA))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
fig.suptitle('OneClassSVM')
ax = fig.add_subplot(221)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(222)
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred, cmap='coolwarm')
ax.set_title('Predictions')

ax = fig.add_subplot(223)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(224)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='coolwarm')
ax.set_title('Predictions')

plt.show()
