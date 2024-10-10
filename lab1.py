# Ex. 1 Use the generate data() function from pyod.utils.data to generate a 2dimensional dataset with 500 normal samples (400 training samples and 100 test samples) with a contamination rate of 0.1. Use pyplot.scatter() function to plot the training samples, choosing a different color for the outliers.
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
import pyod

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', label='Normal', alpha=0.7)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', label='Outliers', alpha=0.7)

plt.title('2D Training Data with Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#Train KNN from pyod

knn = KNN(contamination=0.1)
knn.fit(X_train)

from pyod.utils.data import evaluate_print

predictions = knn.predict(X_train)
evaluate_print('KNN', y_train, predictions)

predictions = knn.predict(X_test)
evaluate_print('KNN', y_test, predictions)

import sklearn.metrics as metrics

print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)

roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.show()



predictions = knn.predict(X_train)
evaluate_print('KNN', y_train, predictions)

scores = knn.decision_scores_

print("Number of TN: " + metrics.confusion_matrix(y_train, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_train, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_train, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_train, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_train, scores)

roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.show()