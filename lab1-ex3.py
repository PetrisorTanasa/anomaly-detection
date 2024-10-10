#Generate a unidimensional dataset with 10 % contamination rate, 1000 training samples and no testing samples using generate data(). Try to detect the anomalies in the dataset by using the Z-scores. In order to do that you should compute the Z-score threshold that would classify the given percent (contamination rate) of data as anomalies (use np.quantile() function). Compute the balanced accuracy of the designed method.

import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
import sklearn.metrics as metrics

X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0, 0], np.zeros_like(X_train[y_train == 0, 0]), c='blue', label='Normal', alpha=0.7)
plt.scatter(X_train[y_train == 1, 0], np.zeros_like(X_train[y_train == 1, 0]), c='red', label='Outliers', alpha=0.7)

plt.legend()
plt.show()

#get z scores
z_scores = abs((X_train - np.mean(X_train)) / np.std(X_train))

threshold = np.quantile(z_scores, 1 - 0.1)
predictions = np.where(z_scores > threshold, 1, 0)

tn, fp, fn, tp = metrics.confusion_matrix(y_train, predictions).ravel()
balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp)) / 2
print("Balanced accuracy: " + balanced_accuracy.astype(str))
print("------------------------------------")
#Same as Ex. 3 but for a multidimensional dataset. Choose your own mean and variance and build your dataset by hand. All other tasks as in Ex. 3.

X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=3, contamination=0.1, random_state=42)


#plot in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='blue', label='Normal', alpha=0.7)
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='red', label='Outliers', alpha=0.7)

plt.legend()
plt.show()

#get z scores
covariance_matrix = np.cov(X_train.T)
y_form = X_train - np.mean(X_train, axis=0)

L = np.linalg.cholesky(covariance_matrix)
z_scores = abs(np.linalg.solve(L, y_form.T).T)

threshold = np.quantile(z_scores, 1-0.1)

predictions = np.where(z_scores > threshold, 1, 0)

prediction_array = np.all(predictions == 1, axis=1).astype(int)

tn, fp, fn, tp = metrics.confusion_matrix(y_train, prediction_array).ravel()
balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp)) / 2
print("Balanced accuracy: " + balanced_accuracy.astype(str))
