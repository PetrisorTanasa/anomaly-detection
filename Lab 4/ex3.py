from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data = loadmat('/Users/ptanasa/Desktop/Anomaly Detection/Lab 3/shuttle.mat')
X = data['X']
y = data['y'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

contamination_rate = np.sum(y) / len(y)

print(X.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import pyod.models.ocsvm as ocsvm
import pyod.models.deep_svdd as deepsvdd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

oc_svm = ocsvm.OCSVM(cache_size=200, gamma=0.75, kernel='sigmoid', max_iter=50, nu=0.01, contamination=contamination_rate)
oc_svm.fit(X_train, y_train)

predictions = oc_svm.predict(X_test)
print("OCSVM")
print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'OCSVM ROC curve (area = {roc_auc:.2f})')
plt.show()

BA = balanced_accuracy_score(y_test, predictions)
print("Balanced Accuracy: " + BA.astype(str))


deep_svdd = deepsvdd.DeepSVDD(hidden_neurons=[16, 16, 32, 32, 8, 4, 2], epochs=50, batch_size=32, n_features=9, hidden_activation='relu', output_activation='sigmoid', random_state=42, contamination=contamination_rate)
deep_svdd.fit(X_train, y_train)

predictions = deep_svdd.predict(X_test)
print("DeepSVDD")
print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DeepSVDD ROC curve (area = {roc_auc:.2f})')
plt.show()

BA = balanced_accuracy_score(y_test, predictions)
print("Balanced Accuracy: " + BA.astype(str))


deep_svdd = deepsvdd.DeepSVDD(hidden_neurons=[16, 32, 8, 4, 2], epochs=50, batch_size=32, n_features=9, hidden_activation='relu', output_activation='sigmoid', random_state=42, contamination=contamination_rate)
deep_svdd.fit(X_train, y_train)

predictions = deep_svdd.predict(X_test)
print("DeepSVDD")
print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DeepSVDD ROC curve (area = {roc_auc:.2f})')
plt.show()

BA = balanced_accuracy_score(y_test, predictions)
print("Balanced Accuracy: " + BA.astype(str))


deep_svdd = deepsvdd.DeepSVDD(hidden_neurons=[16, 16, 32, 32, 8, 2], epochs=50, batch_size=32, n_features=9, hidden_activation='relu', output_activation='sigmoid', random_state=42, contamination=contamination_rate)
deep_svdd.fit(X_train, y_train)

predictions = deep_svdd.predict(X_test)
print("DeepSVDD")
print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DeepSVDD ROC curve (area = {roc_auc:.2f})')
plt.show()

BA = balanced_accuracy_score(y_test, predictions)
print("Balanced Accuracy: " + BA.astype(str))


deep_svdd = deepsvdd.DeepSVDD(hidden_neurons=[16, 16, 32,  32, 8, 4, 2], epochs=50, n_features=9, hidden_activation='relu', output_activation='sigmoid', random_state=42, contamination=contamination_rate)
deep_svdd.fit(X_train, y_train)

predictions = deep_svdd.predict(X_test)
print("DeepSVDD")
print("Number of TN: " + metrics.confusion_matrix(y_test, predictions)[0, 0].astype(str))
print("Number of FP: " + metrics.confusion_matrix(y_test, predictions)[0, 1].astype(str))
print("Number of FN: " + metrics.confusion_matrix(y_test, predictions)[1, 0].astype(str))
print("Number of TP: " + metrics.confusion_matrix(y_test, predictions)[1, 1].astype(str))

fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'DeepSVDD ROC curve (area = {roc_auc:.2f})')
plt.show()

BA = balanced_accuracy_score(y_test, predictions)
print("Balanced Accuracy: " + BA.astype(str))

