from scipy.io import loadmat
from sklearn.model_selection import train_test_split

data = loadmat('/Users/ptanasa/Desktop/Anomaly Detection/Lab 3/shuttle.mat')
X = data['X']
y = data['y'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

from sklearn.preprocessing import StandardScaler
from pyod.models.pca import PCA

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(contamination=0.02)
pca.fit(X_train)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
ax[0].step(range(1, len(pca.explained_variance_)+1), pca.explained_variance_.cumsum())
ax[1].bar(range(1, len(pca.explained_variance_)+1), pca.explained_variance_)
plt.show()

from sklearn.metrics import balanced_accuracy_score
print("Balanced Accuracy: ", balanced_accuracy_score(y_test, pca.predict(X_test)))
print("Balanced Accuracy: ", balanced_accuracy_score(y_train, pca.predict(X_train)))


import numpy as np
import matplotlib.pyplot as plt

# !!! Stiu ca am facut practic split doar pe 0.4 din date, dar cu toate dura muult prea mult, nu parea sa aibe final asteptarea ca am stat destul dupa el ... !!!

from pyod.models.kpca import KPCA
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(X_train, y_train, test_size=0.6, random_state=42)

kpca = KPCA(contamination=0.02, max_iter=1)
kpca.fit(X_train_sampled)

print("Balanced Accuracy: ", balanced_accuracy_score(y_test_sampled, kpca.predict(X_test_sampled)))
print("Balanced Accuracy: ", balanced_accuracy_score(y_train_sampled, kpca.predict(X_train_sampled)))