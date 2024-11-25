import pyod.utils.data as data

X1, X2, y1, y2 = data.generate_data(300, 200, 3, 0.15, random_state=0)

import matplotlib.pyplot as plt
import pyod.models.ocsvm as ocsvm

clf = ocsvm.OCSVM(contamination=0.15, kernel='linear')
clf.fit(X1, y1)

#create 4 3d plots showing the ground truth and the predictions
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Linear kernel')
ax = fig.add_subplot(221, projection='3d')
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(222, projection='3d')
y1_pred = clf.predict(X1)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1_pred, cmap='coolwarm')
ax.set_title('Predictions')

ax = fig.add_subplot(223, projection='3d')
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(224, projection='3d')
y2_pred = clf.predict(X2)
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2_pred, cmap='coolwarm')
ax.set_title('Predictions')

plt.show()

#calculate the balanced accuracy of the model
from sklearn.metrics import balanced_accuracy_score
predictions = clf.predict(X2)
BA = balanced_accuracy_score(y2, predictions)
print(BA)

clf = ocsvm.OCSVM(contamination=0.15)
clf.fit(X1, y1)

#create 4 3d plots showing the ground truth and the predictions
fig = plt.figure(figsize=(10, 10))
fig.suptitle('RBF kernel')
ax = fig.add_subplot(221, projection='3d')
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(222, projection='3d')
y1_pred = clf.predict(X1)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1_pred, cmap='coolwarm')
ax.set_title('Predictions')

ax = fig.add_subplot(223, projection='3d')
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(224, projection='3d')
y2_pred = clf.predict(X2)
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2_pred, cmap='coolwarm')
ax.set_title('Predictions')

plt.show()

#calculate the balanced accuracy of the model
from sklearn.metrics import balanced_accuracy_score
predictions = clf.predict(X2)
BA = balanced_accuracy_score(y2, predictions)
print(BA)

import pyod.models.deep_svdd as deep_svdd

clf = deep_svdd.DeepSVDD(contamination=0.15, hidden_neurons=[8, 4, 2], epochs=10, n_features=3)
clf.fit(X1, y1)

#create 4 3d plots showing the ground truth and the predictions
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Deep SVDD')
ax = fig.add_subplot(221, projection='3d')
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(222, projection='3d')
y1_pred = clf.predict(X1)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=y1_pred, cmap='coolwarm')
ax.set_title('Predictions')

ax = fig.add_subplot(223, projection='3d')
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2, cmap='coolwarm')
ax.set_title('Ground truth')

ax = fig.add_subplot(224, projection='3d')
y2_pred = clf.predict(X2)
ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=y2_pred, cmap='coolwarm')
ax.set_title('Predictions')

plt.show()

#calculate the balanced accuracy of the model
from sklearn.metrics import balanced_accuracy_score
predictions = clf.predict(X2)
BA = balanced_accuracy_score(y2, predictions)
print(BA)