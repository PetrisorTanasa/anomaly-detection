import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pyod.models.iforest as iforest
import pyod.models.loda as loda
import pyod.models.dif as DIF

X, y = make_blobs(n_samples=500, centers=[(10, 0), (0, 10)], cluster_std=1, n_features=2, random_state=0)

model = iforest.IForest(contamination=0.02)
model.fit(X)

test_data = np.random.uniform(-10, 20, (1000, 2))


anomaly_scores = model.decision_function(test_data)
plt.scatter(test_data[:,0], test_data[:,1], c=anomaly_scores, cmap='viridis')
plt.colorbar()
plt.show()

iforest_scores = model.decision_function(test_data)

model = DIF.DIF(contamination=0.02, hidden_neurons= [8, 4, 2])
model.fit(X)

test_data = np.random.uniform(-10, 20, (1000, 2))   

anomaly_scores = model.decision_function(test_data)
plt.scatter(test_data[:,0], test_data[:,1], c=anomaly_scores, cmap='viridis')
plt.colorbar()
plt.show()

dif_scores = model.decision_function(test_data)

model = loda.LODA(contamination=0.02, n_bins=5)
model.fit(X)

anomaly_scores = model.decision_function(test_data)

plt.scatter(test_data[:,0], test_data[:,1], c=anomaly_scores, cmap='viridis')
plt.colorbar()
plt.show()

loda_scores = model.decision_function(test_data)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(test_data[:,0], test_data[:,1], c=iforest_scores, cmap='viridis')
axs[0].set_title('IForest')

axs[1].scatter(test_data[:,0], test_data[:,1], c=dif_scores, cmap='viridis')
axs[1].set_title('DIF')

axs[2].scatter(test_data[:,0], test_data[:,1], c=loda_scores, cmap='viridis')
axs[2].set_title('LODA')

plt.show()

# ------------------------------------------------------------------------------------------

X, y = make_blobs(n_samples=500, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1, n_features=3, random_state=0)

model = iforest.IForest(contamination=0.02)
model.fit(X)

test_data = np.random.uniform(-10, 20, (1000, 3))

anomaly_scores = model.decision_function(test_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2], c=anomaly_scores, cmap='viridis')
plt.show()

model = DIF.DIF(contamination=0.02, hidden_neurons= [8, 4, 2])
model.fit(X)

anomaly_scores = model.decision_function(test_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2], c=anomaly_scores, cmap='viridis')
plt.show()

model = loda.LODA(contamination=0.02, n_bins=5)
model.fit(X)

anomaly_scores = model.decision_function(test_data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_data[:,0], test_data[:,1], test_data[:,2], c=anomaly_scores, cmap='viridis')
plt.show()