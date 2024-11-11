from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, _ = make_blobs(n_samples=500, centers=1, cluster_std=1.0, random_state=42)

import numpy as np

num_projections = 5
projection_vectors = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=num_projections)
#print(projection_vectors)

# projection_vectors = [
#  [ 0.04430052,  0.27374279],
#  [-0.69755666, -0.46701213],
#  [ 0.02998183,  0.57825051],
#  [-1.04221321, -0.46402995],
#  [-0.1404863,   0.29615486]
#  ]

projection_vectors = np.array([v / np.linalg.norm(v) for v in projection_vectors])

num_bins = 320
histograms = []
bin_edges = []
projections = []

for vector in projection_vectors:
    projected_data = data @ vector
    projections.append(projected_data)
    
    range_min, range_max = projected_data.min() - 500, projected_data.max() + 500
    
    histogram, bins = np.histogram(projected_data, bins=num_bins, range=(range_min, range_max), density=True)
    
    histograms.append(histogram)
    bin_edges.append(bins)

    # plt.hist(projected_data, bins=num_bins, range=(range_min, range_max), density=True)
    # plt.title('Histogram of Projected Data')
    # plt.xlabel('Projected Value')
    # plt.ylabel('Density')
    # plt.show()

anomaly_scores = []
for point in data:
    scores = []
    for i, vector in enumerate(projection_vectors):
        projected_value = point @ vector
        bin_idx = np.digitize(projected_value, bin_edges[i]) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        scores.append(histograms[i][bin_idx])
    anomaly_scores.append(np.mean(scores))


test_data = np.random.uniform(low=-3, high=3, size=(500, 2))
test_scores = []
for point in test_data:
    scores = []
    for i, vector in enumerate(projection_vectors):
        projected_value = point @ vector
        bin_idx = np.digitize(projected_value, bin_edges[i]) - 1
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
        scores.append(histograms[i][bin_idx])
    test_scores.append(np.mean(scores))

plt.scatter(test_data[:, 0], test_data[:, 1], c=test_scores, cmap='coolwarm', edgecolor='k')
plt.colorbar(label='Anomaly Score')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Anomaly Score Map for Test Data')
plt.show()
