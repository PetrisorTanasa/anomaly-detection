import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mean = [5,10,2]
cov = [[3,2,2],[2,10,1],[2,1,2]]
data = np.random.multivariate_normal(mean, cov, 500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.show()

data_centered = data - np.mean(data, axis=0)
cov_matrix = np.cov(data_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)
print("cov_matrix: ", cov_matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_centered[:,0], data_centered[:,1], data_centered[:,2])
plt.show()

sorted_eigenvalues = np.sort(eigenvalues)[::-1]
cumulative_variance = np.cumsum(sorted_eigenvalues)
print("Sorted eigenvalues: ", sorted_eigenvalues)
print("Cumulative variance: ", cumulative_variance)

fig, ax = plt.subplots(2)
ax[0].step(range(1, len(sorted_eigenvalues)+1), cumulative_variance)
ax[1].bar(range(1, len(sorted_eigenvalues)+1), sorted_eigenvalues)
plt.show()

projected_data = data_centered @ eigenvectors

pca_3rd_component = projected_data[:, 2]
contamination_rate = 0.1
threshold_3rd = np.quantile(pca_3rd_component, 1 - contamination_rate)
outliers_3rd = pca_3rd_component > threshold_3rd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=~outliers_3rd, label='Inliers')
ax.scatter(data[outliers_3rd, 0], data[outliers_3rd, 1], data[outliers_3rd, 2], c='red', label='Outliers')
ax.set_title("Outliers based on 3rd Principal Component")
ax.legend()
plt.show()

pca_2nd_component = projected_data[:, 1]
threshold_2nd = np.quantile(pca_2nd_component, 1 - contamination_rate)
outliers_2nd = pca_2nd_component > threshold_2nd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=~outliers_2nd, label='Inliers')
ax.scatter(data[outliers_2nd, 0], data[outliers_2nd, 1], data[outliers_2nd, 2], c='green', label='Outliers')
ax.set_title("Outliers based on 2nd Principal Component")
ax.legend()
plt.show()

centroid = np.mean(projected_data, axis=0)
normalized_distances = np.abs((projected_data - centroid) / np.std(projected_data, axis=0))
total_distance = np.sum(normalized_distances, axis=1)

contamination_rate = 0.1
threshold = np.quantile(total_distance, 1 - contamination_rate)
outliers = total_distance > threshold

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=~outliers, label='Inliers')
ax.scatter(data[outliers, 0], data[outliers, 1], data[outliers, 2], c='red', label='Outliers')
ax.set_title("Outliers based on Normalized Distance in PCA Space")
ax.legend()
plt.show()