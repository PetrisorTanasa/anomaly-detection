#In this exercise we will see the limitations of distance based algorithms like KNN. Specifically, we will observe how KNN behaves when our data clusters have different densities and how pyod.models.lof.LOF solves the problem by considering the variations of the local densities of the datapoints. First, generate 2 clusters (200 and 100 samples respectively) with 2-dimensional samples using (-10,-10) and (10, 10) as centers, 2 and 6 as standard deviations using sklearn.datasets.make blobs() function. Then, fit KNN and LOF with the generated data using a small contamination rate (0.07) and find the predicted labels. Use 2 subplots to plot (using different colors for inliers and outliers) the 2 clusters and observe how the 2 models behave for different n neighbors.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import pyod

X1, y1 = make_blobs(n_samples=200, centers=[(-10, -10)], cluster_std=2, n_features=2, random_state=42)
X2, y2 = make_blobs(n_samples=100, centers=[(10, 10)], cluster_std=6, n_features=2, random_state=42)

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)


for i in range(1,30,3):
    knn = KNN(n_neighbors=i, contamination=0.07)
    knn.fit(X)

    lof = LOF(n_neighbors=i, contamination=0.07)
    lof.fit(X)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    predictions = knn.predict(X)
    plt.scatter(X[predictions == 0, 0], X[predictions == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], c='red', label='Outliers', alpha=0.7)
    plt.title('Predicted labels for KNN')

    # plt.figure(figsize=(6, 6))
    # plt.subplot(1, 2, 3)
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Normal', alpha=0.7)
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Outliers', alpha=0.7)
    # plt.title('Ground truth labels')

    plt.subplot(1, 2, 2)
    predictions = lof.predict(X)
    plt.scatter(X[predictions == 0, 0], X[predictions == 0, 1], c='blue', label='Normal', alpha=0.7)
    plt.scatter(X[predictions == 1, 0], X[predictions == 1, 1], c='red', label='Outliers', alpha=0.7)
    plt.title('Predicted labels for LOF')

    plt.show()