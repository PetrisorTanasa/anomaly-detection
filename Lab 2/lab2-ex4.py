import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

# Step 1: Load the dataset
data = scipy.io.loadmat("/Users/ptanasa/Desktop/Anomaly Detection/cardio.mat")  # Replace with actual path

X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors_values = np.arange(30, 121, 10)
train_scores = []
test_scores = []

for n_neighbors in n_neighbors_values:
    knn = KNN(n_neighbors=n_neighbors)
    knn.fit(X_train)
    
    train_scores.append(knn.decision_scores_)
    test_scores.append(knn.decision_function(X_test))
    
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    
    ba_train = balanced_accuracy_score(y_train, y_train_pred)
    ba_test = balanced_accuracy_score(y_test, y_test_pred)
    
    print(f"n_neighbors={n_neighbors}: Train BA={ba_train:.4f}, Test BA={ba_test:.4f}")

train_scores = np.array(train_scores)
test_scores = np.array(test_scores)

train_scores_norm, test_scores_norm = standardizer(train_scores.T, test_scores.T)

avg_train_scores = average(train_scores_norm)
avg_test_scores = average(test_scores_norm)

max_train_scores = maximization(train_scores_norm)
max_test_scores = maximization(test_scores_norm)

contamination_rate = 0.1
threshold_avg = np.quantile(avg_train_scores, 1 - contamination_rate)
threshold_max = np.quantile(max_train_scores, 1 - contamination_rate)

y_pred_avg = (avg_test_scores > threshold_avg).astype(int)
y_pred_max = (max_test_scores > threshold_max).astype(int)

ba_avg = balanced_accuracy_score(y_test, y_pred_avg)
ba_max = balanced_accuracy_score(y_test, y_pred_max)

print(f"Balanced Accuracy for Average strategy: {ba_avg:.4f}")
print(f"Balanced Accuracy for Maximization strategy: {ba_max:.4f}")
