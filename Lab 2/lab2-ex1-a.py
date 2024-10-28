
import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
import pyod
import sklearn.metrics as metrics

a = 2
b = 1

def generate_data(miu, sigma, var_on = "none"):
    X = np.random.rand(500, 1)
    y = a * X + b
    
    if var_on == 'x':
        X = X + np.random.normal(miu, sigma, (500, 1))
    elif var_on == 'y':
        y = y + np.random.normal(miu, sigma, (500, 1))
    elif var_on == 'both':
        X = X + np.random.normal(miu, sigma, (500, 1))
        y = y + np.random.normal(miu, sigma, (500, 1))
    
    return X, y

X_train_regular, y_train_regular = generate_data(0, 0.1)
X_train_high_variance_x, y_train_high_variance_x = generate_data(0, 0.5, "x")
X_train_high_variance_y, y_train_high_variance_y = generate_data(0, 0.5, "y")
X_train_high_variance_xy, y_train_high_variance_xy = generate_data(0, 1,"both")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# H = X(X^TX)^-1X^T

def calculate_leverage(X):
    X = np.array(X)
    X = X.reshape(-1, 1)
    poly = PolynomialFeatures(degree=1)
    X = poly.fit_transform(X)
    regression = LinearRegression().fit(X, X)
    predictions = regression.predict(X)
    mse = mean_squared_error(X, predictions)
    leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    return leverage, mse

leverage_regular, mse_regular = calculate_leverage(X_train_regular)
leverage_high_variance_x, mse_high_variance_x = calculate_leverage(X_train_high_variance_x)
leverage_high_variance_y, mse_high_variance_y = calculate_leverage(X_train_high_variance_y)
leverage_high_variance_xy, mse_high_variance_xy = calculate_leverage(X_train_high_variance_xy)
plt.figure(figsize=(15, 15))
plt.scatter(X_train_regular, y_train_regular, c='blue', label='Normal', alpha=0.7)
plt.scatter(X_train_high_variance_x, y_train_high_variance_x, c='red', label='High variance x', alpha=0.7)
plt.scatter(X_train_high_variance_y, y_train_high_variance_y, c='green', label='High variance y', alpha=0.7)
plt.scatter(X_train_high_variance_xy, y_train_high_variance_xy, c='yellow', label='High variance xy', alpha=0.7)

plt.scatter(X_train_regular[np.argmax(leverage_regular)], y_train_regular[np.argmax(leverage_regular)], c='purple', label='Normal Highest Leverage', alpha=0.7)
plt.scatter(X_train_high_variance_x[np.argmax(leverage_high_variance_x)], y_train_high_variance_x[np.argmax(leverage_high_variance_x)], c='purple', label='High variance x Highest Leverage', alpha=0.7)
plt.scatter(X_train_high_variance_y[np.argmax(leverage_high_variance_y)], y_train_high_variance_y[np.argmax(leverage_high_variance_y)], c='purple', label='High variance y Highest Leverage', alpha=0.7)
plt.scatter(X_train_high_variance_xy[np.argmax(leverage_high_variance_xy)], y_train_high_variance_xy[np.argmax(leverage_high_variance_xy)], c='purple', label='High variance xy Highest Leverage', alpha=0.7)

plt.title('Regular Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.scatter(X_train_regular, y_train_regular, c='blue', label='Normal', alpha=0.7)
plt.scatter(X_train_regular[np.argmax(leverage_regular)], y_train_regular[np.argmax(leverage_regular)], c='purple', label='Normal Highest Leverage', alpha=0.7)
plt.title('Normal Data')

plt.subplot(2, 2, 2)
plt.scatter(X_train_high_variance_x, y_train_high_variance_x, c='red', label='High variance x', alpha=0.7)
plt.scatter(X_train_high_variance_x[np.argmax(leverage_high_variance_x)], y_train_high_variance_x[np.argmax(leverage_high_variance_x)], c='purple', label='High variance x Highest Leverage', alpha=0.7)
plt.title('High variance x Data')

plt.subplot(2, 2, 3)
plt.scatter(X_train_high_variance_y, y_train_high_variance_y, c='green', label='High variance y', alpha=0.7)
plt.scatter(X_train_regular[np.argmax(leverage_regular)], y_train_regular[np.argmax(leverage_regular)], c='purple', label='Normal Highest Leverage', alpha=0.7)
plt.title('High variance y Data')

plt.subplot(2, 2, 4)
plt.scatter(X_train_high_variance_xy, y_train_high_variance_xy, c='yellow', label='High variance xy', alpha=0.7)
plt.scatter(X_train_high_variance_xy[np.argmax(leverage_high_variance_xy)], y_train_high_variance_xy[np.argmax(leverage_high_variance_xy)], c='purple', label='High variance xy Highest Leverage', alpha=0.7)
plt.title('High variance xy Data')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

print("MSE Regular: " + str(mse_regular))
print("MSE High Variance X: " + str(mse_high_variance_x))
print("MSE High Variance Y: " + str(mse_high_variance_y))
print("MSE High Variance XY: " + str(mse_high_variance_xy))

print("Leverage Regular: " + str(np.max(leverage_regular)))
print("Leverage High Variance X: " + str(np.max(leverage_high_variance_x)))
print("Leverage High Variance Y: " + str(np.max(leverage_high_variance_y)))
print("Leverage High Variance XY: " + str(np.max(leverage_high_variance_xy)))

def generate_data_b(miu, sigma, var_on="none", n_points=500):

    X1 = np.random.rand(n_points, 1)
    X2 = np.random.rand(n_points, 1)
    
    y = 3 * X1 + 2 * X2 + b
    
    if var_on == 'x1':
        X1 = X1 + np.random.normal(miu, sigma, (n_points, 1))
    elif var_on == 'x2':
        X2 = X2 + np.random.normal(miu, sigma, (n_points, 1))
    elif var_on == 'y':
        y = y + np.random.normal(miu, sigma, (n_points, 1))
    elif var_on == 'both':
        X1 = X1 + np.random.normal(miu, sigma, (n_points, 1))
        X2 = X2 + np.random.normal(miu, sigma, (n_points, 1))
        y = y + np.random.normal(miu, sigma, (n_points, 1))
    
    return X1, X2, y

X1, X2, y = generate_data_b(0, 0.1)
X1_high_variance_x, X2_high_variance_x, y_high_variance_x = generate_data_b(0, 0.5, "x")
X1_high_variance_y, X2_high_variance_y, y_high_variance_y = generate_data_b(0, 0.5, "y")
X1_high_variance_xy, X2_high_variance_xy, y_high_variance_xy = generate_data_b(0, 1, "both")

def calculate_leverage_b(X1, X2):
    X = np.concatenate((X1, X2), axis=1)
    X = np.array(X)
    X = X.reshape(-1, 2)
    poly = PolynomialFeatures(degree=1)
    X = poly.fit_transform(X)
    regression = LinearRegression().fit(X, X)
    predictions = regression.predict(X)
    mse = mean_squared_error(X, predictions)
    leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    return leverage, mse

leverage_regular_b, mse_regular_b = calculate_leverage_b(X1, X2)
leverage_high_variance_x_b, mse_high_variance_x_b = calculate_leverage_b(X1_high_variance_x, X2_high_variance_x)
leverage_high_variance_y_b, mse_high_variance_y_b = calculate_leverage_b(X1_high_variance_y, X2_high_variance_y)
leverage_high_variance_xy_b, mse_high_variance_xy_b = calculate_leverage_b(X1_high_variance_xy, X2_high_variance_xy)

print("MSE Regular: " + str(mse_regular_b))
print("MSE High Variance X: " + str(mse_high_variance_x_b))
print("MSE High Variance Y: " + str(mse_high_variance_y_b))
print("MSE High Variance XY: " + str(mse_high_variance_xy_b))



fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, c='blue', label='Normal', alpha=0.7)
ax.scatter(X1[np.argmax(leverage_regular_b)], X2[np.argmax(leverage_regular_b)], y[np.argmax(leverage_regular_b)], c='purple', label='Normal Highest Leverage', alpha=0.7)

plt.legend()
plt.show()
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1_high_variance_x, X2_high_variance_x, y_high_variance_x, c='red', label='High variance x', alpha=0.7)
ax.scatter(X1_high_variance_x[np.argmax(leverage_high_variance_x_b)], X2_high_variance_x[np.argmax(leverage_high_variance_x_b)], y_high_variance_x[np.argmax(leverage_high_variance_x_b)], c='purple', label='High variance x Highest Leverage', alpha=0.7)

plt.legend()
plt.show()
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1_high_variance_y, X2_high_variance_y, y_high_variance_y, c='green', label='High variance y', alpha=0.7)
ax.scatter(X1_high_variance_y[np.argmax(leverage_high_variance_y_b)], X2_high_variance_y[np.argmax(leverage_high_variance_y_b)], y_high_variance_y[np.argmax(leverage_high_variance_y_b)], c='purple', label='High variance y Highest Leverage', alpha=0.7)

plt.legend()
plt.show()
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1_high_variance_xy, X2_high_variance_xy, y_high_variance_xy, c='yellow', label='High variance xy', alpha=0.7)
ax.scatter(X1_high_variance_xy[np.argmax(leverage_high_variance_xy_b)], X2_high_variance_xy[np.argmax(leverage_high_variance_xy_b)], y_high_variance_xy[np.argmax(leverage_high_variance_xy_b)], c='purple', label='High variance xy Highest Leverage', alpha=0.7)

plt.legend()
plt.show()