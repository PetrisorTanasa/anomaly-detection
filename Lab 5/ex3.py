from scipy.io import loadmat
from sklearn.model_selection import train_test_split

data = loadmat('/Users/ptanasa/Desktop/Anomaly Detection/Lab 3/shuttle.mat')
X = data['X']
y = data['y'].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# use min-max Normalization to have it in [0-1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras

class AutoEncoder(keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(3, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(9, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AutoEncoder()
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=1024, validation_data=(X_test, X_test))

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

import numpy as np
from sklearn.metrics import balanced_accuracy_score

reconstruction_train = autoencoder.predict(X_train)
reconstruction_test = autoencoder.predict(X_test)

reconstruction_error_train = np.mean(np.square(X_train - reconstruction_train), axis=1)
reconstruction_error_test = np.mean(np.square(X_test - reconstruction_test), axis=1)

contamination_rate = np.mean(y_train)

threshold = np.quantile(reconstruction_error_train, 1 - contamination_rate)

y_train_pred = (reconstruction_error_train > threshold).astype(int)
y_test_pred = (reconstruction_error_test > threshold).astype(int)

train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Autoencoder Balanced Accuracy - Train: {train_accuracy}, Test: {test_accuracy}")