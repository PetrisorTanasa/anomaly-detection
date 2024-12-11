#In this exercise we will use the mnist dataset from tensorflow.keras.datasets.mnist. After you load the dataset with tensorflow.keras.datasets.mnist.load data() you will normalize it by dividing with 255. In order to simulate anomalies, you will add some noise to the images with tensorflow.random.normal (multiplied by a factor of 0.35). You will use tensorflow.clip by value to keep the range of the pixels [0, 1].
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train[:1280]
X_test = X_test[:1280]
y_train = y_train[:1280]
y_test = y_test[:1280]

noise_factor = 0.35
X_train_noisy = X_train + noise_factor * tf.random.normal(shape=X_train.shape)
X_test_noisy = X_test + noise_factor * tf.random.normal(shape=X_test.shape)


X_train_noisy = tf.clip_by_value(X_train_noisy, clip_value_min=0, clip_value_max=1)
X_test_noisy = tf.clip_by_value(X_test_noisy, clip_value_min=0, clip_value_max=1)

class ConvAutoencoder(Model):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = Sequential([
            layers.Conv2D(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(4, (3, 3), activation='relu', strides=2, padding='same')
        ])
        # Decoder
        self.decoder = Sequential([
            layers.Conv2DTranspose(4, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2DTranspose(8, (3, 3), activation='relu', strides=2, padding='same'),
            layers.Conv2D(1, (28, 28), activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = ConvAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(X_train_noisy, X_train_noisy, epochs=10, batch_size=2, validation_data=(X_test_noisy, X_test_noisy))

reconstruction_train = autoencoder.predict(X_train)
reconstruction_error_train = np.mean(np.square(X_train - reconstruction_train), axis=(1, 2, 3))
threshold = np.mean(reconstruction_error_train) + np.std(reconstruction_error_train)


reconstruction_train_noisy = autoencoder.predict(X_train_noisy)
reconstruction_error_train_noisy = np.mean(np.square(X_train_noisy - reconstruction_train_noisy), axis=(1, 2, 3))
threshold = np.mean(reconstruction_error_train_noisy) + np.std(reconstruction_error_train_noisy)

reconstruction_test = autoencoder.predict(X_test)
reconstruction_error_test = np.mean(np.square(X_test - reconstruction_test), axis=(1, 2, 3))

y_train_pred = (reconstruction_error_train > threshold).astype(int)
y_test_pred = (reconstruction_error_test > threshold).astype(int)

from sklearn.metrics import balanced_accuracy_score
train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Autoencoder Balanced Accuracy - Train: {train_accuracy}, Test: {test_accuracy}")

plt.figure(figsize=(20, 10))

for i in range(5):
    print(reconstruction_error_train[i])
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 6)
    plt.imshow(X_train_noisy[i].numpy().reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 11)
    plt.imshow(reconstruction_train[i].reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 16)
    plt.imshow(reconstruction_train_noisy[i].reshape(28, 28))
    plt.axis('off')
plt.show()


denoising_autoencoder = ConvAutoencoder()
denoising_autoencoder.compile(optimizer='adam', loss='mse')

history = denoising_autoencoder.fit(X_train_noisy, X_train, epochs=10, batch_size=2, validation_data=(X_test_noisy, X_test))

reconstruction_train = denoising_autoencoder.predict(X_train_noisy)
reconstruction_error_train = np.mean(np.square(X_train - reconstruction_train), axis=(1, 2, 3))
threshold = np.mean(reconstruction_error_train) + np.std(reconstruction_error_train)


reconstruction_train_noisy = autoencoder.predict(X_train_noisy)
reconstruction_error_train_noisy = np.mean(np.square(X_train_noisy - reconstruction_train_noisy), axis=(1, 2, 3))
threshold = np.mean(reconstruction_error_train_noisy) + np.std(reconstruction_error_train_noisy)

reconstruction_test = denoising_autoencoder.predict(X_test_noisy)
reconstruction_error_test = np.mean(np.square(X_test - reconstruction_test), axis=(1, 2, 3))

y_train_pred = (reconstruction_error_train > threshold).astype(int)
y_test_pred = (reconstruction_error_test > threshold).astype(int)

train_accuracy = balanced_accuracy_score(y_train, y_train_pred)
test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

print(f"Denoising Autoencoder Balanced Accuracy - Train: {train_accuracy}, Test: {test_accuracy}")

plt.figure(figsize=(20, 10))

for i in range(5):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 6)
    plt.imshow(X_train_noisy[i].numpy().reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 11)
    plt.imshow(reconstruction_train[i].reshape(28, 28))
    plt.axis('off')

    plt.subplot(4, 5, i + 16)
    plt.imshow(reconstruction_train_noisy[i].reshape(28, 28))
    plt.axis('off')
plt.show()