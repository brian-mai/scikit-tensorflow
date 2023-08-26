# setup keras in 3.3

# load MNIST dataset
from keras.datasets import mnist

# images = np arrays, labels = arr of digit 0-9
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))