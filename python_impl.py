import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import tensorflow_datasets as tfds
import psutil

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 2)  # Convert bytes to MB

# Load the data
(train_data, test_data), ds_info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)

# Preprocess the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [28, 28])  # Keep the image as 28x28 but cast to float32
    label = tf.one_hot(label, 10)
    return image, label

train_data = train_data.map(preprocess).batch(100).shuffle(60000)
test_data = test_data.map(preprocess).batch(10000)

# Define the RNN model
model = models.Sequential([
    layers.SimpleRNN(64, activation='tanh', input_shape=(28, 28)),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer=optimizers.SGD(learning_rate=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
train_log = []
epochs = 4

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} - Memory usage before training: {get_memory_usage():.2f} MB")
    model.fit(train_data, epochs=1)
    print(f"Epoch {epoch + 1} - Memory usage after training: {get_memory_usage():.2f} MB")

    train_loss, train_acc = model.evaluate(train_data)
    test_loss, test_acc = model.evaluate(test_data)

    train_log.append({
        'epoch': epoch + 1,
        'loss': train_loss,
        'acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    })

    print(f"Epoch {epoch + 1}: Train accuracy = {train_acc:.2f}, Test accuracy = {test_acc:.2f}")
    print(f"Epoch {epoch + 1} - Memory usage after evaluation: {get_memory_usage():.2f} MB")

# Evaluate the model
train_loss, train_acc = model.evaluate(train_data)
print(f"Final Train accuracy: {train_acc:.2f}")
print(f"Final Train Memory usage: {get_memory_usage():.2f} MB")

test_loss, test_acc = model.evaluate(test_data)
print(f"Final Test accuracy: {test_acc:.2f}")
print(f"Final Test Memory usage: {get_memory_usage():.2f} MB")
