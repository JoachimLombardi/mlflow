import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Chargement et prétraitement des données
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Construction du modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configuration de MLflow
mlflow.set_experiment("MNIST_Classification")

# Entraînement du modèle avec MLflow
with mlflow.start_run():
    model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)
    loss, accuracy = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
    mlflow.tensorflow.log_model(model, "mnist_model")
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)
