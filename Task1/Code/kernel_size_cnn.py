import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


# -----------------------------
# 1. Reproducibility
# -----------------------------
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# 2. Load CIFAR-10
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Original training data:", x_train.shape)
print("Original test data:", x_test.shape)


# -----------------------------
# 3. Normalize data
# -----------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# -----------------------------
# 4. Create validation split
# -----------------------------
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=SEED,
    stratify=y_train
)

print("Training data:", x_train.shape)
print("Validation data:", x_val.shape)
print("Test data:", x_test.shape)


# -----------------------------
# 5. Build CNN
# -----------------------------
# Main controlled change:
# First Conv2D kernel changed from 3x3 to 5x5

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (5, 5), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(10, activation="softmax")
])


# -----------------------------
# 6. Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 7. Display model information
# -----------------------------
model.summary()

print("\nTotal parameters:", model.count_params())


# -----------------------------
# 8. Train model
# -----------------------------
start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

training_time = time.time() - start_time


# -----------------------------
# 9. Evaluate model
# -----------------------------
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

train_val_gap = (train_accuracy - val_accuracy) * 100


# -----------------------------
# 10. Print results
# -----------------------------
print("\n========== KERNEL SIZE RESULTS ==========")

print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Train-Val Gap: {train_val_gap:.2f} percentage points")
print(f"Parameters: {model.count_params():,}")
print(f"Training Time: {training_time:.2f} seconds")


# -----------------------------
# 11. Accuracy graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Kernel Size Change - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("Task1/results/kernel_size_accuracy.png")
plt.show()


# -----------------------------
# 12. Loss graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Kernel Size Change - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("Task1/results/kernel_size_loss.png")
plt.show()