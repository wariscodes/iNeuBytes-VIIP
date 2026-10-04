import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =========================
# 1. Load CIFAR-10 Dataset
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

# =========================
# 2. Train-Validation Split
# =========================
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print("Training set:", x_train.shape)
print("Validation set:", x_val.shape)
print("Test set:", x_test.shape)

# =========================
# 3. Normalize Images
# =========================
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Normalized pixel range:",
      x_train.min(), "to", x_train.max())

# =========================
# 4. Deeper CNN Architecture
# =========================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),

    # Original Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Original Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # NEW: Additional Conv Block
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# =========================
# 5. Compile Model
# =========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Total parameters:", model.count_params())

# =========================
# 6. Train Model
# =========================
start_time = time.time()

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64
)

training_time = time.time() - start_time

# =========================
# 7. Evaluate on Test Data
# =========================
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

train_val_gap = (train_accuracy - val_accuracy) * 100

print("\n===== DEPTH EXPERIMENT RESULTS =====")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Train-Validation Gap: {train_val_gap:.2f} percentage points")
print(f"Parameters: {model.count_params():,}")
print(f"Training Time: {training_time:.2f} seconds")

# =========================
# 8. Accuracy Curve
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Depth Experiment - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("Task1/results/depth_accuracy.png")
plt.show()

# =========================
# 9. Loss Curve
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Depth Experiment - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("Task1/results/depth_loss.png")
plt.show()