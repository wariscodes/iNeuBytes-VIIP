import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -----------------------------
# 1. Reproducibility
# -----------------------------
SEED = 42

tf.random.set_seed(SEED)


# -----------------------------
# 2. Load CIFAR-10 dataset
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
# 4. Remove extra label dimension
# -----------------------------
y_train = y_train.flatten()
y_test = y_test.flatten()


# -----------------------------
# 5. Create validation split
# -----------------------------
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
# 6. Selected data augmentation
#    Proven useful: Horizontal Flip
# -----------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal")
], name="data_augmentation")


# -----------------------------
# 7. Build final CNN
#
# Selected techniques:
# - Horizontal Flip
# - Increased Depth
# - Dropout
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    data_augmentation,

    # First convolution block
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Additional depth
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification layers
    layers.Flatten(),
    layers.Dense(128, activation="relu"),

    # Proven regularization technique
    layers.Dropout(0.5),

    layers.Dense(10, activation="softmax")
])


# -----------------------------
# 8. Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 9. Display model information
# -----------------------------
model.summary()

print("\nTotal parameters:", model.count_params())


# -----------------------------
# 10. Train model
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
# 11. Evaluate on test data
# -----------------------------
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


# -----------------------------
# 12. Calculate final metrics
# -----------------------------
train_accuracy = history.history["accuracy"][-1]
val_accuracy = history.history["val_accuracy"][-1]

train_val_gap = (train_accuracy - val_accuracy) * 100

total_params = model.count_params()


# -----------------------------
# 13. Print final results
# -----------------------------
print("\n" + "=" * 55)
print("FINAL CNN RESULTS")
print("=" * 55)

print(f"Train Accuracy       : {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy  : {val_accuracy * 100:.2f}%")
print(f"Test Accuracy        : {test_accuracy * 100:.2f}%")
print(f"Test Loss            : {test_loss:.4f}")
print(f"Train-Val Gap        : {train_val_gap:.2f} percentage points")
print(f"Parameters           : {total_params:,}")
print(f"Training Time        : {training_time:.2f} seconds")

print("=" * 55)


# -----------------------------
# 14. Generate predictions
# -----------------------------
y_pred_probabilities = model.predict(
    x_test,
    verbose=0
)

y_pred = y_pred_probabilities.argmax(axis=1)


# -----------------------------
# 15. Class names
# -----------------------------
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# -----------------------------
# 16. Create confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_test,
    y_pred
)


# -----------------------------
# 17. Display and save confusion matrix
# -----------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

plt.title("Final CNN - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "Task1/results/final_confusion_matrix.png"
)

plt.close()


# -----------------------------
# 18. Find most confused classes
# -----------------------------
confusion_pairs = []

for actual_class in range(len(class_names)):
    for predicted_class in range(len(class_names)):

        if actual_class != predicted_class:

            confusion_pairs.append(
                (
                    cm[actual_class][predicted_class],
                    class_names[actual_class],
                    class_names[predicted_class]
                )
            )


confusion_pairs.sort(reverse=True)


print("\n" + "=" * 55)
print("MOST CONFUSED CLASS PAIRS")
print("=" * 55)

for count, actual, predicted in confusion_pairs[:5]:

    print(
        f"Actual {actual} -> Predicted {predicted}: {count}"
    )

print("=" * 55)


# -----------------------------
# 19. Save accuracy graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Final CNN - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    "Task1/results/final_accuracy.png"
)

plt.close()


# -----------------------------
# 20. Save loss graph
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Final CNN - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "Task1/results/final_loss.png"
)

plt.close()


print("\nGraphs saved successfully.")
print("Confusion matrix saved successfully.")