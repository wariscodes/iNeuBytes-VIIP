import os
import random
import time

import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. Configuration
# ============================================================

VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 200

# Experiment variable:
EMBEDDING_DIM = 64

LSTM_UNITS = 64
DROPOUT_RATE = 0.2

EPOCHS = 5
BATCH_SIZE = 64


# ============================================================
# 3. Load Dataset
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "Dataset", "aclImdb", "train")
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "aclImdb", "test")


def load_reviews(directory):
    texts = []
    labels = []

    for label_name, label in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(directory, label_name)

        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                texts.append(text)
                labels.append(label)

    return texts, np.array(labels)


print("Loading dataset...")

train_texts, train_labels = load_reviews(TRAIN_DIR)
test_texts, test_labels = load_reviews(TEST_DIR)

print(f"Training samples: {len(train_texts)}")
print(f"Testing samples: {len(test_texts)}")


# ============================================================
# 4. Tokenization
# ============================================================

print("\nTokenizing text...")

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_texts)

X_train_full = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train_full = tf.keras.preprocessing.sequence.pad_sequences(
    X_train_full,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)

X_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test,
    maxlen=SEQUENCE_LENGTH,
    padding="post",
    truncating="post"
)


# ============================================================
# 5. Train / Validation Split
# ============================================================

validation_size = 5000

X_train = X_train_full[:-validation_size]
X_val = X_train_full[-validation_size:]

y_train = train_labels[:-validation_size]
y_val = train_labels[-validation_size:]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


# ============================================================
# 6. Build LSTM Model
# ============================================================

print("\nBuilding LSTM model...")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.LSTM(LSTM_UNITS),

    tf.keras.layers.Dropout(DROPOUT_RATE),

    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.build(input_shape=(None, SEQUENCE_LENGTH))

model.summary()


# ============================================================
# 7. Train Model
# ============================================================

print("\nStarting training...")

start_time = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

training_time = time.time() - start_time


# ============================================================
# 8. Training Metrics
# ============================================================

train_loss, train_accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)


# ============================================================
# 9. Test Prediction
# ============================================================

test_probabilities = model.predict(
    X_test,
    verbose=0
)

test_predictions = (test_probabilities >= 0.5).astype(int).flatten()


test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions
)

test_recall = recall_score(
    test_labels,
    test_predictions
)

test_f1 = f1_score(
    test_labels,
    test_predictions
)


# ============================================================
# 10. Train-Validation Gap
# ============================================================

train_val_gap = train_accuracy - val_accuracy


# ============================================================
# 11. Model Parameters
# ============================================================

total_params = model.count_params()


# ============================================================
# 12. Final Results
# ============================================================

print("\n" + "=" * 60)
print("LSTM EMBEDDING DIMENSION EXPERIMENT RESULTS")
print("=" * 60)

print(f"Embedding Dimension : {EMBEDDING_DIM}")
print(f"Sequence Length     : {SEQUENCE_LENGTH}")
print(f"LSTM Units          : {LSTM_UNITS}")
print(f"Dropout             : {DROPOUT_RATE}")
print(f"Vocabulary Size     : {VOCAB_SIZE}")

print("\nDataset:")
print(f"Training samples    : {len(X_train)}")
print(f"Validation samples  : {len(X_val)}")
print(f"Test samples        : {len(X_test)}")

print("\nModel:")
print(f"Total Parameters    : {total_params:,}")

print("\nPerformance:")
print(f"Train Accuracy      : {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")
print(f"Test Accuracy       : {test_accuracy * 100:.2f}%")
print(f"Precision           : {test_precision * 100:.2f}%")
print(f"Recall              : {test_recall * 100:.2f}%")
print(f"F1 Score            : {test_f1 * 100:.2f}%")

print("\nOverfitting:")
print(f"Train-Val Gap       : {train_val_gap * 100:.2f} percentage points")

print("\nTraining:")
print(f"Training Time       : {training_time:.2f} seconds")

print("=" * 60)